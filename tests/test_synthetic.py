import torch

from vsa_embed.compose import FrameComposer, uniform_bundle
from vsa_embed.synthetic import make_frame_teacher, make_polysemy_teacher


def test_teachers_do_not_touch_the_global_rng() -> None:
    torch.manual_seed(7); expected = torch.rand(2)
    torch.manual_seed(7)
    make_frame_teacher(concept_count=60, holdout_concepts=10, seed=1)
    make_polysemy_teacher(concept_count=80, holdout_concepts=10, seed=1)
    torch.testing.assert_close(torch.rand(2), expected)


def test_teachers_are_deterministic_per_seed() -> None:
    a = make_frame_teacher(concept_count=60, holdout_concepts=10, seed=3)
    b = make_frame_teacher(concept_count=60, holdout_concepts=10, seed=3)
    torch.testing.assert_close(a.samples["train"]["targets"], b.samples["train"]["targets"])


def test_static_teacher_is_exactly_an_hrr_bundle() -> None:
    data = make_frame_teacher(concept_count=60, holdout_concepts=10, weighting="static", seed=2)
    composer = FrameComposer(data.schedule, data.atomics.shape[0], data.roles.shape[0], data.atomics.shape[1])
    with torch.no_grad():
        composer.atomics.copy_(data.atomics)
        composer.transform.roles.copy_(data.roles)
    ids = data.samples["train"]["concepts"]
    torch.testing.assert_close(uniform_bundle(composer, ids), data.samples["train"]["targets"], atol=1e-5, rtol=1e-4)


def test_holdouts_are_composition_disjoint_and_supported() -> None:
    data = make_frame_teacher(concept_count=120, holdout_concepts=20, seed=4)
    assert set(data.train_concepts.tolist()).isdisjoint(data.test_concepts.tolist())
    assert set(data.train_contexts.tolist()).isdisjoint(data.test_contexts.tolist())
    edges = lambda ids: {int(data.schedule.fillers[e]) for i in ids.tolist()
                         for e in range(int(data.schedule.offsets[i]), int(data.schedule.offsets[i + 1]))}
    assert edges(data.test_concepts) <= edges(data.train_concepts)


def test_contextual_teacher_varies_with_context_and_preserves_mass() -> None:
    data = make_frame_teacher(concept_count=80, holdout_concepts=10, samples_per_concept=6, seed=5)
    train = data.samples["train"]
    segments = torch.repeat_interleave(torch.arange(train["concepts"].numel()), data.schedule.degrees[train["concepts"]])
    mass = torch.zeros(train["concepts"].numel()).index_add(0, segments, train["weights"])
    torch.testing.assert_close(mass, data.schedule.degrees[train["concepts"]].float())
    same_concept = train["targets"][:6]
    assert float((same_concept @ same_concept.T).min()) < 0.99


def test_polysemy_teacher_marks_hidden_senses() -> None:
    data = make_polysemy_teacher(concept_count=200, holdout_concepts=20, seed=6)
    poly_edges = torch.isin(data.schedule.fillers, data.polysemous_atomics)
    assert bool((data.edge_atomic_sense[poly_edges] >= 0).all())
    assert bool((data.edge_atomic_sense[~poly_edges] == -1).all())
    assert set(data.edge_atomic_sense[poly_edges].tolist()) == {0, 1}
