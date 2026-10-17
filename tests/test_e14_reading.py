"""E14 RD-4 / RD-7 (`e14_reading recall`) on the tiny E9 world (CPU): noisy frames, recall text, the implicit route
reproducing E11's own persistence scores, the explicit routes, a host without a channel, and the pooled report."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_e9 import _ok, items, world  # noqa: E402,F401  (module fixtures re-used)
from test_e11_read_to_learn import _heldout_set  # noqa: E402

from vsa_embed.experiments import e5_common as common  # noqa: E402
from vsa_embed.experiments import e9_ontology_edit as edit  # noqa: E402
from vsa_embed.experiments import e11_read_to_learn as e11  # noqa: E402
from vsa_embed.experiments import e14_reading as rd  # noqa: E402

pytestmark = pytest.mark.skipif(not _ok(), reason="gpt2 tokenizer or WordNet not available")
QUIET = lambda message: None  # noqa: E731


def test_noisy_frames_are_seeded_and_typed() -> None:
    pools = {0: np.array([1, 2, 3]), 1: np.array([7])}
    frame = [(0, 1), (0, 2), (1, 7)]
    assert rd.noisy_frame(frame, 0.0, pools, seed=0, key="c") == frame
    full = rd.noisy_frame(frame, 1.0, pools, seed=0, key="c")
    assert [r for r, _ in full] == [0, 0, 1] and full[0][1] != 1 and full[1][1] != 2 and full[2] == (1, 7)   # lone filler kept
    assert all(a in {1, 2, 3} for r, a in full if r == 0)
    assert rd.noisy_frame(frame, 0.5, pools, seed=3, key="c") == rd.noisy_frame(frame, 0.5, pools, seed=3, key="c")


def test_default_conditions_follow_the_host() -> None:
    plain = rd.default_conditions(False, False, (0.1,))
    assert "implicit:oracle" not in plain and "explicit-sym:oracle" in plain and "noisy-explicit:0.1" in plain
    full = rd.default_conditions(True, True, (0.1,))
    assert {"implicit:linker", "both:oracle", "noisy-implicit:0.1", "explicit-vsa:oracle"} <= set(full)


@pytest.fixture(scope="module")
def heldout(world, tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("e14rd")
    items_dir = _heldout_set(world, root)
    e11_out = root / "e11"
    e11.main(["evaluate", "--run", str(world["runs"]["C5"]), "--items", str(items_dir), "--output", str(e11_out),
              "--methods", "frames,persistence", "--readers", "oracle,stated,typeprior,linker,none", "--styles", "prose",
              "--device", "cpu", "--batch-size", "8", "--resamples", "50", "--smoke"])
    return {"items": items_dir, "e11": e11_out, "root": root}


def test_recall_routes_on_a_compose_host(world, heldout) -> None:
    out = heldout["root"] / "rd4-c5"
    rd.main(["recall", "--run", str(world["runs"]["C5"]), "--items", str(heldout["items"]), "--output", str(out),
             "--frames", str(heldout["e11"] / "frames.jsonl"), "--style", "prose", "--noise", "0.5",
             "--device", "cpu", "--batch-size", "8", "--resamples", "50", "--smoke"])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["compose"] and summary["vsa"]
    assert {"none", "implicit:oracle", "explicit-sym:oracle", "explicit-vsa:oracle", "both:oracle", "definition",
            "noisy-explicit:0.5", "noisy-implicit:0.5"} <= set(summary["conditions"])
    # The implicit route is E11's own route: same items, same frames, same values as E11's persistence scores.
    e11_means = json.loads((heldout["e11"] / "summary.json").read_text())["items"]["means"]
    for reader in ("oracle", "typeprior", "none"):
        ours = summary["items"]["means"]["none" if reader == "none" else f"implicit:{reader}"]["property"][0]
        assert ours == pytest.approx(e11_means[f"prose|{reader}"]["property"][0], abs=1e-9)
    # Recall text: symbolic lines carry confidence 1.00; the decoded store recovers a share of the frame's edges.
    assert "(1.00)" in summary["context_samples"]["explicit-sym:oracle"]
    share = summary["diagnostics"]["explicit-vsa:oracle"]["decode_recovered"]
    assert share is not None and 0.0 <= share <= 1.0
    assert summary["diagnostics"]["noisy-explicit:0.5"]["corrupted_share"] > 0
    assert summary["context_tokens"]["none"] == 0.0 and summary["context_tokens"]["explicit-sym:oracle"] > 0
    assert {r["a"] for r in summary["items"]["primary"]} == {"explicit-sym:oracle", "explicit-sym:stated"}
    assert all("p_holm" in r for r in summary["items"]["primary"])
    assert (out / "report.md").read_text().startswith("**SMOKE TEST")
    rows = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    assert {r["condition"] for r in rows} == set(summary["conditions"])
    heldout["c5_out"] = out


def test_recall_on_a_host_without_a_channel_and_pooled_report(world, heldout) -> None:
    out = heldout["root"] / "rd4-c0p"
    rd.main(["recall", "--run", str(world["runs"]["C0p"]), "--items", str(heldout["items"]), "--output", str(out),
             "--frames", str(heldout["e11"] / "frames.jsonl"), "--style", "prose", "--noise", "0.3",
             "--device", "cpu", "--batch-size", "8", "--resamples", "50", "--smoke"])
    summary = json.loads((out / "summary.json").read_text())
    assert not summary["compose"] and not summary["vsa"]
    assert not any(c.startswith(("implicit", "both", "explicit-vsa", "noisy-implicit")) for c in summary["conditions"])
    with pytest.raises(SystemExit):
        rd.main(["recall", "--run", str(world["runs"]["C0p"]), "--items", str(heldout["items"]), "--output",
                 str(heldout["root"] / "bad"), "--frames", str(heldout["e11"] / "frames.jsonl"), "--style", "prose",
                 "--conditions", "none,implicit:oracle", "--device", "cpu", "--smoke"])
    report = heldout["root"] / "report"
    rd.main(["recall-report", "--runs", str(heldout.get("c5_out", out)), str(out), str(heldout["root"] / "missing"),
             "--output", str(report), "--resamples", "50"])
    data = json.loads((report / "report.json").read_text())
    assert data["missing"] == [str(heldout["root"] / "missing")]
    assert {g["model"] for g in data["groups"]} >= {Path(world["runs"]["C0p"]).name.rsplit("-s", 1)[0]}
    assert "pooled report" in (report / "report.md").read_text()


def test_recall_text_writes_the_track_statement(world) -> None:
    ctx = e11.make_context("wordnet", world["ontology"], world["table"], edit.WordNetLexicon())
    frame = ctx.frame(next(e for e in range(len(world["table"].entry_concepts)) if ctx.frame(e)))
    text = rd.RecallText(world["ontology"], ctx.lexicon).symbolic("blicket", frame)
    assert text.startswith("recall(blicket):") and text.count("\n") == len(frame) and "(1.00)" in text
    assert common.json_ready({"x": np.float32(1.0)}) == {"x": 1.0}


# -- RD-3 ----------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_host(tmp_path_factory):
    import transformers
    from vsa_embed.logic.hosts import HostLM
    path = tmp_path_factory.mktemp("tinyhost")
    transformers.AutoTokenizer.from_pretrained("gpt2").save_pretrained(path)
    torch_model = transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=50257, n_positions=128, n_embd=32, n_layer=1, n_head=2))
    torch_model.save_pretrained(path)
    return HostLM(str(path), device="cpu", batch_size=4)


def test_host_scores_and_choices_are_consistent(tiny_host) -> None:
    s = tiny_host.score(["The cat", "The cat"], [" sat", " sat down"])
    assert s.tokens.tolist() == [1, 2] and np.all(np.isfinite(s.logprob))
    whole = tiny_host.score(["The cat sat"], [" down"]).logprob[0]
    assert s.logprob[1] == pytest.approx(s.logprob[0] + whole, abs=1e-4)      # chain rule over the continuation
    best, values = tiny_host.choose("The cat", [" sat", " sat down"], null="")
    assert best in (0, 1) and len(values) == 2
    assert isinstance(tiny_host.generate(["The cat"], max_new_tokens=3)[0], str)


def test_trigram_index_and_spans() -> None:
    index = rd.TrigramIndex({1: "antioxidant", 2: "fatty acid", 3: "carbon"})
    assert index.top("anti-oxidant", min_score=0.5)[0][0] == 1
    assert index.top("zzz") == []
    spans = rd.word_spans("a thia fatty acid", max_words=2)
    assert (2, 6) in spans and (7, 17) in spans and all(e > s for s, e in spans)


def test_frame_scores() -> None:
    gold, stated = [(0, 1), (0, 2), (1, 3)], [(0, 1), (1, 3)]
    s = rd.frame_scores([(0, 1), (2, 3)], gold, stated, ambiguous_atoms={3})
    assert s["precision"] == 0.5 and s["stated_recall"] == 0.5 and s["stated_f1"] == pytest.approx(0.5)
    assert s["ambiguous"] == 1 and s["ambiguous_labelled"] == 1 and s["ambiguous_correct"] == 0     # partner 3 found, wrong relation
    assert rd.frame_scores(None, gold, [], set())["stated_f1"] is None


def test_relation_readers_on_the_toy_world(world, tiny_host) -> None:
    ctx = e11.make_context("wordnet", world["ontology"], world["table"], edit.WordNetLexicon())
    readers = rd.RelationReaders(tiny_host, ctx, max_fuzzy=2)
    entry = next(e for e in range(len(world["table"].entry_concepts)) if ctx.frame(e))
    name = next(t for _, a in ctx.frame(entry) if (t := ctx.text(a)))
    text = f"blicket: a kind of {name} that is often seen."
    frame, lm = readers.mc_rel(text, "blicket", ["blicket"])
    assert all(r in ctx.typing.candidates(a) for r, a in frame + lm) and [a for _, a in frame] == [a for _, a in lm]
    if frame:
        prior = readers.log_prior(frame[0][1], ctx.typing.candidates(frame[0][1]))
        assert np.isclose(np.exp(prior).sum(), 1.0)
    react, details = readers.react(text, "blicket", ["blicket"])
    assert all(r in ctx.typing.candidates(a) for r, a in react) and isinstance(details["steps"], list)
    assert len({r for r, _ in react if r in ctx.typing.functional}) == sum(1 for r, _ in react if r in ctx.typing.functional)
