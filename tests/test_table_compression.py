"""Group-wise quantization and compressed embedding tables (WP-Q: D4.3, E4.5)."""

import math

import pytest
import torch
from torch import nn

from vsa_embed.compose import FrameComposer, FrameSchedule
from vsa_embed.evaluation.quantization import group_fake_quantize, group_quantized_bytes, index_bytes
from vsa_embed.evaluation.table_compression import (METHODS, ComposedTable, FactorizedTable, HashEmbeddingTable,
                                                     HashedTable, QRTable, QuantizedTable, ReplacedRowsEmbedding, TTTable,
                                                     balanced_factors, entry_subschedule, row_factors)

N, D = 150, 32


@pytest.fixture(scope="module")
def target() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(N, D, generator=generator) @ torch.randn(D, D, generator=generator) / D**0.5


@pytest.fixture(scope="module")
def keys() -> torch.Tensor:
    return torch.randperm(50_000, generator=torch.Generator().manual_seed(1))[:N]


def _composed(target_rows: int = N, width: int = D, *, delta_bits: int = 0, dimension: int = 8, **extra) -> ComposedTable:
    offsets, relations = torch.tensor([0, 2, 5, 6, 8, 10]), torch.tensor([0, 1, 0, 1, 2, 0, 1, 2, 0, 2])
    fillers = torch.tensor([3, 4, 3, 5, 6, 7, 3, 9, 4, 8])
    schedule, atomics = entry_subschedule(offsets, relations, fillers, torch.tensor([0, 1, 3, 4]))
    entries = torch.arange(target_rows) % 4
    return ComposedTable(target_rows, width, entries=entries, schedule=schedule, atomic_count=atomics.numel(), relation_count=3,
                         dimension=dimension, delta_bits=delta_bits, **extra)


# -- group-wise quantization ---------------------------------------------------------------------------


def test_group_quantized_bytes_are_exact() -> None:
    # 10 rows × 8 columns, groups of 4: packed codes + (scale, zero) in FP16 per group.
    assert group_quantized_bytes(10, 8, 4, 4) == 10 * 8 * 4 // 8 + 10 * 2 * 4 == 120
    assert group_quantized_bytes(10, 8, 2, 4) == 20 + 80
    assert group_quantized_bytes(3, 5, 3, 4) == math.ceil(3 * 5 * 3 / 8) + 3 * 2 * 4      # ragged last group
    assert group_quantized_bytes(10, 8, 8, None, symmetric=True) == 80 + 10 * 2           # per-row symmetric
    assert group_quantized_bytes(10, 8, 16) == 160 and group_quantized_bytes(10, 8, 32) == 320
    assert index_bytes(16, 10) == 5 and index_bytes(8192, 3) == math.ceil(39 / 8) and index_bytes(1, 9) == 2


@pytest.mark.parametrize("bits,group,symmetric", [(2, 4, False), (4, 8, False), (3, 5, False), (8, None, True), (4, 8, True)])
def test_quantized_values_fit_their_code_budget(bits: int, group: int | None, symmetric: bool) -> None:
    x = torch.randn(6, 16)
    q = group_fake_quantize(x, bits, group, symmetric=symmetric, clip_search=True)
    size = 16 if group is None else group
    for row in q:
        for start in range(0, 16, size):
            assert row[start:start + size].unique().numel() <= 2 ** bits
    assert q.shape == x.shape


def test_int8_and_int4_roundtrip_error_is_small_on_random_weights() -> None:
    w = torch.randn(256, 512, generator=torch.Generator().manual_seed(3))
    relative = lambda q: float((q - w).square().sum() / w.square().sum())
    assert relative(group_fake_quantize(w, 8, None, symmetric=True)) < 1e-4
    assert relative(group_fake_quantize(w, 4, 128)) < 2e-2
    assert relative(group_fake_quantize(w, 4, 64, clip_search=True)) <= relative(group_fake_quantize(w, 4, 64)) + 1e-9
    assert torch.equal(group_fake_quantize(w, 32), w)
    assert torch.equal(group_fake_quantize(w, 16), w.half().float())


def test_straight_through_gradient_is_the_identity() -> None:
    x = torch.randn(4, 8, requires_grad=True)
    group_fake_quantize(x, 2, 4, straight_through=True).sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_torchao_int8_matches_the_simulated_scheme() -> None:
    from vsa_embed.evaluation.quantization import quantize_weight_only, tensor_storage_bytes
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 32))
    x = torch.randn(5, 64)
    reference = model(x)
    simulated = group_fake_quantize(model[0].weight.detach(), 8, None, symmetric=True)
    quantize_weight_only(model, 8)
    assert torch.allclose(model(x), reference, atol=0.02)
    weight = model[0].weight
    dequantized, step = weight.qdata.float() * weight.scale.float(), weight.scale.float()
    # Same codes up to rounding-boundary flips by one step (the simulated scale is stored in FP16,
    # torchao's in the weight dtype).
    assert bool(((simulated - dequantized).abs() <= 1.01 * step).all())
    assert float(((simulated - dequantized).abs() > step / 2).float().mean()) < 0.05
    # Measured layout: int8 codes plus per-row scale and zero point.
    assert tensor_storage_bytes(weight) >= 32 * 64 + 32 * 2


# -- compressed tables --------------------------------------------------------------------------------


def test_balanced_and_row_factors() -> None:
    assert balanced_factors(512) == (8, 8, 8) and math.prod(balanced_factors(576)) == 576
    assert balanced_factors(7) == (1, 1, 7)
    n1, n2, n3 = row_factors(25_047)
    assert n1 * n2 * n3 >= 25_047 and n1 * n2 * n3 < 1.1 * 25_047


def test_equal_bytes_accounting_is_exact(keys: torch.Tensor) -> None:
    fp16 = lambda *shape: 2 * math.prod(shape)
    assert QuantizedTable(N, D, bits=4, group_size=8).nbytes == N * D // 2 + N * 4 * 4
    mixed = QuantizedTable(N, D, bits=2, group_size=8, high_bits=3, high_rows=10)
    assert mixed.storage() == {"rows": (N - 10) * D * 2 // 8 + (N - 10) * 4 * 4, "high_rows": math.ceil(10 * D * 3 / 8) + 10 * 4 * 4,
                               "precision_map": math.ceil(N / 8)}
    assert FactorizedTable(N, D, rank=5).nbytes == fp16(N, 5) + fp16(5, D)
    assert QRTable(N, D, collisions=7).nbytes == fp16(math.ceil(N / 7), D) + fp16(7, D)
    assert HashedTable(N, D, buckets=40, keys=keys).nbytes == fp16(40, D) + 16
    assert HashEmbeddingTable(N, D, buckets=40, keys=keys, hashes=2).nbytes == fp16(40, D) + fp16(N, 2) + 16
    tt = TTTable(N, D, rank=3)
    (n1, n2, n3), (d1, d2, d3) = tt.row_shape, tt.column_shape
    assert tt.nbytes == fp16(n1, d1, 3) + fp16(3, n2, d2, 3) + fp16(3, n3, d3)
    composed = _composed(delta_bits=2)
    schedule = composed.composer.schedule
    expected = {"dictionary.atomics": fp16(schedule.fillers.max().item() + 1, 8), "dictionary.transform.roles": fp16(3, 8),
                "schedule.degrees": index_bytes(4, 4), "schedule.relations": index_bytes(3, 9),
                "schedule.fillers": index_bytes(composed.composer.atomic_count, 9), "projector.weight": fp16(D, 8),
                "projector.bias": fp16(D), "row_entries": index_bytes(4, N), "delta": group_quantized_bytes(N, D, 2, 64)}
    assert composed.storage() == expected and composed.nbytes == sum(expected.values())
    assert composed.marginal_bytes() == fp16(D, 8) + fp16(D) + index_bytes(4, N) + group_quantized_bytes(N, D, 2, 64)


@pytest.mark.parametrize("method", ["int", "albert", "qr", "hashing", "hash_embedding", "tt"])
def test_budgets_are_respected_and_tight(method: str, keys: torch.Tensor) -> None:
    cls = METHODS[method]
    extra = {"keys": keys} if method in {"hashing", "hash_embedding"} else {}
    for budget in (1_500, 3_000, 6_000):
        table = cls.for_budget(budget, N, D, group_size=8, **extra)
        if table is None:
            continue
        assert table.nbytes <= budget
        settings = table.settings()
        # One step more capacity would not fit.
        if method == "int":
            bigger = QuantizedTable(N, D, bits=settings["bits"], group_size=8, high_bits=settings["bits"] + 1,
                                    high_rows=settings["high_rows"] + 1) if settings["high_rows"] + 1 <= N else None
        elif method == "albert":
            bigger = FactorizedTable(N, D, rank=settings["rank"] + 1) if settings["rank"] < D else None
        elif method == "qr":
            bigger = QRTable(N, D, collisions=settings["collisions"] - 1) if settings["collisions"] > 1 else None
        elif method == "tt":
            bigger = TTTable(N, D, rank=settings["rank"] + 1)
        else:
            bigger = cls(N, D, buckets=settings["buckets"] + 1, **extra)
        if bigger is not None and bigger.settings() != settings:
            assert bigger.nbytes > budget
    assert METHODS["int"].for_budget(10, N, D) is None


def test_full_capacity_tables_reproduce_the_rows(target: torch.Tensor) -> None:
    for table in (QuantizedTable(N, D, bits=32), FactorizedTable(N, D, rank=D, bits=32), QRTable(N, D, collisions=1, bits=32),
                  TTTable(N, D, rank=10_000, bits=32), _composed(delta_bits=32)):
        stats = table.fit(target)
        assert stats["relative_error"] < 1e-9, table.method
        torch.testing.assert_close(table.dense(), target, rtol=1e-5, atol=1e-5)


def test_more_bits_or_rank_reduce_the_error(target: torch.Tensor, keys: torch.Tensor) -> None:
    errors = [QuantizedTable(N, D, bits=b, group_size=8).fit(target)["relative_error"] for b in (2, 3, 4, 8)]
    assert errors == sorted(errors, reverse=True) and errors[-1] < 1e-3
    ranks = [FactorizedTable(N, D, rank=k).fit(target)["relative_error"] for k in (2, 8, 16)]
    assert ranks == sorted(ranks, reverse=True)
    deltas = [_composed(delta_bits=b).fit(target)["relative_error"] for b in (0, 2, 4, 8)]
    assert deltas == sorted(deltas, reverse=True)
    trained = HashEmbeddingTable(N, D, buckets=60, keys=keys)
    assert trained.fit(target, steps=60, lr=1e-2)["relative_error"] < HashEmbeddingTable(N, D, buckets=60, keys=keys).fit(target)["relative_error"]


def test_composed_table_loads_a_trained_composer() -> None:
    offsets, relations = torch.tensor([0, 2, 5, 6]), torch.tensor([0, 1, 0, 1, 2, 0])
    fillers = torch.tensor([3, 4, 3, 5, 6, 7])
    source = FrameComposer(FrameSchedule(offsets, relations, fillers), 10, 3, 8, mode="attentive", key_dimension=4)
    entries = torch.tensor([0, 2])
    schedule, atomics = entry_subschedule(offsets, relations, fillers, entries)
    assert atomics.tolist() == [3, 4, 7] and schedule.fillers.tolist() == [0, 1, 2]
    table = ComposedTable(4, 6, entries=torch.tensor([0, 1, 1, 0]), schedule=schedule, atomic_count=3, relation_count=3,
                          dimension=8, composition="attentive", key_dimension=4, fit_dictionary=False)
    table.load_dictionary(source, atomics)
    with torch.no_grad():
        torch.testing.assert_close(table._concepts(torch.arange(4)), source.compose(entries)[torch.tensor([0, 1, 1, 0])],
                                   rtol=2e-3, atol=2e-3)      # FP16 storage of the dictionary
    assert not any(p.requires_grad for p in table.composer.parameters())


def test_replaced_rows_embedding_forward_equals_the_dense_forward(target: torch.Tensor) -> None:
    transformers = pytest.importorskip("transformers")
    from vsa_embed.integrations.transformers import ChannelLM
    torch.manual_seed(0)
    host = transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=400, n_positions=32, n_embd=D, n_layer=1, n_head=2))
    model = ChannelLM(host).eval()
    ids = torch.randint(0, 400, (2, 24))
    dense = model(ids, labels=ids, reduction="none")["loss"]
    token_ids = torch.randperm(400)[:N]
    embedding = host.get_input_embeddings()
    rows = embedding.weight.detach()[token_ids]
    for table in (QuantizedTable(N, D, bits=32), FactorizedTable(N, D, rank=D, bits=32), TTTable(N, D, rank=10_000, bits=32),
                  _composed(delta_bits=32)):
        table.fit(rows)
        replaced = ReplacedRowsEmbedding(embedding, token_ids, table)
        host.set_input_embeddings(replaced)
        torch.testing.assert_close(model(ids, labels=ids, reduction="none")["loss"], dense, rtol=1e-4, atol=1e-4)
        replaced.cache()
        torch.testing.assert_close(model(ids, labels=ids, reduction="none")["loss"], dense, rtol=1e-4, atol=1e-4)
        host.set_input_embeddings(embedding)
    # A lossy table changes the loss only through the replaced rows; the tied head is untouched.
    table = QuantizedTable(N, D, bits=1, group_size=None)
    table.fit(rows)
    host.set_input_embeddings(ReplacedRowsEmbedding(embedding, token_ids, table))
    assert host.get_output_embeddings().weight is embedding.weight
    lossy = model(ids, labels=ids, reduction="none")["loss"]
    assert not torch.allclose(lossy, dense)
    host.set_input_embeddings(embedding)
    untouched = torch.tensor([[t for t in range(400) if t not in set(token_ids.tolist())][:24]])
    host.set_input_embeddings(ReplacedRowsEmbedding(embedding, token_ids, table))
    after = model(untouched, labels=untouched)["loss"]
    host.set_input_embeddings(embedding)
    torch.testing.assert_close(after, model(untouched, labels=untouched)["loss"])


def test_finetune_gradients_reach_only_the_table(target: torch.Tensor) -> None:
    embedding = nn.Embedding(400, D)
    embedding.weight.requires_grad_(False)
    token_ids = torch.arange(N)
    table = _composed(delta_bits=4)
    table.fit(target)
    replaced = ReplacedRowsEmbedding(embedding, token_ids, table)
    replaced(torch.tensor([[1, 2, 300]])).sum().backward()
    assert table.delta.grad is not None and table.projector.weight.grad is not None and embedding.weight.grad is None
