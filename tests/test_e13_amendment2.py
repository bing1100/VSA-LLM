"""E13 pre-registration amendment 2 (decision 64): L1's wrong-frame co-primary and the six-test Holm family, the definition
comparators (L1: the definition in context on the noread arm's step-0 model; L5: recall:own − definition with the read
set's definitions on T7-ROOD), L5's token accounting and cost criterion, and the plan / queue script (queued commands
unchanged)."""

import json
import shlex
from pathlib import Path

import numpy as np
import pytest
import torch

from vsa_embed.experiments import e13_cycle as e13

torch.set_num_threads(1)


# -- L1: the co-primary gate and the Holm family --------------------------------------------------------------------------

def test_family_holm_counts_every_registered_test() -> None:
    assert e13.HOLM_FAMILY == ("L1", "L1_random", "L2", "L3", "L4", "L5")
    full = e13.family_holm({"L1": 0.001, "L1_random": 0.004, "L2": 0.5, "L3": 0.02, "L4": 0.9, "L5": 0.01})
    assert full["L1"] == pytest.approx(0.006) and full["L1_random"] == pytest.approx(0.02)       # 6 × p, then 5 × p
    missing = e13.family_holm({"L1": 0.001, "L1_random": None, "L2": None, "L3": None, "L4": None, "L5": None})
    assert missing == {"L1": pytest.approx(0.006)}                                                 # never a family of one
    assert e13.family_holm({k: None for k in e13.HOLM_FAMILY}) == {}


def _effect(mean: float, low: float, high: float, p: float = 0.001) -> dict:
    return {"mean": mean, "ci_low": low, "ci_high": high, "p_value": p}


def test_l1_needs_both_co_primaries_and_is_not_evaluable_without_random() -> None:
    stats = {"holm_alpha": 0.05}
    block = {"L1": _effect(-0.05, -0.07, -0.03), "L1_random": _effect(-0.02, -0.03, -0.01)}
    adjusted = e13.family_holm({"L1": 0.001, "L1_random": 0.001})
    out = e13.verdicts(block, adjusted, stats)
    assert out["L1"] is True and out["L1_vs_noread"] and out["L1_vs_random"]
    any_frame = {**block, "L1_random": _effect(0.001, -0.01, 0.012, p=0.8)}        # a wrong frame does as well (T4's C5sh)
    out = e13.verdicts(any_frame, e13.family_holm({"L1": 0.001, "L1_random": 0.8}), stats)
    assert out["L1_vs_noread"] is True and out["L1_vs_random"] is False and out["L1"] is False
    out = e13.verdicts({"L1": block["L1"]}, e13.family_holm({"L1": 0.001}), stats)
    assert out["L1"] is None and out["L1_vs_noread"] is True                          # no random arm: not evaluable


# -- L1: the definition comparator --------------------------------------------------------------------------------------

def _curves(level: float, counts: np.ndarray) -> tuple:
    return np.array([0, 100]), np.stack([level * counts, level * counts]), np.stack([counts, counts]).astype(float)


def test_l1_definition_differences_out_each_jobs_baseline() -> None:
    rng = np.random.default_rng(0)
    counts = rng.integers(1, 9, 40).astype(float)
    counts[2] = 0                                                                     # a window without round-2 targets
    read, noread = [_curves(2.0, counts)] * 2, [_curves(2.5, counts)] * 2
    defined = (np.arange(40) % 2).astype(np.int64)
    context = {"none": np.stack([2.6 * counts, counts]), "definition": np.stack([2.3 * counts, counts]),
               "added_tokens": 20 * defined, "defined_terms": defined, "starts": np.arange(40)}
    out = e13.l1_definition(read, noread, [context, context], 200,
                            writes=[{"reading_cost": {"forward_tokens": 600}, "definitions_read": 3}])
    # (read − noread) − (definition − none) = −0.5 − (−0.3): the context job's own baseline (2.6) cancels
    assert out["read_minus_definition"]["mean"] == pytest.approx(-0.2) and out["definition_minus_none"]["mean"] == pytest.approx(-0.3)
    assert out["windows"] == 39 and out["windows_with_definitions"] == 20
    assert out["tokens"]["definition_added_per_defined_window"] == 20 and out["tokens"]["read_added_per_window"] == 0
    assert out["tokens"]["read_one_off_forward_tokens_per_term"] == 200
    with pytest.raises(ValueError):
        e13.l1_definition(read, noread, [{**context, "none": context["none"][:, :10]}] * 2, 50)


# -- L5: recall:own − definition, the token accounting and the cost criterion -------------------------------------------

def _reasons(recall: float, definition: float, tokens_recall: int, tokens_definition: int, seeds: int = 2) -> list[dict]:
    """Stage-3 summaries of `seeds` seeds: 30 anchors, a relation item (every condition) and a reverse item (no definition)."""
    out = []
    for s in range(seeds):
        units, added, anchor_of = {"none": {}, "recall:own": {}, "definition": {}}, {"none": {}, "recall:own": {}, "definition": {}}, {}
        for a in range(30):
            for kind in ("relation", "reverse"):
                item = f"a{a}-{kind}"
                anchor_of[item] = f"a{a}"
                jitter = 0.1 * ((a + s) % 3 - 1)
                units["none"][item] = {"accuracy": 0.5}
                added["none"][item] = 0
                units["recall:own"][item] = {"accuracy": recall + jitter}
                added["recall:own"][item] = tokens_recall
                if kind == "relation":
                    units["definition"][item] = {"accuracy": definition + jitter}
                    added["definition"][item] = tokens_definition
        out.append({"units": units, "added_tokens": added, "anchor_of": anchor_of, "conditions": {},
                    "anchors_with_definition": [f"a{a}" for a in range(10)]})
    return out


def test_l5_definition_comparator_token_accounting_and_cost_criterion() -> None:
    reasons = _reasons(0.9, 0.9, 12, 40)
    table = e13.anchor_table(reasons, "recall:own", "definition")
    assert table.shape == (30, 2) and np.allclose(table, 0.0)                         # paired on the relation items only
    assert np.allclose(e13.anchor_table(reasons, "recall:own", "definition", value="tokens"), -28.0)
    l5 = e13._l5(reasons, 1000, margin=0.05)
    assert l5["mean"] == pytest.approx(0.4) and l5["ci_low"] > 0
    tokens = l5["tokens"]
    assert tokens["recall:own"]["added_tokens"] == 12 and tokens["definition"]["added_tokens"] == 40
    assert tokens["recall:own"]["gain_per_100_tokens"] == pytest.approx(100 * 0.4 / 12)
    assert tokens["definition"]["items"] == 30 and tokens["recall:own"]["items"] == 60
    vs = l5["vs_definition"]
    assert vs["noninferior"] and not vs["inferior"] and vs["tokens_recall_minus_definition"]["fewer"]
    assert l5["anchors_with_definition"]["clusters"] == 10
    adjusted = e13.family_holm({"L5": l5["p_value"]})
    assert e13.verdicts({"L5": l5}, adjusted, {"holm_alpha": 0.05})["L5_store_advantage"] is True
    worse = e13._l5(_reasons(0.8, 0.95, 12, 40), 200, margin=0.05)                    # the definition wins by 0.15 > δ
    assert worse["vs_definition"]["inferior"] and not worse["vs_definition"]["noninferior"]
    assert e13.verdicts({"L5": worse}, e13.family_holm({"L5": worse["p_value"]}), {})["L5_store_advantage"] is False
    longer = e13._l5(_reasons(0.9, 0.9, 50, 40), 200, margin=0.05)                    # non-inferior but more tokens
    assert longer["vs_definition"]["noninferior"] and not longer["vs_definition"]["tokens_recall_minus_definition"]["fewer"]
    old = [{k: v for k, v in r.items() if k not in ("added_tokens", "anchors_with_definition")} for r in reasons]
    assert "tokens" not in e13._l5(old, 100) and "tokens_recall_minus_definition" not in e13._l5(old, 100)["vs_definition"]


# -- stage 3: the read set's definitions and the prompt tokens ---------------------------------------------------------------

transformers = pytest.importorskip("transformers")


def _gpt2_ok() -> bool:
    try:
        transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True); return True
    except OSError:
        return False


@pytest.mark.skipif(not _gpt2_ok(), reason="gpt2 tokenizer not cached")
def test_cycle_builder_reads_the_read_sets_definitions_and_counts_tokens(tmp_path: Path) -> None:
    from test_e9_binding import ATOMS, RELATIONS, TERMS, _frame, _lexicon  # noqa: F401
    from vsa_embed.experiments import e9_understanding as und
    from vsa_embed.experiments import e12_self_query as sqx
    ontology = e13.with_frames({"entry_count": len(TERMS), "relation_names": RELATIONS, "atomic_names": ATOMS,
                                "concept_names": TERMS, "entry_concepts": [(i,) for i in range(len(TERMS))], "heldout_entries": []},
                               [_frame(t) for t in TERMS])
    root = tmp_path / "understanding-toy-rel-v1"
    root.mkdir()
    entry = TERMS.index("Brightwater Ledger")
    names = [[RELATIONS[r], ATOMS[f]] for r, f in _frame("Brightwater Ledger")]
    concepts = [{"concept": "u-bl", "subset": "heldout", "surface": "Brightwater Ledger", "entry": entry, "frame": names,
                 "random_frame": None, "source": "Brightwater Ledger", "frequency": 0}]
    base = {"subset": "heldout", "anchor": "u-bl", "slots": {"x": "u-bl"}, "text": {}, "null": {"x": "this"}, "pair": None, "meta": {},
            "family": "negation", "relation": "owned_by", "gold": 0, "chance": 0.5}
    items = [{**base, "id": "u-bl-negation-owned_by-affirm", "test": "affirm", "templates": ["{x} is owned by"],
              "candidates": [" the Zash Team", " nobody"], "pair": "n1"},
             {**base, "id": "u-bl-negation-owned_by-negate", "test": "negate", "templates": ["{x} is not owned by"],
              "candidates": [" nobody", " the Zash Team"], "pair": "n1"},
             {**base, "id": "u-bl-paraphrase-area", "family": "paraphrase", "test": "paraphrase", "relation": "area",
              "templates": ["{x} sits in the"], "candidates": [" money side", " selling side"]}]
    (root / "concepts.jsonl").write_text("".join(json.dumps(c) + "\n" for c in concepts))
    und.write_jsonl_gz(root / "items.jsonl.gz", items)
    (root / "manifest.json").write_text(json.dumps({"schema": und.SCHEMA, "track": "toy", "family": "gpt2"}))
    item_set = sqx.load_item_set(root, families=("reverse", *sqx.RELATION_FAMILIES))
    note = "Brightwater Ledger: the ledger the Zash Team keeps for the money side."
    builder = e13.CycleContextBuilder(item_set, ontology, _lexicon(), {}, store_frames={}, definitions={entry: note})
    texts, records = builder.build(sqx.Condition("definition"))
    assert texts == {"u-bl-negation-owned_by-affirm": note, "u-bl-negation-owned_by-negate": note}     # relation items only
    assert {r["text"] for r in records} == {note}
    empty = e13.CycleContextBuilder(item_set, ontology, _lexicon(), {}, store_frames={}, definitions={})
    assert empty.build(sqx.Condition("definition"))[0] == {}             # no definition: not scored (never without context)
    writerless = e13.CycleContextBuilder(item_set, ontology, _lexicon(), {}, store_frames={})   # E12 has no `toy` writer
    assert writerless.build(sqx.Condition("definition"))[0] == {}
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    counts = e13.context_tokens(tokenizer, {**texts, "slot": "a {x} slot"})
    assert counts["u-bl-negation-owned_by-affirm"] == len(tokenizer(note + "\n", add_special_tokens=False)["input_ids"])
    assert counts["slot"] == len(tokenizer("a { x} slot\n", add_special_tokens=False)["input_ids"])     # E12's guarded text
    assert e13.context_tokens(tokenizer, {}) == {}


def test_round2_definitions_follow_the_write_style() -> None:
    t7 = e13.load_config(e13.ROOT / "t7-rood.yaml")
    concepts = [json.loads(line) for line in (e13.item_path(t7, "read_set", "smollm2") / "concepts.jsonl").read_text().splitlines()]
    entries = {int(c["entry"]) for c in concepts[:20]}
    notes = e13.round2_definitions(t7, "smollm2", entries | {10**9})
    assert set(notes) == entries and all(": " in text for text in notes.values())    # `<headword>: <note>` (style scr)
    assert t7["reason"]["definition_source"] == "read_set" and "definition" in t7["reason"]["conditions"].split(",")
    t5 = e13.load_config(e13.ROOT / "t5.yaml")
    assert t5["reason"]["definition_source"] == "writer" and e13.DEFINITION_SOURCES == ("writer", "read_set")


# -- L1: the definition in context on the step-0 model ---------------------------------------------------------------------

class _TargetModel(torch.nn.Module):
    """Per-token "loss" = the target token id: the definition pass scores the same targets iff the alignment is right."""

    channel = True

    def __init__(self, tokenizer, aliases: dict[int, str]) -> None:
        super().__init__()
        self.tokenizer, self.aliases, self.prefixed = tokenizer, aliases, 0

    def forward(self, ids, spans=None, labels=None, reduction="none"):
        for b, s, e, entry in zip(spans["batch"].tolist(), spans["start"].tolist(), spans["end"].tolist(), spans["entry"].tolist()):
            assert self.aliases[entry] in self.tokenizer.decode(ids[b, s:e + 1].tolist()).lower()   # spans still name their alias
        self.prefixed += int(ids.shape[1] > 32)
        return {"loss": labels[:, 1:].float()}


@pytest.mark.skipif(not _gpt2_ok(), reason="gpt2 tokenizer not cached")
def test_definition_context_scores_the_windows_own_targets_with_the_prefix_linked(tmp_path: Path) -> None:
    from vsa_embed.data.corpus import TokenCorpus, build_corpus, eval_windows
    from vsa_embed.span_channel import AliasTable, CausalLinker
    from vsa_embed.training.lm import save_reference_strata
    table = AliasTable.from_pairs([("hydroxychloroquine", 0), ("new york", 1)])
    texts = [f"Doc {i}: hydroxychloroquine in New York and more words after it here." for i in range(30)]
    build_corpus(texts, tmp_path / "c", tokenizer_name="gpt2", table=table, eos_id=50256, max_tokens=50_000, batch_texts=8, workers=1)
    corpus = TokenCorpus.open(tmp_path / "c")
    starts = eval_windows(corpus, count=6, length=32)
    masks = e13.reference_masks(corpus, starts, 32, 1, {"round2": {0}, "round1": {1}})
    save_reference_strata(tmp_path / "ref.npz", starts, masks, 32)
    run = {"data": {"eval": str(tmp_path / "c"), "min_subtokens": 1}, "model": {"seq_len": 32},
           "eval": {"windows": 6, "batch": 4, "reference_strata": str(tmp_path / "ref.npz")}}
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    model = _TargetModel(tokenizer, {0: "hydroxychloroquine", 1: "new york"})
    definition = "hydroxychloroquine: a drug first made in New York."
    out = e13.definition_context_losses(model, run, {0: definition}, table, tokenizer, torch.device("cpu"), log=lambda *_: None)
    scored = out["none"][1] > 0
    assert scored.sum() == (masks["round2"].any(1)).sum() and model.prefixed > 0
    assert np.array_equal(out["none"], out["definition"])                             # the same targets, the same values
    prefix_ids, prefix_spans = e13.link_text(tokenizer, CausalLinker(table, min_subtokens=1), definition + "\n\n", 1)
    assert (out["added_tokens"][scored] == len(prefix_ids)).all() and (out["defined_terms"][scored] == 1).all()
    assert sorted(prefix_spans["entry"].tolist()) == [0, 1]                           # the prefix is linked like any text
    window = corpus.window(int(starts[0]), 32, min_subtokens=1)
    ids, spans = e13.with_prefix((prefix_ids, prefix_spans), window)
    assert len(ids) == len(prefix_ids) + 32 and (spans["inject"][len(prefix_spans["inject"]):] == window[1]["inject"] + len(prefix_ids)).all()
    assert e13.window_definitions(window[1], {1: "x", 0: "y"}) == [0, 1]              # first-mention order


# -- plan and queue script ---------------------------------------------------------------------------------------------------

def _commands(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        if "jobqueue add" in line and not line.lstrip().startswith("#"):
            words = shlex.split(line.split("   #")[0])
            out[words[words.index("--name") + 1]] = line.split("   #")[0]
    return out


NEW_JOBS = {"e13-t5-Qwen3-1.7B-Base-random-s1": 54.49865, "e13-t7-rood-Qwen3-1.7B-Base-random-s1": 54.49967,
            **{f"e13-t5-{h}-s{s}-context": 54.49885 for h, s in (("SmolLM2-360M", 1), ("SmolLM2-360M", 2), ("SmolLM2-360M", 3), ("Qwen3-1.7B-Base", 1))},
            **{f"e13-t7-rood-{h}-s{s}-context": 54.49977 for h, s in (("SmolLM2-360M", 1), ("SmolLM2-360M", 2), ("SmolLM2-360M", 3), ("Qwen3-1.7B-Base", 1))}}


def test_plan_keeps_every_queued_command_and_adds_the_amendment_jobs() -> None:
    jobs = [j for track in ("t5.yaml", "t7-rood.yaml") for j in e13.plan(e13.load_config(e13.ROOT / track), write_configs=False)]
    plan = {line.split("--name ")[1].split()[0]: line.split("   #")[0] for line in e13.queue_lines(jobs)}
    queued = _commands(e13.ROOT / "queue-commands.sh")
    assert len(queued) == 138 and all(plan[name] == line for name, line in queued.items())    # unchanged command lines
    added = {name: job for name, job in ((j["name"], j) for j in jobs) if name not in queued}
    assert {n: j["priority"] for n, j in added.items()} == NEW_JOBS
    context = added["e13-t7-rood-SmolLM2-360M-s2-context"]["command"]
    assert context[3] == "context" and context[context.index("--arm") + 1].endswith("configs/t7-rood/round2/SmolLM2-360M-noread-s2.yaml")
    assert context[-1].endswith("runs/t7-rood/cycle/SmolLM2-360M-s2/context")
    assert all(0 < j["hours"] < 2.5 for j in added.values()) and 3 < sum(j["hours"] for j in added.values()) < 5
    script = _commands(e13.ROOT / "queue-commands-decision64.sh")
    assert script == {name: plan[name] for name in NEW_JOBS}                          # the script adds exactly these lines
    text = (e13.ROOT / "queue-commands-decision64.sh").read_text()
    assert "NOT EXECUTED" in text and "# CANCEL: (none)" in text
    for track, name in (("t5", "Qwen3-1.7B-Base-random-s1"), ("t7-rood", "Qwen3-1.7B-Base-random-s1")):
        config = e13.load_config(e13.ROOT / f"{track}.yaml")
        written = e13.yaml.safe_load((e13.ROOT / "configs" / track / "round2" / f"{name}.yaml").read_text())
        assert written == e13.round2_config(config, "Qwen3-1.7B-Base", "random", 1) and written["e13"]["frames"] == "random"
