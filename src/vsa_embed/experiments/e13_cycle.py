"""E13 — the learning cycle (decision 63; `manuscript/toolkit-methodology-2026-10.md` §4).

Pre-registration: `experiments/e13-learning-cycle/preregistration.md` (committed before any run). One model goes once
around the read–learn–write cycle on one store (`concept_store.ConceptStore`, methodology M2):

| stage | what happens | code |
|---|---|---|
| 0 passive round 1 | the E9 joint fine-tuning recipe (C5; C0′ / C2 references) on the round-1 corpus with a *seed* ontology: round-2 terms have no frame, round-1 frames lose their edges to round-2 terms, two inverse relations are derived (`owns` = owned_by⁻¹, `has_part` = part_of⁻¹, the E10.9b design) and ≈ 20% of the round-1 edges are erased (the stage-1 gold) | `prepare`, `plan` (configs for `training.lm`) |
| 1 learn | `ConceptStore.propose` (default `rule_closure`; TK-L's proposer by `learn.proposer`) → `accept` with the LM held-out utility test on fresh round-1 text, Holm over proposals; M3 null world = filler-swapped proposals tested in the same family | `learn` (L4) |
| 2 write | round-2 terms (absent from every round-1 text) are read from their definitions by E11's `linker` reader into frames over the store's atomics and relations; rows inserted, no gradient; also the arms' frames (gold, random equal-degree) and the subtoken-mean rows | `write` (L1 is read off stage 4's step-0 evaluation) |
| 3 reason | relation (`negation/affirm`), reverse and two-hop items about round-2 terms, recall tool in context vs no tool (E12's harness on the read store) | `reason` (L5) |
| 4 passive round 2 | continue each stage-0 run (`train.init_mode: continue`) on round-2 text (round-2 documents + fresh general text, 50/50) for `round2.tokens`, one run per arm: `read`, `noread` (no row), `fvt` (trainable subtoken-mean rows), `gold`, `defs` (no row; prose definitions prepended to the documents that mention the term: compute-matched tokens), `random`; references `C0p` (E9 C0′ continued) and `C2` (E9 C2 continued, round-2 rows from its fallback row) | `round2` (L2, forgetting, locality) |
| 5 frozen 4-bit host | the stage-0 host quantized (`model.host_quantization`: rtn, nf4) and frozen; `q4-read` / `q4-noread` train the channel only; `qlora` / `qlora-read` add the in-repo LoRA (r 64) | `round2` (L3) |

Round split (T5): round 2 = the 360 real held-out terms of the T5 glossary (never linked and, by the generator, never
named in any training text); round-2 text is new T5 documents focused on them, generated with the T5 generator under a
fresh seed (`round2_documents`), so it is disjoint from the evaluation documents. T7-ROOD: the round split built by
TK-H1 (`experiments/t7-new-vocabulary/ROOD.md`, not in this tree yet; `T7_ROOD_INTERFACE`).

Every stage-4/5 run evaluates the fixed E9 evaluation windows with two reference strata (`eval.reference_strata`):
`ref_round2` (the 8 targets after a round-2 term) and `ref_round1` (after a round-1 term), at `round2.eval_points`;
per-window losses (`eval_windows.npz`) feed the paired cluster bootstrap over windows × seeds (`report`).

    python -m vsa_embed.experiments.e13_cycle prepare --config experiments/e13-learning-cycle/t5.yaml --family smollm2
    python -m vsa_embed.experiments.e13_cycle learn  --config CFG --run STAGE0_RUN --output DIR
    python -m vsa_embed.experiments.e13_cycle write  --config CFG --run STAGE0_RUN --learned DIR --output DIR
    python -m vsa_embed.experiments.e13_cycle reason --config CFG --run STAGE0_RUN --learned DIR --written DIR --output DIR
    python -m vsa_embed.experiments.e13_cycle round2 --config ARM_YAML --output RUN [--resume]
    python -m vsa_embed.experiments.e13_cycle report --config CFG --output DIR
    python -m vsa_embed.experiments.e13_cycle plan   --config CFG [--no-write-configs]   (prints the queue commands; queues nothing)
    python -m vsa_embed.experiments.e13_cycle smoke  --output DIR                          (CPU, SmolLM2-135M, a few steps; SMOKE)
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gzip
import json
import math
import random
import shutil
import sys
import time
from bisect import bisect_right
from collections import Counter, defaultdict
from itertools import accumulate
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np
import torch
import yaml

from .. import concept_store as cs
from ..statistics import holm_adjust, two_way_cluster_bootstrap, wilson_interval

ROOT = Path("experiments/e13-learning-cycle")
AFTER = 8                                           # targets e+1..e+8 after a span ending at e (the trainer's strata)
PRIMARY_ARMS = ("read", "noread", "fvt", "gold", "defs", "random")
REFERENCE_ARMS = ("C0p", "C2")
FROZEN_ARMS = ("q4-read", "q4-noread", "qlora", "qlora-read")
ARM_FRAMES = {"read": "read", "noread": None, "fvt": None, "gold": "gold", "defs": None, "random": "random", "C0p": None,
              "C2": None, "q4-read": "read", "q4-noread": None, "qlora": None, "qlora-read": "read"}
FAMILY_OF = {"SmolLM2-360M": "smollm2", "SmolLM2-135M": "smollm2", "Qwen3-1.7B-Base": "qwen3", "Qwen3-0.6B-Base": "qwen3"}
HOST_MODE_TAG = {"train": "full", "lora": "lora", "frozen": "frozen"}


# ---------------------------------------------------------------- configuration


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text())
    config["_path"] = str(path)
    return config


def data_root(config: dict[str, Any], family: str) -> Path:
    return Path(config["data_root"]).expanduser() / family


def run_root(config: dict[str, Any]) -> Path:
    return Path(config.get("runs_root") or ROOT / "runs" / config["track"])


def host_family(config: dict[str, Any], host: str) -> str:
    return (config.get("families") or {}).get(host) or FAMILY_OF[host]


def e9_track(config: dict[str, Any]) -> str:
    """The E9 track whose lexicon, relation choices and item specs the config uses (`e9_track`; T5: the track itself)."""
    return str(config.get("e9_track") or config["track"])


def alias_table_path(config: dict[str, Any]) -> Path | None:
    """The full evaluation alias table of the config's ontology (`alias_table`, else the E9 track's)."""
    if config.get("alias_table"):
        return Path(config["alias_table"]).expanduser()
    from .e9_tracks import track_spec
    return track_spec(e9_track(config)).alias_table_path


def rounds_root(config: dict[str, Any], family: str) -> Path:
    """A rounds build's data root for a tokenizer family (`rounds_root`: family → path; T7-ROOD, TK-H1)."""
    return Path(config["rounds_root"][family]).expanduser()


def apply_overrides(run: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """`run_overrides` (smoke tests and reduced pilots only): section → keys merged into every trainer config."""
    for section, values in (config.get("run_overrides") or {}).items():
        if isinstance(values, dict):
            run.setdefault(section, {}).update(copy.deepcopy(values))
        else:
            run[section] = values
    return run


def _json(path: Path, value: Any) -> None:
    from .e5_common import write_json
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_json(Path(path), value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_texts(path: Path, texts: Iterable[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for text in texts:
            handle.write(json.dumps({"text": text}) + "\n")
            count += 1
    return count


def _texts(path: Path) -> Iterator[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)["text"]


# ---------------------------------------------------------------- the round split (T5)


def frames_of(ontology: dict[str, Any]) -> list[list[tuple[int, int]]]:
    offsets = np.asarray(ontology["offsets"]); relations = np.asarray(ontology["relations"]); fillers = np.asarray(ontology["fillers"])
    return [list(zip(relations[offsets[e]:offsets[e + 1]].tolist(), fillers[offsets[e]:offsets[e + 1]].tolist()))
            for e in range(offsets.size - 1)]


def with_frames(ontology: dict[str, Any], frames: Sequence[Sequence[tuple[int, int]]], **updates: Any) -> dict[str, Any]:
    """A copy of `ontology` with the given frames (and counts / names from `updates`)."""
    from ..compose import FrameSchedule
    schedule = FrameSchedule.from_frames(frames)
    out = {**ontology, "offsets": schedule.offsets, "relations": schedule.relations, "fillers": schedule.fillers, **updates}
    out["atomic_count"] = len(out["atomic_names"])
    out["relation_count"] = len(out["relation_names"])
    return out


def seed_ontology(ontology: dict[str, Any], round2: Sequence[int], *, erase_fraction: float = 0.2, min_keep: int = 2,
                  derived_inverses: dict[str, str] | None = None, drop_round2_fillers: bool = True, seed: int = 0
                  ) -> tuple[dict[str, Any], list[tuple[int, int, int]]]:
    """The stage-0 ontology and the erased gold edges `(entry, relation, atom)`.

    1. Round-2 entries get no frame (they are new: the ontology does not know them yet).
    2. (`drop_round2_fillers`) round-1 frames lose their edges whose filler names a round-2 term.
    3. (`derived_inverses`, name → base relation) for every round-1 edge `(h, base, a)` whose filler names a round-1
       entry `t`, the edge `(t, name, atom naming h)` (atoms appended when missing): the materialized inverses that give
       rule closure something to close (E10.9b's ontology).
    4. A seeded `erase_fraction` of the round-1 entries' edges is erased, each entry keeping ≥ `min_keep` edges.
    Synthetic (zero-shot) entries keep their frames; the held-out entries stay held out (never linked in round 1)."""
    from ..concept_store import atom_entries
    round2 = {int(e) for e in round2}
    held = {int(e) for e in ontology["heldout_entries"]}
    synthetic = {int(e) for e in ontology.get("synthetic_entries", ())}
    round1 = [e for e in range(int(ontology["entry_count"])) if e not in held]
    atoms = list(ontology["atomic_names"]); relations = list(ontology["relation_names"])
    atom_entry = atom_entries(ontology)
    names = [str(n) for n in ontology["concept_names"]]
    concepts = ontology["entry_concepts"]
    frames = frames_of(ontology)
    record: dict[str, Any] = {"round1_entries": len(round1), "round2_entries": len(round2), "dropped_round2_fillers": 0}
    for e in round2:
        frames[e] = []
    if drop_round2_fillers:
        for e in round1:
            kept = [(r, a) for r, a in frames[e] if int(atom_entry[a]) not in round2]
            record["dropped_round2_fillers"] += len(frames[e]) - len(kept)
            frames[e] = kept
    atom_of: dict[str, int] = {n: i for i, n in enumerate(atoms)}
    derived: dict[str, int] = {}
    round1_set = set(round1)
    for name, base in (derived_inverses or {}).items():
        if base not in relations:
            raise ValueError(f"derived inverse {name!r}: no relation {base!r}")
        relations.append(name)
        r_new, r_base = len(relations) - 1, relations.index(base)
        count = 0
        for h in round1:
            for r, a in list(frames[h]):
                if r != r_base:                            # (derived edges, with atoms appended here, are never the base)
                    continue
                t = int(atom_entry[a])
                if t < 0 or t not in round1_set:
                    continue
                kind = atoms[a].split(":", 1)[0]
                atom_name = f"{kind}:{names[concepts[h][0]]}"
                if atom_name not in atom_of:
                    atom_of[atom_name] = len(atoms)
                    atoms.append(atom_name)
                if (r_new, atom_of[atom_name]) not in frames[t]:
                    frames[t].append((r_new, atom_of[atom_name]))
                    count += 1
        derived[name] = count
    record["derived_inverses"] = derived
    candidates = [(e, r, a) for e in round1 for r, a in frames[e]]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    target = int(round(erase_fraction * len(candidates)))
    remaining = Counter(e for e, _, _ in candidates)
    erased: list[tuple[int, int, int]] = []
    for e, r, a in candidates:
        if len(erased) >= target:
            break
        if remaining[e] > min_keep:
            erased.append((e, r, a))
            remaining[e] -= 1
    gone = set(erased)
    for e in round1:
        frames[e] = [(r, a) for r, a in frames[e] if (e, r, a) not in gone]
    record.update(edges_round1=len(candidates), erased=len(erased), erase_fraction=erase_fraction, min_keep=min_keep, seed=seed,
                  erased_by_relation=dict(Counter(relations[r] for _, r, _ in erased).most_common()),
                  atoms_added=len(atoms) - len(ontology["atomic_names"]), synthetic_entries=len(synthetic))
    out = with_frames(ontology, frames, atomic_names=atoms, relation_names=relations)
    out["e13"] = {**record, "round2_entries": sorted(round2)}
    return out, sorted(erased)


def round2_documents(glossary: dict[str, Any], focus: Iterable[str], *, seed: int, max_chars: int,
                     uniform_focus: float = 0.5) -> Iterator[str]:
    """New T5 documents about the round-2 terms: `benchmarks.glossary.documents`'s generator (same renderer, document
    styles and mixture) with the focus drawn from `focus` (Zipf weight, or uniformly with probability `uniform_focus`),
    every fact visible (only zero-shot terms stay hidden), under a seed the track build never used."""
    from ..benchmarks import glossary as g
    rng = random.Random(seed * 1_000_003 + 13)
    terms = [t for t in glossary["terms"] if t["split"] != "zeroshot"]
    hidden = frozenset(t["name"] for t in glossary["terms"] if t["split"] == "zeroshot")
    r = g._Renderer(terms, hidden, rng)
    wanted = set(focus)
    pool = [t for t in terms if t["name"] in wanted]
    if not pool:
        raise ValueError("no focus term is in the glossary")
    cumulative = list(accumulate(max(t["weight"], 1e-12) for t in pool))
    lookup = {t["name"]: t for t in terms}

    def sample() -> dict[str, Any]:
        if uniform_focus and rng.random() < uniform_focus:
            return pool[rng.randrange(len(pool))]
        return pool[bisect_right(cumulative, rng.random() * cumulative[-1])]

    simple: list[Callable] = [g._memo, g._ticket, g._incident, g._release, g._chat, g._faq, g._glossary_page]
    produced = 0
    while produced < max_chars:
        term = sample()
        roll = rng.random()
        if roll < 0.2:
            neighbours = [lookup[n] for n in r.related(term) if n in lookup][:2]
            extra = [t for t in neighbours + [sample() for _ in range(rng.randint(0, 2))] if t is not term][:3]
            text = (g._meeting if roll < 0.12 else g._onboarding)(term, r, rng, extra)
        else:
            text = rng.choice(simple)(term, r, rng)
        text += "\n" + " ".join(rng.sample(g.BOILERPLATE, rng.randint(1, 3)))
        produced += len(text)
        yield text


def definitions_prepended(texts: Iterable[str], definitions: dict[str, str]) -> Iterator[str]:
    """`defs` arm: each document preceded by the definitions of the round-2 terms it names (first-mention order,
    case-sensitive name match), so the definitions are read as training text."""
    names = sorted(definitions, key=len, reverse=True)
    for text in texts:
        found = sorted(((text.find(n), n) for n in names if n in text), key=lambda x: x[0])
        seen: list[str] = []
        for _, name in found:
            if not any(name in other for other in seen):
                seen.append(name)
        yield ("\n".join(definitions[n] for n in seen) + "\n\n" + text) if seen else text


def reference_masks(corpus: Any, starts: Sequence[int], length: int, min_subtokens: int, groups: dict[str, set[int]]
                    ) -> dict[str, np.ndarray]:
    """Per group of entries a (windows × (length − 1)) target mask: the `AFTER` targets after each linked span of the
    group (the trainer's after-span rule), for `eval.reference_strata`."""
    masks = {name: np.zeros((len(starts), length - 1), dtype=bool) for name in groups}
    for w, start in enumerate(starts):
        _, spans = corpus.window(int(start), length, min_subtokens=min_subtokens)
        for e, end in zip(spans["entry"].tolist(), spans["end"].tolist()):
            lo, hi = int(end), min(length - 1, int(end) + AFTER)
            if lo >= hi:
                continue
            for name, members in groups.items():
                if int(e) in members:
                    masks[name][w, lo:hi] = True
    return masks


def prose_definitions(read_set_dir: Path, entries: set[int], style: str = "prose") -> dict[int, dict[str, str]]:
    """Per entry its E11 definition of `style` and headword (read set `e11-read/1`)."""
    concepts = {c["concept"]: c for c in _read_jsonl(Path(read_set_dir) / "concepts.jsonl")}
    out = {}
    for d in _read_jsonl(Path(read_set_dir) / "definitions.jsonl"):
        c = concepts[d["concept"]]
        if d["style"] == style and c.get("entry") is not None and int(c["entry"]) in entries:
            out[int(c["entry"])] = {"headword": d["headword"], "text": d["text"], "concept": d["concept"]}
    return out


def prepare(config: dict[str, Any], family: str, *, workers: int = 3, log: Callable[[str], None] = print) -> dict[str, Any]:
    """The round split of a track for one tokenizer family (T5; T7-ROOD reads TK-H1's files): seed ontology and erased
    gold, round-2 corpora (`round2/train`, `round2-defs/train`), the learn-stage validation corpus and the reference
    strata of the evaluation windows. Idempotent (finished parts are kept)."""
    if config.get("rounds_root"):
        return prepare_rounds(config, family, workers=workers, log=log)
    from transformers import AutoTokenizer

    from ..data.concat import concat_corpora
    from ..data.corpus import TokenCorpus, build_corpus, eval_windows, tokenizer_fingerprint
    from ..evaluation import channel_probes as cp
    from ..training.lm import save_reference_strata
    from . import e9_tracks as tracks
    from .c3_corpus import iter_texts
    from .host_corpus import tokenizer_normalization
    spec = tracks.track_spec("t5", family)
    root = data_root(config, family)
    root.mkdir(parents=True, exist_ok=True)
    text_root = Path(config["data_root"]).expanduser() / "text"
    ontology = torch.load(spec.ontology, weights_only=False)
    round2 = sorted(int(e) for e in ontology["heldout_real_entries"])
    settings = config.get("seed_ontology") or {}
    seed = int(config.get("seed", 0))
    seeded, erased = seed_ontology(ontology, round2, erase_fraction=float(settings.get("erase_fraction", 0.2)),
                                   min_keep=int(settings.get("min_keep", 2)), derived_inverses=settings.get("derived_inverses"),
                                   drop_round2_fillers=bool(settings.get("drop_round2_fillers", True)), seed=seed)
    (root / "seed").mkdir(parents=True, exist_ok=True)
    torch.save(seeded, root / "seed" / "ontology.pt")
    _json(root / "seed" / "erased.json", {"edges": [[e, seeded["relation_names"][r], seeded["atomic_names"][a]] for e, r, a in erased],
                                          "record": seeded["e13"]})
    log(f"seed ontology: {seeded['e13']['erased']} of {seeded['e13']['edges_round1']} round-1 edges erased; "
        f"derived {seeded['e13']['derived_inverses']}; {seeded['e13']['dropped_round2_fillers']} edges to round-2 terms dropped")
    # texts (tokenizer-independent; written once for every family)
    text = config.get("text") or {}
    glossary = json.loads((tracks.track_spec("t5").data_root / "docs" / "glossary.json").read_text())
    names = [str(n) for n in ontology["concept_names"]]
    round2_names = [names[ontology["entry_concepts"][e][0]] for e in round2]
    definitions = prose_definitions(item_path(config, "read_set", family), set(round2), str(text.get("definition_style", "prose")))
    files = {"round2": text_root / "round2.jsonl.gz", "defs": text_root / "round2-defs.jsonl.gz",
             "validation": text_root / "learn-validation.jsonl.gz"}
    if not files["round2"].exists():
        n = _write_texts(files["round2"], round2_documents(glossary, round2_names, seed=seed + 1, max_chars=int(text["round2_domain_chars"]),
                                                            uniform_focus=float(text.get("round2_uniform_focus", 0.5))))
        log(f"round-2 documents: {n}")
    if not files["defs"].exists():
        by_name = {names[ontology["entry_concepts"][e][0]]: d["text"] for e, d in definitions.items()}
        _write_texts(files["defs"], definitions_prepended(_texts(files["round2"]), by_name))
    if not files["validation"].exists():
        from ..benchmarks.glossary import documents
        _write_texts(files["validation"], documents(glossary, split="train", seed=seed + 2, max_chars=int(text["validation_chars"]),
                                                    uniform_focus=0.5))
    # corpora with the family's tokenizer and the track's full alias table (round-2 terms linked)
    table = cp.load_alias_table(tracks.ensure_alias_table(spec))
    if table.digest() != ontology["alias_table_sha256"]:
        raise ValueError("the T5 alias table does not reproduce the ontology's digest")
    tokenizer_name = tracks.FAMILY_TOKENIZERS[family]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    normalization = tokenizer_normalization(tokenizer)
    build = dict(tokenizer_name=tokenizer_name, table=table, eos_id=tokenizer.eos_token_id, vocab_size=len(tokenizer),
                 workers=workers, reuse=True, extra_manifest={"tokenizer_sha256": tokenizer_fingerprint(tokenizer), "e13": "round2"},
                 **({"normalization": normalization} if normalization else {}))
    share = float(text.get("general_share", 0.5))
    general_shards = [str(Path(p).expanduser()) for p in text["general_shards"]]
    for name, source in (("round2", files["round2"]), ("round2-defs", files["defs"])):
        # each stream is its domain part plus general text from the same first unused document, so the mix is
        # `general_share` in both (the `defs` documents are longer: their general part is a longer prefix)
        domain = build_corpus(_texts(source), root / f"{name}-domain", max_tokens=10**12, **build)
        build_corpus(iter_texts(general_shards, skip=int(text["general_skip_docs"])), root / f"{name}-general",
                     max_tokens=int(round(domain["tokens"] * share / (1 - share))), **build)
        concat_corpora([root / f"{name}-domain", root / f"{name}-general"], root / name / "train", reuse=True,
                       extra_manifest={"tokenizer_sha256": build["extra_manifest"]["tokenizer_sha256"]})
    build_corpus(_texts(files["validation"]), root / "learn-validation", max_tokens=10**12, **build)
    # reference strata on the E9 evaluation windows
    eval_corpus = TokenCorpus.open(spec.eval_corpus)
    length, windows = int(config["round2"].get("seq_len", 1024)), int(config["round2"].get("eval_windows", spec.windows))
    starts = eval_windows(eval_corpus, count=windows, length=length)
    round1 = {e for e in range(int(ontology["entry_count"])) if e not in set(int(x) for x in ontology["heldout_entries"])}
    masks = reference_masks(eval_corpus, starts, length, int(ontology.get("min_subtokens", 2)), {"round2": set(round2), "round1": round1})
    save_reference_strata(root / f"reference-{windows}x{length}.npz", starts, masks, length)
    manifest = {"track": "t5", "family": family, "tokenizer": tokenizer_name, "round2_entries": len(round2),
                "definitions": len(definitions), "seed_ontology": seeded["e13"] | {"round2_entries": len(round2)},
                "corpora": {name: json.loads((root / name / "manifest.json").read_text())["tokens"]
                            for name in ("round2/train", "round2-defs/train", "learn-validation", "round2-domain", "round2-general",
                                         "round2-defs-domain", "round2-defs-general")},
                "reference": {"windows": windows, "length": length, **{k: int(v.sum()) for k, v in masks.items()}},
                "texts": {k: str(v) for k, v in files.items()}}
    _json(root / "manifest.json", manifest)
    return manifest


def tkl_settings(config: dict[str, Any]) -> dict[str, Any]:
    """TK-L's E10.L settings for a rounds track's stage 1 (`learn.tkl`, merged over `e10_learn.DEFAULTS`)."""
    from . import e10_learn as E
    return E.merge(copy.deepcopy(E.DEFAULTS), (config.get("learn") or {}).get("tkl") or {})


def erase_like_tkl(ontology: dict[str, Any], frames: dict[int, list[tuple[int, int]]], settings: dict[str, Any]
                   ) -> tuple[list[int], list[int], dict[int, list[tuple[int, int]]], set[tuple[int, int, int]]]:
    """E10.L's erasure (`e10_learn.run_erasure`'s first step, with the same arguments): its seen entries, content relations
    (≥ `min_distinct_fillers` fillers) and `learn.erase_edges`. Done once at preparation, so the stage-0 store never sees
    the erased edges, and TK-L's pipeline, rerun on the stage-0 run with the curated frames, erases exactly the same ones."""
    from .. import learn as L
    from . import e10_learn as E
    erase = settings["erase"]
    seen = E.seen_entries(ontology, frames, min_frequency=int(erase["min_frequency"]), min_degree=int(erase["min_degree"]))
    content = E.content_relations(frames, seen, len(ontology["relation_names"]), int(erase["min_distinct_fillers"]))
    erased_frames, erased = L.erase_edges(frames, seen, fraction=float(erase["fraction"]), relations=set(content),
                                          seed=int(settings["seed"]), keep=int(erase["keep"]))
    return seen, content, erased_frames, erased


def rounds_alias_table(config: dict[str, Any], ontology: dict[str, Any]) -> Path:
    """The rounds build's evaluation alias table: the E9 track's aliases with round 2 as the holdout (checked against the
    rounds ontology's `alias_table_sha256`), written once to `alias_table`."""
    from ..evaluation import channel_probes as cp
    from ..span_channel import AliasTable
    from . import e9_tracks as tracks
    path = alias_table_path(config)
    if path.exists():
        return path
    base = cp.load_alias_table(tracks.ensure_alias_table(tracks.track_spec(e9_track(config))))
    holdout = frozenset(int(c) for e in ontology["heldout_entries"] for c in base.entry_concepts[int(e)])
    table = AliasTable(base.alias_to_entry, base.entry_concepts, holdout, base.normalization)
    if table.digest() != ontology["alias_table_sha256"]:
        raise ValueError("the E9 alias table with round 2 held out does not reproduce the rounds ontology's digest")
    path.parent.mkdir(parents=True, exist_ok=True)
    cp.save_alias_table(table, path)
    return path


def definitions_by_links(corpus: Any, tokenizer: Any, definitions: dict[int, str], min_subtokens: int) -> Iterator[str]:
    """`defs` arm on a built stream: every document decoded (byte-level BPE is lossless) and preceded by the definitions
    of the defined terms it links (≥ ℓ_min subtokens, in first-mention order)."""
    tokens = np.asarray(corpus.tokens)
    eos = int(corpus.manifest.get("eos_id", 0))
    inject, entries, lengths = corpus.spans["inject"], corpus.spans["entry"], corpus.spans["length"]
    ends = np.flatnonzero(tokens == eos).tolist()
    if len(tokens) and int(tokens[-1]) != eos:
        ends.append(len(tokens))
    start = 0
    for end in ends:
        text = tokenizer.decode(tokens[start:end].tolist())
        lo, hi = np.searchsorted(inject, np.asarray([start, end], dtype=inject.dtype), side="left")
        linked = [int(e) for e, n in zip(entries[lo:hi].tolist(), lengths[lo:hi].tolist()) if n >= min_subtokens and int(e) in definitions]
        shown = list(dict.fromkeys(linked))
        yield ("\n".join(definitions[e] for e in shown) + "\n\n" + text) if shown else text
        start = end + 1


def build_round2_items(config: dict[str, Any], family: str, output: Path | None = None, *, seed: int = 0) -> dict[str, Any]:
    """Understanding items (`e9_understanding`, the E9 track's spec and exclusion table) of a rounds track's round-2 anchors:
    built on the rounds ontology and alias table, where round 2 is the held-out set, so the `heldout` subset is round 2;
    families relation (`negation/affirm`), reverse, paraphrase and two-hop (T7's spec has no path); no seen, rare or new
    subsets (CPU; written to `items.understanding[family]`)."""
    from transformers import AutoTokenizer

    from ..evaluation import channel_probes as cp
    from . import e9_understanding as und
    from .e9_freqbias import ensure_exclusion_table
    from .e9_tracks import FAMILY_TOKENIZERS, lexicon_for, track_spec
    track = e9_track(config)
    path = rounds_root(config, family) / "ontology.pt"
    ontology = torch.load(path, weights_only=False)
    table_path = rounds_alias_table(config, ontology)
    ctx = und.BuildContext(track=track, family=family, ontology=ontology, table=cp.load_alias_table(table_path),
                           lexicon=lexicon_for(track_spec(track), ontology),
                           tokenizer=AutoTokenizer.from_pretrained(FAMILY_TOKENIZERS[family], local_files_only=True),
                           tokenizer_name=FAMILY_TOKENIZERS[family], exclusions=torch.load(ensure_exclusion_table(track, family), weights_only=False),
                           families_spec=und.TRACK_SPECS[track], ontology_path=path, alias_table_path=table_path)
    return und.build_items_from_context(ctx, Path(output or item_path(config, "understanding", family)), counts={"seen": 0, "rare": 0},
                                        new_items=None, seed=seed, families=("two_hop", "reverse", "paraphrase", "negation"))


def prepare_rounds(config: dict[str, Any], family: str, *, workers: int = 3, log: Callable[[str], None] = print) -> dict[str, Any]:
    """A rounds build (T7-ROOD: TK-H1's `rounds-v1`, `experiments/t7-new-vocabulary/ROOD.md` §5) in the E13 layout for one
    tokenizer family: the seed ontology (round-2 frames empty, round-1 edges to round-2 fillers dropped, E10.L's erasure:
    `erase_like_tkl`), the curated frames and erased gold, the rounds alias table, `round2/train` (a link to the build's
    `train-round2`), `round2-defs/train` (the same documents with the definitions prepended, topped up with general text
    no T7 build read so the stream stays 50/50) and the reference strata of the `eval-round2` windows. Idempotent."""
    from transformers import AutoTokenizer

    from ..data.concat import concat_corpora
    from ..data.corpus import TokenCorpus, build_corpus, eval_windows, tokenizer_fingerprint
    from ..evaluation import channel_probes as cp
    from ..training.lm import save_reference_strata
    from .c3_corpus import iter_texts
    from .host_corpus import tokenizer_normalization
    rounds, root = rounds_root(config, family), data_root(config, family)
    (root / "seed").mkdir(parents=True, exist_ok=True)
    ontology = torch.load(rounds / "ontology.pt", weights_only=False)
    round2 = sorted(int(e) for e in ontology["heldout_entries"])
    settings = config.get("seed_ontology") or {}
    curated, _ = seed_ontology(ontology, round2, erase_fraction=0.0, derived_inverses=settings.get("derived_inverses"),
                               drop_round2_fillers=bool(settings.get("drop_round2_fillers", True)))
    frames = dict(enumerate(frames_of(curated)))
    tkl = tkl_settings(config)
    # Every family erases the same edges: the seen set comes from the reference family's training frequencies (the frames
    # are the same in every relink; a tokenizer's own frequencies would shift the seeded draws).
    reference = settings.get("reference_family")
    basis = curated
    if reference and reference != family:
        basis = {**curated, "train_frequency": torch.load(rounds_root(config, reference) / "ontology.pt", weights_only=False)["train_frequency"]}
    seen, content, erased_frames, erased = erase_like_tkl(basis, frames, tkl)
    seeded = with_frames(curated, [erased_frames[e] for e in range(len(frames))])
    relation_names, atomic_names = seeded["relation_names"], seeded["atomic_names"]
    seeded["e13"] = {**curated["e13"], "erased": len(erased), "erase_rule": "e10_learn: learn.erase_edges over content relations",
                     "erase": tkl["erase"], "erase_seed": int(tkl["seed"]), "seen": len(seen), "seen_from": reference or family,
                     "content_relations": [relation_names[r] for r in content],
                     "erased_by_relation": dict(Counter(relation_names[r] for _, r, _ in erased).most_common())}
    torch.save(seeded, root / "seed" / "ontology.pt")
    torch.save({"frames": frames, "seen": seen, "erased": sorted(erased)}, root / "seed" / "curated.pt")
    _json(root / "seed" / "erased.json", {"edges": [[e, relation_names[r], atomic_names[a]] for e, r, a in sorted(erased)],
                                          "record": seeded["e13"]})
    log(f"seed ontology ({family}): {len(erased)} of {sum(len(frames[e]) for e in seen)} edges of {len(seen)} seen round-1 entries "
        f"erased (content relations {seeded['e13']['content_relations']}); {seeded['e13']['dropped_round2_fillers']} edges to "
        "round-2 terms dropped")
    table = cp.load_alias_table(rounds_alias_table(config, ontology))
    (root / "round2").mkdir(parents=True, exist_ok=True)
    if not (root / "round2" / "train").exists():
        (root / "round2" / "train").symlink_to(rounds / "train-round2", target_is_directory=True)
    stream = TokenCorpus.open(rounds / "train-round2")
    tokenizer_name = stream.manifest["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    normalization = tokenizer_normalization(tokenizer)
    build = dict(tokenizer_name=tokenizer_name, table=table, eos_id=int(stream.manifest["eos_id"]), vocab_size=len(tokenizer),
                 workers=workers, reuse=True, extra_manifest={"tokenizer_sha256": tokenizer_fingerprint(tokenizer), "e13": "round2-defs"},
                 **({"normalization": normalization} if normalization else {}))
    style = str((config.get("write") or {}).get("style", "prose"))
    definitions = {e: d["text"] for e, d in prose_definitions(item_path(config, "read_set", family), set(round2), style).items()}
    min_subtokens = int(ontology.get("min_subtokens", 2))
    defs: dict[str, Any] = {"built": False}
    if any("defs" in spec.get("round2_arms", ()) for host, spec in config["hosts"].items() if host_family(config, host) == family):
        documents = build_corpus(definitions_by_links(stream, tokenizer, definitions, min_subtokens), root / "round2-defs-documents",
                                 max_tokens=10**12, **build)
        sources = json.loads((rounds / "train-round2" / "sources.json").read_text())["tokens"]
        added = int(documents["tokens"]) - int(stream.manifest["tokens"])
        text = config.get("text") or {}
        top_up = max(0, int(sources["pubmed"]) + added - int(sources["general"]))
        build_corpus(iter_texts([str(Path(p).expanduser()) for p in text["general_shards"]], skip=int(text["general_skip_docs"])),
                     root / "round2-defs-general", max_tokens=top_up, **build)
        concat_corpora([root / "round2-defs-documents", root / "round2-defs-general"], root / "round2-defs" / "train", reuse=True,
                       extra_manifest={"tokenizer_sha256": build["extra_manifest"]["tokenizer_sha256"]})
        defs = {"built": True, "definition_tokens_added": added, "general_top_up_tokens": top_up,
                "tokens": json.loads((root / "round2-defs" / "train" / "manifest.json").read_text())["tokens"]}
    eval_corpus = TokenCorpus.open(rounds / "eval-round2")
    length = int(config["round2"].get("seq_len", 1024))
    windows = int(config["stage0"]["eval_windows"][family])
    starts = eval_windows(eval_corpus, count=windows, length=length)
    held = set(round2)
    masks = reference_masks(eval_corpus, starts, length, min_subtokens,
                            {"round2": held, "round1": {e for e in range(int(ontology["entry_count"])) if e not in held}})
    save_reference_strata(root / f"reference-{windows}x{length}.npz", starts, masks, length)
    manifest = {"track": config["track"], "family": family, "rounds_root": str(rounds), "tokenizer": tokenizer_name,
                "round2_entries": len(round2), "definitions": len(definitions), "seed_ontology": seeded["e13"] | {"round2_entries": len(round2)},
                "defs": defs, "corpora": {"round2/train": int(stream.manifest["tokens"])},
                "reference": {"windows": len(starts), "length": length, **{k: int(v.sum()) for k, v in masks.items()}}}
    _json(root / "manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------- stage 0


def stage0_name(host: str, mode: str, seed: int) -> str:
    return f"{host}-{HOST_MODE_TAG[mode]}-C5-s{seed}"


def stage0_config(config: dict[str, Any], host: str, seed: int) -> dict[str, Any]:
    """The E9 C5 config of this host and seed (its recipe unchanged) on the seed ontology, with the round-1/round-2
    reference strata and `channel.skip_empty_frames` (round-2 entries have no frame)."""
    family = host_family(config, host)
    source = Path(config["stage0"]["configs"][host].format(seed=seed))
    run = apply_overrides(yaml.safe_load(source.read_text()), config)
    root = data_root(config, family)
    # A rounds track (T7-ROOD): the E9 recipe on the build's round-1 corpus, evaluated on every `eval-round2` window
    # (`stage0.data` paths relative to `rounds_root`; `stage0.eval_windows` per family). T5 sets neither.
    for key, value in ((config["stage0"].get("data") or {}).items()):
        run["data"][key] = str(rounds_root(config, family) / value)
    if (config["stage0"].get("eval_windows") or {}).get(family):
        run["eval"]["windows"] = int(config["stage0"]["eval_windows"][family])
    run["data"]["ontology"] = str(root / "seed" / "ontology.pt")
    run["channel"]["skip_empty_frames"] = True
    run["eval"]["reference_strata"] = str(reference_path(config, family, run))
    run["experiment"] = f"e13-{config['track']}-stage0-{host}-C5-s{seed}"
    run["e13"] = {"stage": 0, "source_config": str(source)}
    return run


def reference_path(config: dict[str, Any], family: str, run: dict[str, Any]) -> Path:
    return data_root(config, family) / f"reference-{int(run['eval']['windows'])}x{int(run['model']['seq_len'])}.npz"


def stage0_dir(config: dict[str, Any], host: str, seed: int) -> Path:
    mode = yaml.safe_load(Path(config["stage0"]["configs"][host].format(seed=seed)).read_text())["model"]["host_mode"]
    return run_root(config) / "stage0" / stage0_name(host, mode, seed)


def cycle_dir(config: dict[str, Any], host: str, seed: int) -> Path:
    return run_root(config) / "cycle" / f"{host}-s{seed}"


# ---------------------------------------------------------------- stage 1: learn


def run_family(config: dict[str, Any], run: Any) -> str:
    """The tokenizer family of an opened run (its host's)."""
    return host_family(config, str(run.config["model"]["pretrained"]).split("/")[-1])


def _open(run_dir: Path, config: dict[str, Any], *, device: str | None = None, batch_size: int = 16) -> Any:
    from .e5_common import open_run
    return open_run(Path(run_dir), device=device, alias_table=alias_table_path(config), batch_size=batch_size,
                    max_length=int(config.get("max_length", 512)))


def accepted_edges(store: cs.ConceptStore, learned: Path | None) -> list[cs.Proposal]:
    """The stage-1 accepted edges of a `learn` folder (none if `learned` is None)."""
    if learned is None:
        return []
    rows = json.loads((Path(learned) / "accepted.json").read_text())["edges"]
    return [cs.Proposal(int(e), store.relation_id[r], store.atomic_id[a], source="learned") for e, r, a in rows
            if r in store.relation_id and a in store.atomic_id]


LEARN_METHODS = ("rule_closure", "tkl")


def learn(config: dict[str, Any], run_dir: Path, output: Path, *, device: str | None = None, limit: int | None = None,
          method: str | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Stage 1 (L4) by `method` (default `learn.primary`; absent: `rule_closure`, T5's registered proposer): `rule_closure`
    (`learn_closure`) or `tkl` (TK-L's learn tool, `learn_tkl`)."""
    method = method or (config.get("learn") or {}).get("primary", "rule_closure")
    if method not in LEARN_METHODS:
        raise ValueError(f"learn method must be one of {LEARN_METHODS}")
    if method == "tkl":
        return learn_tkl(config, run_dir, output, device=device, limit=limit, log=log)
    return learn_closure(config, run_dir, output, device=device, limit=limit, log=log)


def learn_tkl(config: dict[str, Any], run_dir: Path, output: Path, *, device: str | None = None, limit: int | None = None,
              log: Callable[[str], None] = print) -> dict[str, Any]:
    """Stage 1 with TK-L's learn tool (`vsa_embed.learn`: `propose` — the function behind `store_proposer` — with sources
    decompose and closure) under its pre-registered `holm+decoy` rule, run through E10.L's `run_erasure` (the facade alone
    has no null-world decoys and would apply Holm only). Inputs: the stage-0 store (its composer: dictionary and typed
    candidates of the erased frames), the curated frames (`erase_like_tkl` erases the same edges again; checked), and host
    hidden states of the stage-0 run at round-1 terms' occurrences (`e10_learn.extract_features`; `learn.tkl.extract`),
    split by document into the passive half (decomposed) and the evidence half (the acceptance test). Writes
    `occurrences.npz`, `accepted.json` (the edges stages 2 and 4 write), `summary.json` (L4: precision against the erased
    gold of the probes, false acceptance in E10.L's null worlds) and `proposals.jsonl.gz`."""
    from scipy import stats

    from .. import learn as L
    from . import e10_learn as E
    from .e9_binding_chain import load_composer
    started = time.monotonic()
    settings = tkl_settings(config)
    if limit:
        settings["erase"]["probe_cap"] = int(limit)
    run_config = yaml.safe_load((Path(run_dir) / "resolved_config.yaml").read_text())
    root = data_root(config, host_family(config, str(run_config["model"]["pretrained"]).split("/")[-1]))
    curated = torch.load(root / "seed" / "curated.pt", weights_only=False)
    frames = {int(e): [(int(r), int(a)) for r, a in f] for e, f in curated["frames"].items()}
    seen = [int(e) for e in curated["seen"]]
    output.mkdir(parents=True, exist_ok=True)
    features = output / "occurrences.npz"
    extract = settings.get("extract") or {}
    if not features.exists():
        splits = [(p.split(":")[0], int(p.split(":")[1]) if ":" in p else None) for p in str(extract.get("splits", "eval,train:20000")).split(",")]
        data = E.extract_features(Path(run_dir), seen, window=int(extract.get("window", 512)), max_per_entry=int(extract.get("max_per_entry", 32)),
                                  batch=int(extract.get("batch", 8)), splits=splits, device=device, alias_table=alias_table_path(config))
        np.savez(features, **data)
    composer, _, ontology = load_composer(Path(run_dir))
    names = list(ontology["relation_names"])
    passive, observations, meta = E.load_features(features, layer=settings["evidence"]["layer"], split_seed=int(settings["seed"]))
    keep, cap = set(seen), int(settings["evidence"]["max_observations"])
    inputs = E.ErasureInputs(Path(run_dir).name, frames, seen, names,
                             lambda fr: L.Dictionary.from_composer(composer, fr, relation_names=names),
                             {c: v for c, v in passive.items() if c in keep}, {c: v[:cap] for c, v in observations.items() if c in keep},
                             E.atom_concepts(ontology).tolist(), meta={"store": str(run_dir), "features": str(features), **meta})
    out = E.run_erasure(inputs, settings)
    summary = out["summary"]
    if int(summary["erased"]) != len(curated["erased"]):
        raise RuntimeError(f"TK-L's erasure ({summary['erased']} edges) differs from the seed ontology's ({len(curated['erased'])})")
    primary = settings["test"]["correction"]
    accepted = [r for r in out["records"] if r.get("world") == "real" and r["accepted"]]
    decision = summary["primary"]["all"]
    worlds = {kind: rules[primary] for kind, rules in summary["nulls"].items() if primary in rules}
    null_tested, null_accepted = sum(w["tested"] for w in worlds.values()), sum(w["accepted"] for w in worlds.values())
    threshold = float((config.get("learn") or {}).get("precision_threshold", 0.8))
    metrics = {"proposer": "vsa_embed.learn (decompose + closure; store_proposer's propose)", "rule": primary,
               "proposals": decision.get("proposed"), "accepted": decision["accepted"], "accepted_gold": decision["tp"],
               "erased_gold": decision["gold"], "precision": decision["precision"], "recall": decision["recall"],
               "null": {"null_proposals": null_tested, "accepted": null_accepted, "rate": null_accepted / null_tested if null_tested else float("nan"),
                        "rate_max": summary["null_far_max"].get(primary), "by_world": worlds},
               "p_precision_above_threshold": float(stats.binom.sf(decision["tp"] - 1, decision["accepted"], threshold)) if decision["accepted"] else 1.0,
               "probes": summary["probes"], "erased": summary["erased"], "decoders": summary["decoders"], "by_source": summary["primary"],
               "seconds": time.monotonic() - started}
    atoms = list(ontology["atomic_names"])
    _json(output / "accepted.json", {"edges": [[int(r["concept"]), names[int(r["relation"])], atoms[int(r["filler"])]] for r in accepted],
                                     "metrics": metrics})
    _json(output / "summary.json", metrics)
    with gzip.open(output / "proposals.jsonl.gz", "wt") as handle:
        for record in out["records"]:
            handle.write(json.dumps(record, default=str) + "\n")
    log(f"learn (TK-L, {primary}): accepted {decision['accepted']} (precision {decision['precision']:.3f} on {summary['probes']} probes), "
        f"null false acceptance ≤ {metrics['null']['rate_max']}")
    return metrics


def learn_closure(config: dict[str, Any], run_dir: Path, output: Path, *, device: str | None = None, limit: int | None = None,
                  log: Callable[[str], None] = print) -> dict[str, Any]:
    """Stage 1 by rule closure (L4 on T5; a secondary on T7-ROOD): propose missing edges of round-1 entries, test them
    with the true and null proposals in one Holm family (`HeldOutUtilityTest` on the learn-validation corpus, or the rounds
    split `learn.validation.corpus`), score the accepted ones against the erased gold."""
    from ..data.corpus import TokenCorpus
    started = time.monotonic()
    settings = config.get("learn") or {}
    run = _open(run_dir, config, device=device)
    root = data_root(config, run_family(config, run))
    store = cs.ConceptStore.from_run(run)
    erased_rows = json.loads((root / "seed" / "erased.json").read_text())["edges"]
    gold = {(int(e), store.relation_id[r], store.atomic_id[a]) for e, r, a in erased_rows}
    held = {int(e) for e in run.ontology["heldout_entries"]}
    round1 = [e for e in range(int(run.ontology["entry_count"])) if e not in held]
    rules = cs.RuleSettings(**(settings.get("rules") or {}))
    proposer = settings.get("proposer", "rule_closure")
    options = {"settings": rules} if proposer == "rule_closure" else dict(settings.get("proposer_options") or {})
    proposals = store.propose(cs.Evidence(entries=round1), proposer, **options)
    if limit:
        proposals = proposals[:limit]
    nulls = cs.null_proposals(store, proposals, seed=int(settings.get("null_seed", 0)), exclude=gold)
    log(f"learn: {len(proposals)} proposals ({sum(p.key in gold for p in proposals)} erased-gold), {len(nulls)} null proposals")
    validation = settings.get("validation") or {}
    corpus = TokenCorpus.open(rounds_root(config, run_family(config, run)) / validation["corpus"] if validation.get("corpus")
                              else root / "learn-validation")
    windows = cs.validation_windows(corpus, sorted({p.entry for p in proposals + nulls}), per_entry=int(validation.get("per_entry", 8)),
                                    length=int(validation.get("length", 128)), min_subtokens=int(run.config["data"]["min_subtokens"]),
                                    seed=int(validation.get("seed", 0)))
    test = cs.HeldOutUtilityTest(run.model, windows, device=run.device, alpha=float(settings.get("alpha", 0.05)),
                                 batch=int(settings.get("batch", 16)), min_windows=int(validation.get("min_windows", 3)))
    decisions = store.accept(proposals + nulls, test)
    real = [d for d in decisions if d.key.source != "null"]
    accepted = [d.key for d in real if d.accept]
    hits = sum(p.key in gold for p in accepted)
    far = cs.false_acceptance_rate(decisions)
    precision = hits / len(accepted) if accepted else float("nan")
    low, high = wilson_interval(hits, len(accepted)) if accepted else (float("nan"), float("nan"))
    from scipy import stats
    p_l4 = float(stats.binom.sf(hits - 1, len(accepted), float(settings.get("precision_threshold", 0.8)))) if accepted else 1.0
    metrics = {"proposals": len(proposals), "proposals_gold": sum(p.key in gold for p in proposals), "erased_gold": len(gold),
               "tested": sum(1 for d in real if d.n), "accepted": len(accepted), "accepted_gold": hits,
               "precision": precision, "precision_ci": [low, high], "recall": hits / len(gold) if gold else float("nan"),
               "proposal_recall": sum(p.key in gold for p in proposals) / len(gold) if gold else float("nan"),
               "null": far, "p_precision_above_threshold": p_l4, "validation_windows": sum(len(v) for v in windows.values()),
               "entries_with_windows": len(windows), "forward_tokens": test.forwarded,
               "rules": [{"head": store.relation_names[r["head"]], "kind": r["kind"], "args": [store.relation_names[a] for a in r["args"]],
                          "support": r["support"], "confidence": r["confidence"], "predicted": len(r["predicted"])}
                         for r in cs.mined_rules(store, rules)] if proposer == "rule_closure" else None,
               "proposer": proposer, "seconds": time.monotonic() - started}
    output.mkdir(parents=True, exist_ok=True)
    name = lambda p: [int(p.entry), store.relation_names[p.relation], store.atomic_names[p.atom]]      # noqa: E731
    _json(output / "accepted.json", {"edges": [name(p) for p in accepted], "metrics": metrics})
    _json(output / "decisions.json", [{"edge": name(d.key), "source": d.key.source, "gold": d.key.key in gold, "score": d.key.score,
                                       "rule": d.key.meta.get("rule"), "mean": d.mean, "n": d.n, "accept": d.accept, **d.extra}
                                      for d in decisions])
    _json(output / "summary.json", metrics)
    log(f"learn: accepted {len(accepted)} (precision {precision:.3f}), null false-acceptance {far['rate']:.3f}")
    return metrics


# ---------------------------------------------------------------- stage 2: write


def item_path(config: dict[str, Any], key: str, family: str) -> Path:
    """`items.<key>`: one path for every host, or a mapping family → path."""
    value = config["items"][key]
    return Path(value[family] if isinstance(value, dict) else value)


def _read_set(config: dict[str, Any], entries: set[int], family: str) -> Any:
    import dataclasses

    from . import e11_read_to_learn as e11
    read_set = e11.load_read_set(item_path(config, "read_set", family))
    keep = {c["concept"] for c in read_set.concepts if c.get("entry") is not None and int(c["entry"]) in entries}
    return dataclasses.replace(read_set, concepts=[c for c in read_set.concepts if c["concept"] in keep],
                               definitions=[d for d in read_set.definitions if d["concept"] in keep],
                               items=[i for i in read_set.items if i["concept"] in keep])


def round2_entries(ontology: dict[str, Any]) -> list[int]:
    return sorted(int(e) for e in (ontology.get("e13") or {}).get("round2_entries", ontology.get("heldout_real_entries", ())))


def committed(run: Any, store: cs.ConceptStore, learned: Path | None) -> int:
    """Write the stage-1 accepted edges into the run's store and its ontology frames (the learned store)."""
    added = store.commit(accepted_edges(store, learned))
    if added:
        s = store.schedule
        run.ontology = {**run.ontology, "offsets": s.offsets.cpu(), "relations": s.relations.cpu(), "fillers": s.fillers.cpu()}
    return added


@torch.no_grad()
def fvt_rows(run: Any, entries: Sequence[int], *, sample: int = 2048, seed: int = 0) -> tuple[torch.Tensor, dict[str, Any]]:
    """`fvt` arm: per entry the subtoken mean of the host's input embeddings over its aliases (`e9_rowsource`), scaled to
    the mean norm of the channel's static rows of a sample of round-1 entries (as `e9_dim3_baselines`' surface_mean)."""
    from .e9_rowsource import subtoken_mean_rows
    aliases: dict[int, list[str]] = defaultdict(list)
    for alias, entry in run.table.alias_to_entry.items():
        aliases[int(entry)].append(alias)
    raw, info = subtoken_mean_rows(run.model.model.get_input_embeddings().weight, run.tokenizer, [aliases[e] for e in entries])
    held = {int(e) for e in run.ontology["heldout_entries"]}
    degrees = np.diff(np.asarray(run.ontology["offsets"]))
    pool = [e for e in range(int(run.ontology["entry_count"])) if e not in held and degrees[e] > 0]
    rng = random.Random(seed)
    reference = torch.tensor(sorted(rng.sample(pool, min(sample, len(pool)))), device=run.device)
    norm = float(run.channel.rows({"entry": reference}).float().norm(dim=-1).mean())
    rows = raw * (norm / raw.norm(dim=-1, keepdim=True).clamp_min(1e-8))
    return rows, {**info, "target_norm": norm, "reference_entries": int(reference.numel())}


@torch.no_grad()
def general_window_losses(model: Any, config: dict[str, Any], corpus_path: Path, *, windows: int = 256,
                          device: torch.device | str = "cpu") -> np.ndarray:
    """Per-window mean loss on the general-text corpus (locality), the trainer's evaluation path (`all` stratum)."""
    from ..data.corpus import TokenCorpus, eval_windows
    from ..training.lm import evaluate
    corpus = TokenCorpus.open(Path(corpus_path))
    starts = eval_windows(corpus, count=windows, length=int(config["model"]["seq_len"]))
    sink: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    evaluate(model, corpus, starts, config, None, set(), torch.device(device), window_sink=sink)
    sums, counts = (np.concatenate(x) for x in sink["all"])
    return np.stack([sums, counts])


def write(config: dict[str, Any], run_dir: Path, learned: Path | None, output: Path, *, device: str | None = None,
          limit: int | None = None, only: set[int] | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Stage 2: read every round-2 definition into a frame on the learned store (E11 readers; the pre-registered
    `linker` is the read arm), write the arms' frames (`frames.json`: read, gold, random; secondary readers' metrics),
    the `fvt` rows and the stage-0 model's general-text losses (locality baseline). No gradient. `only` / `limit`
    (smoke tests only) restrict the terms read."""
    from . import e11_read_to_learn as e11
    started = time.monotonic()
    settings = config.get("write") or {}
    run = _open(run_dir, config, device=device)
    store = cs.ConceptStore.from_run(run)
    added = committed(run, store, learned)
    entries = set(round2_entries(run.ontology))
    read_set = _read_set(config, entries & set(only) if only is not None else entries, run_family(config, run)).limit(limit)
    ctx = e11.make_context(e9_track(config), run.ontology, run.table, store.lexicon)
    style = str(settings.get("style", "prose"))
    readers = ["oracle", "random", settings.get("reader", "linker"), *settings.get("secondary_readers", ["linker-joint", "typeprior"])]
    scorer = e11.DefinitionScorer(run, read_set)
    results, mentions = e11.run_readers(run, read_set, ctx, list(dict.fromkeys(readers)), [style], scorer=scorer, log=log)
    rows = e11.frame_rows(results, read_set, ctx, mentions)
    entry_of = e11.link_entry_of(read_set, run)
    by_reader: dict[str, dict[int, list[list[str]] | None]] = defaultdict(dict)
    for r in results:
        by_reader[r.reader][entry_of[r.concept]] = ctx.names(r.frame)
    arms = {"read": by_reader[settings.get("reader", "linker")], "gold": by_reader["oracle"], "random": by_reader["random"]}
    output.mkdir(parents=True, exist_ok=True)
    rows_fvt, info_fvt = fvt_rows(run, sorted(entries))
    torch.save({"entries": torch.tensor(sorted(entries)), "rows": rows_fvt.cpu()}, output / "fvt_rows.pt")
    items = score_round2_items(config, run, {int(e) for e in entry_of.values()}, output, log=log) \
        if settings.get("items", True) else {"items": 0}
    base = general_window_losses(run.model, run.config, Path(run.config["data"]["eval"]).with_name("eval-general"),
                                 windows=int(config["round2"].get("general_windows", 256)), device=run.device)
    np.save(output / "general_base.npy", base)
    summary = {"learned_edges_written": added, "round2_entries": len(entries), "definitions_read": len(read_set.concepts),
               "style": style, "frames": e11.summarize_frames(rows),
               "reading_cost": {"forward_tokens": scorer.forward_tokens, "training_token_equivalent": scorer.forward_tokens / 3.0,
                                "rows_scored": scorer.rows, "headword_unlinked": scorer.unlinked_rows},
               "fvt": info_fvt, "general_base_loss": float(base[0].sum() / max(1.0, base[1].sum())),
               "items": {"items": items["items"], "accuracy": {c: v["sum"]["mean"] for c, v in items["summary"]["sets"]["all"]["accuracy"].items()}
                         if items.get("summary") else None,
                         "contrasts": [{k: c[k] for k in ("a", "b", "mean", "ci_low", "ci_high") if k in c}
                                       for c in items["summary"]["sets"]["all"]["contrasts"]] if items.get("summary") else None},
               "seconds": time.monotonic() - started}
    _json(output / "frames.json", {"arms": {k: {str(e): f for e, f in v.items()} for k, v in arms.items()},
                                   "readers": {k: {str(e): f for e, f in v.items()} for k, v in by_reader.items()}})
    _json(output / "summary.json", summary)
    with gzip.open(output / "frame_rows.jsonl.gz", "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
    log(f"write: {len(read_set.concepts)} definitions read; reader F1 "
        + ", ".join(f"{k} {v.get('f1')}" for k, v in summary["frames"].get(style, {}).items()))
    return summary


ITEM_FAMILIES = {("negation", "affirm"): "relation", ("paraphrase", "paraphrase"): "property"}
ITEM_CONDITIONS = ("none", "store:linker", "store:oracle", "store:random", "definition-in-context")


def round2_rank_items(config: dict[str, Any], family: str, definitions: dict[int, dict[str, str]]) -> list[Any]:
    """Stage 2's relation and property items in the ranking harness's format (`benchmarks.ranking`, `rank-items/1`): the
    E9 understanding items `negation/affirm` (relation) and `paraphrase` (property) of the round-2 anchors with a
    definition, one item per template; the term is written (`insert: true`) with its gold frame and the definition the
    harness's readers read."""
    from ..benchmarks.ranking import Item, Term
    from . import e9_understanding as und
    _, concepts, items = und.load_items(item_path(config, "understanding", family))
    by = {c["concept"]: c for c in concepts}
    out = []
    for item in items:
        kind = ITEM_FAMILIES.get((item["family"], item["test"]))
        anchor = by[item["anchor"]]
        entry = anchor.get("entry")
        if kind is None or item["subset"] != "heldout" or entry is None or int(entry) not in definitions:
            continue
        fills = {slot: by[cid]["surface"] for slot, cid in item["slots"].items()} | dict(item["text"])
        term = Term(anchor["surface"], str(anchor.get("source") or anchor["surface"]), tuple(tuple(e) for e in anchor["frame"]),
                    definitions[int(entry)]["text"], True)
        for k, template in enumerate(item["templates"]):
            out.append(Item(f"{item['id']}#{k}", kind, und.render(template, fills), [und.render(c, fills) for c in item["candidates"]],
                            int(item["gold"]), [term], meta={"anchor": item["anchor"], "entry": int(entry), "test": item["test"],
                                                             "template": k}))
    return out


def score_round2_items(config: dict[str, Any], run: Any, entries: set[int], output: Path, *, log: Callable[[str], None] = print
                       ) -> dict[str, Any]:
    """The stage-2 items under `write.item_conditions` (no row, the reader's / gold / random rows, the definition in
    context), scored by the shared harness (`ranking.Evaluation`: summed log-probability, ties 1/k); per item rows to
    `items.jsonl.gz`, accuracies and contrasts with `none` (paired over items) to `items-summary.json`."""
    from ..benchmarks.ranking import Evaluation, parse_condition, summarize
    settings = config.get("write") or {}
    definitions = prose_definitions(item_path(config, "read_set", run_family(config, run)), entries, str(settings.get("style", "prose")))
    items = round2_rank_items(config, run_family(config, run), definitions)
    if not items:
        return {"items": 0}
    conditions = [parse_condition(c) for c in settings.get("item_conditions", ITEM_CONDITIONS)]
    evaluation = Evaluation(run, items, batch_size=int(settings.get("item_batch", 32)), log=log)
    result = evaluation.evaluate(conditions)
    summary = summarize(result["items"], [c.name for c in conditions], resamples=int((config.get("statistics") or {}).get("resamples", 2000)))
    with gzip.open(output / "items.jsonl.gz", "wt") as handle:
        for row in result["items"].values():
            handle.write(json.dumps(row, default=str) + "\n")
    record = {"items": len(items), "terms": len(evaluation.terms), "readers": evaluation.reader_summary(),
              "scorer": dict(evaluation.scorer.stats), "timing": result["timing"], "summary": summary}
    _json(output / "items-summary.json", record)
    return record


def arm_frames(written: Path, arm: str, store_names: tuple[dict[str, int], dict[str, int]]) -> dict[int, list[tuple[int, int]]]:
    """The round-2 frames of an arm (ids in the given relation / atom maps; unknown names dropped)."""
    key = ARM_FRAMES[arm]
    if key is None:
        return {}
    relation_id, atomic_id = store_names
    frames = json.loads((Path(written) / "frames.json").read_text())["arms"][key]
    return {int(e): [(relation_id[r], atomic_id[a]) for r, a in (f or []) if r in relation_id and a in atomic_id]
            for e, f in frames.items()}


# ---------------------------------------------------------------- stage 3: reason


def reason(config: dict[str, Any], run_dir: Path, learned: Path | None, written: Path, output: Path, *, device: str | None = None,
           limit: int | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Stage 3 (L5): relation (`negation/affirm`), reverse and two-hop items about round-2 terms, scored with the recall
    of the read store in context (`recall:own`), with no tool (`none`), and with the gold frame (`symbolic`) or the
    definition as text — E12's scoring on the learned store with the read frames written."""
    from . import e12_self_query as e12
    from ..self_query import RecallStore
    started = time.monotonic()
    run = _open(run_dir, config, device=device)
    store = cs.ConceptStore.from_run(run)
    committed(run, store, learned)
    entries = set(round2_entries(run.ontology))
    item_set = e12.load_item_set(item_path(config, "understanding", run_family(config, run)), families=("two_hop", "reverse", "negation", "affirm"))
    anchors = {c["concept"] for c in item_set.concepts if c.get("entry") is not None and int(c["entry"]) in entries}
    prompts = [p for p in item_set.prompts if p.item["subset"] == "heldout" and p.item["anchor"] in anchors]
    cap = int((config.get("reason") or {}).get("max_anchors") or 0)       # opt-in (T7-ROOD): a seeded sample of anchors
    if cap:
        pool = sorted({p.item["anchor"] for p in prompts})
        keep = set(random.Random(int(config.get("seed", 0))).sample(pool, min(cap, len(pool))))
        prompts = [p for p in prompts if p.item["anchor"] in keep]
    if limit:
        keep = sorted({p.item["anchor"] for p in prompts})[:limit]
        prompts = [p for p in prompts if p.item["anchor"] in keep]
    used = {cid for p in prompts for cid in p.item["slots"].values()}
    bridges = {p.item["meta"].get("bridge") for p in prompts if p.item["family"] == "two_hop"}
    item_set.prompts = prompts
    item_set.concepts = [c for c in item_set.concepts if c["concept"] in used or (c.get("role") == "bridge" and c.get("source") in bridges)]
    frames = arm_frames(written, "read", (store.relation_id, store.atomic_id))
    conditions = e12.parse_conditions(config.get("reason", {}).get("conditions", "none,recall:own,symbolic,definition"))
    with store.written({e: frames.get(e) for e in entries}):
        loaded = e12.LoadedStore("own", Path(run_dir), RecallStore(run.composer), run.ontology, run.composer.operator,
                                 store.relation_id, store.atomic_id)
        store_frames = {c["concept"]: store.frame(int(c["entry"])) for c in item_set.concepts if c.get("entry") is not None}
        result = evaluate_items(run, item_set, conditions, {"own": loaded}, store_frames, log=log)
    summary = e12.summarize(item_set, result["results"], result["records"], result["resolved"])
    output.mkdir(parents=True, exist_ok=True)
    units = summary["conditions"]
    _json(output / "summary.json", {"items": len(prompts), "anchors": len({p.item["anchor"] for p in prompts}),
                                    "anchor_of": {p.id: p.item["anchor"] for p in prompts},
                                    "conditions": {k: v["items"]["by"] for k, v in units.items()},
                                    "units": {k: v["items"]["units"] for k, v in units.items()}, "timings": result["timings"],
                                    "linked": summary["linked"], "seconds": time.monotonic() - started})
    with gzip.open(output / "recalls.jsonl.gz", "wt") as handle:
        for name, records in result["records"].items():
            for record in records:
                handle.write(json.dumps({"condition": name, **record}, default=str) + "\n")
    return summary


def evaluate_items(run: Any, item_set: Any, conditions: Sequence[Any], stores: dict[str, Any], store_frames: dict[str, list],
                   *, log: Callable[[str], None] = print) -> dict[str, Any]:
    """`e12_self_query.evaluate` with `CycleContextBuilder` (relation items get the anchor's whole recalled frame)."""
    from . import e12_self_query as e12
    family = item_set.manifest.get("family") or run.config.get("e9_family") or "smollm2"
    lexicon = e12.lexicon_for_track(item_set.track, family, run.ontology)
    builder = CycleContextBuilder(item_set, run.ontology, lexicon, stores, store_frames=store_frames)
    results, records, timings = {}, {}, {}
    with e12.host_view(run, item_set) as (adapter, ids):
        resolved = e12.link_status(adapter, item_set, ids)
        cache = None
        for condition in conditions:
            t0 = time.monotonic()
            contexts, recs = builder.build(condition)
            prompts = [p for p in item_set.prompts if condition.kind == "none" or p.id in contexts]
            log(f"  reason: {condition.name} — {len(prompts)} items")
            with e12.smaller_batches(adapter, 2 if contexts else 1):
                rows, cache, _ = e12.score_prompts(adapter, prompts, contexts, cache)
            results[condition.name], records[condition.name] = rows, recs
            timings[condition.name] = round(time.monotonic() - t0, 1)
    return {"results": results, "records": records, "resolved": resolved, "timings": timings}


def _cycle_builder_base() -> type:
    from .e12_self_query import ContextBuilder
    return ContextBuilder


class CycleContextBuilder(_cycle_builder_base()):          # type: ignore[misc]
    """E12's context builder on the read store: whole-frame recalls decode the store's own slots (the relations the
    read frame holds, which the tool knows) and the relation items (`negation/affirm`) get the anchor's recall."""

    def __init__(self, *args: Any, store_frames: dict[str, list], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.store_frames = store_frames

    def _frame_lines(self, condition: Any, concept: str, *, vector_of: str | None = None) -> list:
        if condition.kind == "symbolic" or concept not in self.store_frames:
            return super()._frame_lines(condition, concept, vector_of=vector_of)
        store = self.store(condition.store)
        slots = [r for r, _ in self.store_frames[concept]]
        return store.store.decode_slots(self.vector(store, concept), slots, cleanup=self.cleanup) if slots else []

    def _understanding(self, condition: Any) -> tuple[dict[str, str], list[dict[str, Any]]]:
        texts, records = super()._understanding(condition)
        writer = self.writer(confidence=condition.kind != "noconf")
        for prompt in self.items.prompts:
            if prompt.item["family"] != "negation":
                continue
            concept = prompt.item["anchor"]
            surface = self.concepts[concept]["surface"]
            text = self._definition(concept) if condition.kind == "definition" else writer.render(surface, self._frame_lines(condition, concept))
            texts[prompt.id] = text
            records.append({"condition": condition.name, "item": prompt.id, "text": text})
        return texts, records


# ---------------------------------------------------------------- stages 4 and 5: round 2


def round2_name(host: str, arm: str, seed: int, scheme: str | None = None) -> str:
    return f"{host}-{arm if scheme is None else f'{arm}-{scheme}'}-s{seed}"


def round2_config(config: dict[str, Any], host: str, arm: str, seed: int, *, scheme: str | None = None) -> dict[str, Any]:
    """The trainer config of one round-2 run (stage 4: `scheme` None; stage 5: a quantization scheme) with its `e13`
    materialization block (read by `materialize` before training)."""
    family = host_family(config, host)
    root = data_root(config, family)
    settings = config["round2"]
    if arm in REFERENCE_ARMS:
        source = Path(config["stage0"]["references"][arm][host].format(seed=seed))
        base = apply_overrides(yaml.safe_load((source / "resolved_config.yaml").read_text()), config)
        init = source / "final.pt"
    else:
        base = stage0_config(config, host, seed)
        init = stage0_dir(config, host, seed) / "final.pt"
    run = copy.deepcopy(base)
    name = round2_name(host, arm, seed, scheme)
    arm_root = root / "round2" / name
    tokens_per_step = int(run["model"]["seq_len"]) * int(run["train"]["micro_batch"]) * int(run["train"]["grad_accum"])
    total = int(settings["tokens"])
    run["data"].update(train=str(root / ("round2-defs" if arm == "defs" else "round2") / "train"), ontology=str(arm_root / "ontology.pt"),
                       seed=int(settings.get("data_seed", 4321)))
    run["train"].update(total_tokens=total, init_from=str(init), init_mode="continue", checkpoint_minutes=10,
                        warmup_tokens=min(int(run["train"].get("warmup_tokens", total)), max(tokens_per_step, total // 10)))
    run["eval"].update(points=[int(p) for p in settings["eval_points"]], reference_strata=str(reference_path(config, family, run)),
                       save_window_losses=True)
    if run["channel"]["mode"] == "compose":
        run["channel"]["skip_empty_frames"] = True
    if arm in ("fvt", "C2"):
        run["channel"]["entry_rows"] = {"path": str(arm_root / "entry_rows.pt"), "trainable": True}
    if scheme is not None:
        lora = arm.startswith("qlora")
        run["model"]["host_mode"] = "lora" if lora else "frozen"
        if lora:
            run["model"]["lora_rank"] = int(settings.get("lora_rank", 64))
            run["train"]["host_lr"] = float(settings.get("lora_host_lr", 2e-4))
        else:
            run["train"].pop("host_lr", None)
        if base["model"]["host_mode"] == "lora":           # a LoRA stage 0 (Qwen3): its adapters are part of the host
            run["train"]["init_merge_lora"] = True
        run["model"]["host_quantization"] = {"scheme": scheme, "group_size": settings.get("group_size"),
                                             "calibration_windows": int(settings.get("calibration_windows", 64))}
        run["train"]["save_trainable_only"] = True
    run["experiment"] = f"e13-{config['track']}-round2-{name}"
    stage = 5 if scheme else 4
    run["e13"] = {"stage": stage, "arm": arm, "host": host, "seed": seed, "scheme": scheme, "family": family, "arm_root": str(arm_root),
                  "seed_ontology": str(root / "seed" / "ontology.pt"),
                  "learned": None if arm in ("C0p",) or not settings.get("use_learned", True) else str(cycle_dir(config, host, seed) / "learn"),
                  "written": str(cycle_dir(config, host, seed) / "write"), "frames": ARM_FRAMES[arm],
                  "general": str(Path(run["data"]["eval"]).with_name("eval-general")), "general_windows": int(settings.get("general_windows", 256))}
    if settings.get("extra_evals"):                        # opt-in (T7-ROOD: forgetting on `eval-round1`): start / end losses
        run["e13"]["extra"] = {name: str(rounds_root(config, family) / split) for name, split in settings["extra_evals"].items()}
    return run


def materialize(run: dict[str, Any]) -> dict[str, Any]:
    """The arm's ontology (seed + learned edges + the arm's round-2 frames; round-2 entries linked in training, only the
    synthetic zero-shot entries stay held out) and entry rows (`fvt`: the stage-2 subtoken-mean rows; `C2`: the C2 run's
    fallback row for every round-2 entry). Idempotent."""
    spec = run["e13"]
    root = Path(spec["arm_root"])
    done = root / "materialized.json"
    if done.exists():
        return json.loads(done.read_text())
    root.mkdir(parents=True, exist_ok=True)
    ontology = torch.load(spec["seed_ontology"], weights_only=False)
    relation_id = {n: i for i, n in enumerate(ontology["relation_names"])}
    atoms = list(ontology["atomic_names"])
    atomic_id = {n: i for i, n in enumerate(atoms)}
    frames = frames_of(ontology)
    learned = 0
    if spec.get("learned"):
        for e, r, a in json.loads((Path(spec["learned"]) / "accepted.json").read_text())["edges"]:
            if (relation_id[r], atomic_id[a]) not in frames[int(e)]:
                frames[int(e)].append((relation_id[r], atomic_id[a])); learned += 1
    entries = round2_entries(ontology)
    written = 0
    if spec.get("frames"):
        names = json.loads((Path(spec["written"]) / "frames.json").read_text())["arms"][spec["frames"]]
        for e, frame in names.items():
            edges = []
            for r, a in frame or []:
                if a not in atomic_id:                     # a new atomic the reader introduced
                    atomic_id[a] = len(atoms); atoms.append(a)
                edges.append((relation_id[r], atomic_id[a]))
            frames[int(e)] = edges
            written += bool(edges)
    out = with_frames(ontology, frames, atomic_names=atoms,
                      heldout_entries=sorted(int(e) for e in ontology.get("synthetic_entries", ())))
    out["e13"] = {**ontology.get("e13", {}), "arm": spec["arm"], "learned_edges": learned, "round2_frames": written}
    torch.save(out, root / "ontology.pt")
    record = {"arm": spec["arm"], "learned_edges": learned, "round2_frames_written": written, "round2_entries": len(entries)}
    if spec["arm"] == "fvt":
        shutil.copyfile(Path(spec["written"]) / "fvt_rows.pt", root / "entry_rows.pt")
    elif spec["arm"] == "C2":
        record["c2_fallback"] = _c2_fallback_rows(run, entries, root / "entry_rows.pt")
    _json(done, record)
    return record


@torch.no_grad()
def _c2_fallback_rows(run: dict[str, Any], entries: Sequence[int], path: Path) -> dict[str, Any]:
    from ..training.lm import load_final
    model = load_final(Path(run["train"]["init_from"]))           # its held-out entries (round 2 among them) are unseen
    rows = model.channel.rows({"entry": torch.tensor(sorted(entries))}).float()
    torch.save({"entries": torch.tensor(sorted(entries)), "rows": rows}, path)
    return {"rows": int(rows.shape[0]), "norm": float(rows.norm(dim=-1).mean())}


def initial_model(config: dict[str, Any], device: torch.device) -> Any:
    """The model a round-2 run starts from (the trainer's construction: host, channel, continuation, quantization)."""
    from ..training import lm
    config = lm.resolve_config(config)
    ontology = torch.load(config["data"]["ontology"], weights_only=False)
    base = lm.build_model(config)
    channel, context = lm.build_channel(config, ontology, base.get_input_embeddings().weight.shape[1], host=base)
    if channel is not None:
        channel.set_unseen(ontology["heldout_entries"])
    model = lm.wrap_host(config, base, channel, context).to(device)
    lm.load_continuation_state(model, torch.load(config["train"]["init_from"], weights_only=False, map_location="cpu"),
                               merge_lora=bool(config["train"].get("init_merge_lora", False)))
    if config["model"].get("host_quantization"):
        lm.quantize_frozen_host(model, config)
    return model.eval()


def run_round2(config_path: Path, output: Path, *, resume: bool = False, keep_checkpoint: bool = False) -> dict[str, Any]:
    """One stage-4/5 job: materialize the arm, measure the starting model's general-text loss, train, measure it again
    (locality), drop the checkpoint (the final state and the per-window losses are what the report reads)."""
    from ..training.lm import load_final, train
    run = yaml.safe_load(Path(config_path).read_text())
    record = materialize(run)
    device = torch.device(run.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    output.mkdir(parents=True, exist_ok=True)
    general, windows = Path(run["e13"]["general"]), int(run["e13"]["general_windows"])
    base_file = Path(run["e13"]["arm_root"]) / "general-start.npy"
    extra = {name: Path(path) for name, path in (run["e13"].get("extra") or {}).items()}
    starts = {name: Path(run["e13"]["arm_root"]) / f"{name}-start.npy" for name in extra}
    if (not base_file.exists() and general.exists()) or any(not starts[n].exists() for n in extra):
        model = initial_model(run, device)
        if not base_file.exists() and general.exists():
            np.save(base_file, general_window_losses(model, run, general, windows=windows, device=device))
        for name, path in extra.items():
            if not starts[name].exists():
                np.save(starts[name], general_window_losses(model, run, path, windows=windows, device=device))
        del model
    result = train(run, output, resume=resume)
    if (output / "final.pt").exists() and not (output / "locality.json").exists() and general.exists():
        model = load_final(output / "final.pt", device)
        end = general_window_losses(model, run, general, windows=windows, device=device)
        start = np.load(base_file)
        np.save(output / "general_windows.npy", np.stack([start, end]))
        _json(output / "locality.json", {"start": float(start[0].sum() / start[1].sum()), "end": float(end[0].sum() / end[1].sum()),
                                         "windows": windows, "materialized": record})
        for name, path in extra.items():                    # e.g. forgetting on round-1 evaluation text
            first, last = np.load(starts[name]), general_window_losses(model, run, path, windows=windows, device=device)
            np.save(output / f"{name}_windows.npy", np.stack([first, last]))
            _json(output / f"{name}.json", {"start": float(first[0].sum() / first[1].sum()), "end": float(last[0].sum() / last[1].sum()),
                                            "windows": windows, "corpus": str(path)})
    if not keep_checkpoint and (output / "final.pt").exists() and (output / "manifest.json").exists():
        (output / "checkpoint.pt").unlink(missing_ok=True)
    return result


# ---------------------------------------------------------------- analysis


def load_curves(run_dir: Path, stratum: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(tokens, sums, counts) of a run's per-window losses of one stratum: points × windows."""
    from ..training.lm import load_window_losses
    data = load_window_losses(Path(run_dir) / "eval_windows.npz")
    index = data["strata"].index(stratum)
    tokens = np.asarray(sorted(data["evals"]))
    sums = np.stack([data["evals"][t][0][index] for t in tokens]).astype(np.float64)
    counts = np.stack([data["evals"][t][1][index] for t in tokens]).astype(np.float64)
    return tokens, sums, counts


def tokens_to_criterion(tokens: np.ndarray, losses: np.ndarray, target: float) -> float:
    """First token count at which the loss curve reaches `target` (≤), linear interpolation; inf if never."""
    if losses[0] <= target:
        return float(tokens[0])
    for (x0, y0), (x1, y1) in zip(zip(tokens, losses), zip(tokens[1:], losses[1:])):
        if y1 <= target:
            return float(x1 if y1 == y0 else x0 + (y0 - target) / (y0 - y1) * (x1 - x0))
    return float("inf")


def aulc(tokens: np.ndarray, losses: np.ndarray) -> float:
    """Mean loss over the token axis (trapezoid / range): lower is better."""
    return float(np.trapezoid(losses, tokens) / (tokens[-1] - tokens[0]))


def efficiency(curves: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]], candidate: str, reference: str, *,
               offset: float = 0.0, resamples: int = 2000, seed: int = 0) -> dict[str, Any]:
    """L2: tokens to criterion candidate / reference, criterion = the reference's final loss per seed; the windows (shared
    by every arm and seed) and the seeds are resampled together (pigeonhole bootstrap). `offset`: training-token
    equivalents added to the candidate's tokens (the reading cost). Also the AULC difference."""
    keep = np.ones(curves[candidate][0][1].shape[1], dtype=bool)        # windows with targets in every run of both arms
    for (_, _, nc), (_, _, nr) in zip(curves[candidate], curves[reference]):
        keep &= (nc[0] > 0) & (nr[0] > 0)
    pairs = [((tc, sc[:, keep], nc[:, keep]), (tr, sr[:, keep], nr[:, keep]))
             for (tc, sc, nc), (tr, sr, nr) in zip(curves[candidate], curves[reference])]
    n_windows, n_seeds = int(keep.sum()), len(pairs)
    rng = np.random.default_rng(seed)

    def statistic(w: np.ndarray, s: np.ndarray) -> tuple[float, float]:
        ttc_c = ttc_r = 0.0
        area = 0.0
        for weight, ((tc, sc, nc), (tr, sr, nr)) in zip(s, pairs):
            if weight == 0:
                continue
            lc, lr = (sc @ w) / (nc @ w), (sr @ w) / (nr @ w)
            ttc_c += weight * (tokens_to_criterion(tc, lc, lr[-1]) + offset)
            ttc_r += weight * tokens_to_criterion(tr, lr, lr[-1])
            area += weight * (aulc(tc, lc) - aulc(tr, lr))
        return ttc_c / ttc_r if ttc_r > 0 else float("nan"), area / s.sum()

    ratio, area = statistic(np.ones(n_windows), np.ones(n_seeds))
    draws = np.asarray([statistic(rng.multinomial(n_windows, np.full(n_windows, 1 / n_windows)).astype(float),
                                  rng.multinomial(n_seeds, np.full(n_seeds, 1 / n_seeds)).astype(float)) for _ in range(resamples)])
    ratios = np.sort(draws[~np.isnan(draws[:, 0]), 0])                  # ∞ (never reached) is a valid draw
    logs = np.log(np.maximum(ratios, 1e-300))
    p = float(min(1.0, 2 * (min((logs >= 0).sum(), (logs <= 0).sum()) + 1) / (ratios.size + 1))) if ratios.size else 1.0

    def interval(values: np.ndarray) -> list[float | None]:
        """Percentile interval by order statistics (no interpolation, so infinite draws are allowed)."""
        if not values.size:
            return [None, None]
        return [float(values[int(np.floor(0.025 * (values.size - 1)))]), float(values[int(np.ceil(0.975 * (values.size - 1)))])]
    low, high = interval(ratios)
    return {"ratio": ratio, "ci_low": low, "ci_high": high, "p_value": p, "aulc_difference": area,
            "aulc_ci": interval(np.sort(draws[~np.isnan(draws[:, 1]), 1])), "offset_tokens": offset, "seeds": n_seeds,
            "windows": n_windows, "resamples": resamples}


def window_table(curves_a: list, curves_b: list, point: int = 0) -> np.ndarray:
    """windows × seeds table of per-window mean loss differences a − b at evaluation `point` (index; −1 = the end),
    windows without targets in the stratum dropped."""
    columns, keep = [], None
    for (_, sa, na), (_, sb, nb) in zip(curves_a, curves_b):
        with np.errstate(invalid="ignore", divide="ignore"):
            columns.append(sa[point] / na[point] - sb[point] / nb[point])
        valid = (na[point] > 0) & (nb[point] > 0)
        keep = valid if keep is None else keep & valid
    return np.stack(columns, 1)[keep]


def paired_windows(curves_a: list, curves_b: list, point: int, resamples: int) -> dict[str, Any]:
    """The paired window-level difference a − b at an evaluation point: windows × seeds pigeonhole bootstrap
    (`statistics.two_way_cluster_bootstrap`); not evaluable (None) without a window holding targets."""
    table = window_table(curves_a, curves_b, point)
    if not table.size:
        return {"mean": float("nan"), "ci_low": None, "ci_high": None, "p_value": None, "clusters": 0}
    return two_way_cluster_bootstrap(table, resamples=resamples)


def _stage_runs(config: dict[str, Any], host: str, arms: Sequence[str], seeds: Sequence[int], scheme: str | None = None
                ) -> dict[str, list[Path]]:
    out = {}
    for arm in arms:
        paths = [run_root(config) / "round2" / round2_name(host, arm, s, scheme) for s in seeds]
        if all((p / "eval_windows.npz").exists() for p in paths):
            out[arm] = paths
    return out


def report(config: dict[str, Any], output: Path, *, log: Callable[[str], None] = print) -> dict[str, Any]:
    """L1–L5 with their pre-registered statistics and Holm across the five primary tests (`summary.json`, `report.md`)."""
    stats_cfg = config.get("statistics") or {}
    resamples = int(stats_cfg.get("resamples", 2000))
    result: dict[str, Any] = {"track": config["track"], "label": config.get("label"), "hosts": {}}
    for host, spec in config["hosts"].items():
        seeds = [int(s) for s in spec["seeds"]]
        block: dict[str, Any] = {}
        runs = _stage_runs(config, host, list(spec.get("round2_arms", PRIMARY_ARMS)), seeds)
        curves = {arm: [load_curves(p, "ref_round2") for p in paths] for arm, paths in runs.items()}
        if {"read", "noread"} <= set(curves):
            block["L1"] = paired_windows(curves["read"], curves["noread"], 0, resamples)
            cost = [json.loads((cycle_dir(config, host, s) / "write" / "summary.json").read_text())["reading_cost"]["training_token_equivalent"]
                    for s in seeds if (cycle_dir(config, host, s) / "write" / "summary.json").exists()]
            offset = float(np.mean(cost)) if cost else 0.0
            block["L2"] = efficiency(curves, "read", "noread", offset=offset, resamples=resamples)
            block["L2_without_reading_cost"] = efficiency(curves, "read", "noread", resamples=resamples)
            block["secondary_vs_noread"] = {arm: {"step0": paired_windows(curves[arm], curves["noread"], 0, resamples),
                                                  "end": paired_windows(curves[arm], curves["noread"], -1, resamples),
                                                  "efficiency": efficiency(curves, arm, "noread", resamples=resamples)}
                                            for arm in curves if arm not in ("read", "noread") and arm not in REFERENCE_ARMS}
        round1 = {arm: [load_curves(p, "ref_round1") for p in paths] for arm, paths in runs.items()}
        block["forgetting"] = {arm: float(np.mean([(s[-1].sum() / n[-1].sum()) - (s[0].sum() / n[0].sum()) for _, s, n in c]))
                               for arm, c in round1.items()}
        block["locality"] = {arm: float(np.mean([json.loads((p / "locality.json").read_text())["end"] -
                                                 json.loads((p / "locality.json").read_text())["start"] for p in paths
                                                 if (p / "locality.json").exists()] or [float("nan")])) for arm, paths in runs.items()}
        for name in (config["round2"].get("extra_evals") or {}):   # e.g. forgetting on `eval-round1` (rounds tracks)
            block[f"{name}_change"] = {arm: float(np.mean([json.loads((p / f"{name}.json").read_text())["end"] -
                                                           json.loads((p / f"{name}.json").read_text())["start"] for p in paths
                                                           if (p / f"{name}.json").exists()] or [float("nan")])) for arm, paths in runs.items()}
        block["curves"] = {arm: {"tokens": c[0][0].tolist(), "loss": np.mean([(s.sum(1) / n.sum(1)) for _, s, n in c], 0).tolist()}
                           for arm, c in curves.items()}
        l3 = {}
        for scheme, arms in (spec.get("frozen_arms") or {}).items():
            frozen = _stage_runs(config, host, arms, seeds, scheme)
            fc = {arm: [load_curves(p, "ref_round2") for p in paths] for arm, paths in frozen.items()}
            entry: dict[str, Any] = {}
            if {"q4-read", "q4-noread"} <= set(fc):
                entry["read_vs_noread_end"] = paired_windows(fc["q4-read"], fc["q4-noread"], -1, resamples)
                entry["read_vs_noread_step0"] = paired_windows(fc["q4-read"], fc["q4-noread"], 0, resamples)
            if {"q4-read", "qlora"} <= set(fc):
                diff = paired_windows(fc["q4-read"], fc["qlora"], -1, resamples)
                reference = float(np.mean([s[-1].sum() / n[-1].sum() for _, s, n in fc["qlora"]]))
                margin = float(stats_cfg.get("noninferiority_relative", 0.01)) * reference
                entry["vs_qlora_end"] = diff | {"margin": margin, "noninferior": diff["ci_high"] is not None and diff["ci_high"] < margin}
            entry["bytes"] = {arm: _bytes(paths[0]) for arm, paths in frozen.items()}
            l3[scheme] = entry
        block["L3"] = l3
        learned = [json.loads((cycle_dir(config, host, s) / "learn" / "summary.json").read_text()) for s in seeds
                   if (cycle_dir(config, host, s) / "learn" / "summary.json").exists()]
        if learned:
            hits, accepted = sum(m["accepted_gold"] for m in learned), sum(m["accepted"] for m in learned)
            nulls, null_acc = sum(m["null"]["null_proposals"] for m in learned), sum(m["null"]["accepted"] for m in learned)
            from scipy import stats
            block["L4"] = {"precision": hits / accepted if accepted else float("nan"), "precision_ci": list(wilson_interval(hits, accepted)) if accepted else None,
                           "accepted": accepted, "erased_gold": sum(m["erased_gold"] for m in learned),
                           "recall": hits / max(1, sum(m["erased_gold"] for m in learned)),
                           "null_rate": null_acc / nulls if nulls else float("nan"), "null_ci": list(wilson_interval(null_acc, nulls)) if nulls else None,
                           "p_value": float(stats.binom.sf(hits - 1, accepted, float(stats_cfg.get("l4_precision", 0.8)))) if accepted else 1.0}
            worst = [m["null"]["rate_max"] for m in learned if m["null"].get("rate_max") is not None]
            if worst:                                       # TK-L (rounds tracks): the worst of E10.L's null worlds
                block["L4"].update(null_rate=float(max(worst)), null_rate_pooled=null_acc / nulls if nulls else float("nan"),
                                   rule=learned[0].get("rule"))
        scored = [json.loads((cycle_dir(config, host, s) / "write" / "items-summary.json").read_text()) for s in seeds
                  if (cycle_dir(config, host, s) / "write" / "items-summary.json").exists()]
        if scored:                                          # stage-2 relation / property items (secondary; ranking harness)
            block["write_items"] = {"accuracy": [{c: v["sum"]["mean"] for c, v in i["summary"]["sets"]["all"]["accuracy"].items()}
                                                 for i in scored],
                                    "contrasts_vs_none": [i["summary"]["sets"]["all"]["contrasts"] for i in scored]}
        reasons = [json.loads((cycle_dir(config, host, s) / "reason" / "summary.json").read_text()) for s in seeds
                   if (cycle_dir(config, host, s) / "reason" / "summary.json").exists()]
        if reasons and all("recall:own" in r["units"] and "none" in r["units"] for r in reasons):
            block["L5"] = _l5(reasons, resamples)
        result["hosts"][host] = block
    primary_host = next(iter(config["hosts"]))
    p = result["hosts"].get(primary_host, {})
    tests = {"L1": p.get("L1", {}).get("p_value"), "L2": p.get("L2", {}).get("p_value"),
             "L3": (p.get("L3", {}).get("rtn", {}).get("read_vs_noread_end") or {}).get("p_value"),
             "L4": p.get("L4", {}).get("p_value"), "L5": p.get("L5", {}).get("p_value")}
    present = {k: v for k, v in tests.items() if v is not None}
    adjusted = dict(zip(present, holm_adjust(list(present.values())))) if present else {}
    result["holm"] = {k: {"p": present[k], "p_holm": adjusted[k]} for k in present}
    result["verdicts"] = verdicts(p, adjusted, stats_cfg)
    output.mkdir(parents=True, exist_ok=True)
    _json(output / "summary.json", result)
    (output / "report.md").write_text(render_report(result))
    log(f"report: {result['verdicts']}")
    return result


def _bytes(run_dir: Path) -> dict[str, Any]:
    manifest = Path(run_dir) / "manifest.json"
    if not manifest.exists():
        return {}
    m = json.loads(manifest.read_text())
    q = m.get("host_quantization") or {}
    final = torch.load(Path(run_dir) / "final.pt", weights_only=False, map_location="cpu") if (Path(run_dir) / "final.pt").exists() else {}
    adapters = sum(v.numel() for k, v in (final.get("model") or {}).items() if k.endswith((".lora_a", ".lora_b")))
    trainable = int(m.get("channel_parameters") or 0) + adapters
    return {"host_bytes": q.get("host_bytes"), "bits_per_weight": q.get("bits_per_weight"), "adapter_parameters": adapters,
            "trainable_parameters": trainable, "trainable_bytes_fp32": 4 * trainable}


def _l5(reasons: list[dict[str, Any]], resamples: int) -> dict[str, Any]:
    by_anchor: list[dict[str, float]] = []
    for r in reasons:
        diffs: dict[str, list[float]] = defaultdict(list)
        for item, unit in r["units"]["recall:own"].items():
            none = r["units"]["none"].get(item)
            if none is not None:
                diffs[r["anchor_of"][item]].append(unit["accuracy"] - none["accuracy"])
        by_anchor.append({a: float(np.mean(v)) for a, v in diffs.items()})
    common = sorted(set.intersection(*(set(b) for b in by_anchor)))
    table = np.asarray([[b[a] for b in by_anchor] for a in common])
    out = two_way_cluster_bootstrap(table, resamples=resamples) if table.size else {"mean": float("nan"), "p_value": 1.0}
    out["by_family"] = {}
    for r in reasons:
        for name in ("recall:own", "none", "symbolic", "definition"):
            if name in r["conditions"]:
                out["by_family"].setdefault(name, []).append({k: v.get("accuracy") for k, v in r["conditions"][name].items()})
    return out


def verdicts(block: dict[str, Any], adjusted: dict[str, float], stats_cfg: dict[str, Any]) -> dict[str, Any]:
    alpha = float(stats_cfg.get("holm_alpha", 0.05))
    ok = lambda k: k in adjusted and adjusted[k] < alpha        # noqa: E731
    below = lambda value, bound: value is not None and value == value and value < bound     # noqa: E731
    out: dict[str, Any] = {}
    if "L1" in block:
        out["L1"] = bool(ok("L1") and below(block["L1"]["ci_high"], 0))
    if "L2" in block:
        out["L2"] = bool(ok("L2") and below(block["L2"]["ci_high"], 1))
    l3 = (block.get("L3") or {}).get("rtn") or {}
    if "read_vs_noread_end" in l3:
        out["L3"] = bool(ok("L3") and below(l3["read_vs_noread_end"]["ci_high"], 0))
        out["L3_noninferior_to_qlora"] = bool((l3.get("vs_qlora_end") or {}).get("noninferior"))
    if "L4" in block:
        out["L4"] = bool(block["L4"]["precision"] >= float(stats_cfg.get("l4_precision", 0.8))
                         and block["L4"]["null_rate"] <= float(stats_cfg.get("l4_far", 0.05)) and ok("L4"))
    if "L5" in block:
        out["L5"] = bool(ok("L5") and block["L5"].get("ci_low") is not None and block["L5"]["ci_low"] > 0)
    return out


def render_report(result: dict[str, Any]) -> str:
    label = f"{result['label']} — " if result.get("label") else ""
    lines = [f"# {label}E13 learning cycle — {result['track']}", "", "Pre-registration: `experiments/e13-learning-cycle/preregistration.md`.", "",
             "## Primary endpoints (primary host; Holm across L1–L5)", "", "| endpoint | estimate | 95% CI | p | p (Holm) | met |",
             "|---|---:|---|---:|---:|---|"]
    host = next(iter(result["hosts"]), None)
    block = result["hosts"].get(host, {})
    holm, verdict = result["holm"], result["verdicts"]
    fmt = lambda x: "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.4g}"     # noqa: E731
    rows = {"L1": block.get("L1"), "L2": block.get("L2"), "L3": ((block.get("L3") or {}).get("rtn") or {}).get("read_vs_noread_end"),
            "L5": block.get("L5")}
    for name in ("L1", "L2", "L3", "L4", "L5"):
        if name == "L4" and "L4" in block:
            b = block["L4"]
            lines.append(f"| L4 precision (null rate) | {fmt(b['precision'])} ({fmt(b['null_rate'])}) | {b['precision_ci']} | "
                         f"{fmt(b['p_value'])} | {fmt(holm.get('L4', {}).get('p_holm'))} | {verdict.get('L4')} |")
            continue
        b = rows.get(name)
        if not b:
            continue
        estimate = b.get("ratio", b.get("mean"))
        lines.append(f"| {name} | {fmt(estimate)} | [{fmt(b.get('ci_low'))}, {fmt(b.get('ci_high'))}] | {fmt(b.get('p_value'))} | "
                     f"{fmt(holm.get(name, {}).get('p_holm'))} | {verdict.get(name)} |")
    lines += ["", "## Per host", "", "```json", json.dumps({h: {k: v for k, v in b.items() if k != "curves"} for h, b in result["hosts"].items()},
                                                          indent=1, default=str)[:20000], "```", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- plan (queue commands; nothing is queued)


def measured_tokens_per_s(run_dir: Path) -> float | None:
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if '"type": "train"' in line]
    values = sorted(r.get("train_tokens_per_s") or r["tokens_per_s"] for r in rows)
    return float(values[len(values) // 2]) if values else None


def estimate_hours(config: dict[str, Any], host: str, tokens: int, evals: int, *, windows: int = 1024) -> float:
    """GPU hours of a training job from the measured per-step cost of the host's E9 T5 runs (median train tokens/s of
    `metrics.jsonl`; `measure` in the config) plus the evaluation passes (E9's measured 1,024-window pass, scaled)."""
    from .e9_plan import EVAL_PASS_SECONDS_1024, _host_scale
    speed = measured_tokens_per_s(Path(config["measure"][host])) if (config.get("measure") or {}).get(host) else None
    speed = speed or {"SmolLM2-360M": 12_700.0, "Qwen3-1.7B-Base": 4_200.0}.get(host, 12_700.0)
    eval_seconds = EVAL_PASS_SECONDS_1024.get(host, EVAL_PASS_SECONDS_1024["SmolLM2-360M"] * _host_scale(host)) * windows / 1024
    return (tokens / speed + evals * eval_seconds) / 3600


def evaluation_scale(host: str) -> float:
    """Inference cost of a host relative to SmolLM2-360M (`e9_plan._host_scale`: by parameters)."""
    from .e9_plan import _host_scale
    return _host_scale(host)


# GPU hours of the evaluation-type jobs on SmolLM2-360M (scaled to the host by `evaluation_scale`; `estimate_hours` in the
# config overrides them): learn by rule closure ≈ 2.6M forward tokens; TK-L's learn ≈ 14M forward tokens of hidden-state
# extraction plus E10.L's CPU erasure run (≈ 0.15 h, not scaled); write = readers + harness items; reason = E12 scoring.
JOB_HOURS = {"learn_rule_closure": (0.08, 0.0), "learn_tkl": (0.15, 0.15), "write": (0.3, 0.0), "reason": (0.5, 0.0)}


def job_hours(config: dict[str, Any], kind: str, host: str) -> float:
    scaled, fixed = (config.get("estimate_hours") or {}).get(kind, JOB_HOURS[kind])
    return float(scaled) * evaluation_scale(host) + float(fixed)


def plan(config: dict[str, Any], *, python: str | None = None, write_configs: bool = True) -> list[dict[str, Any]]:
    """Every E13 job of the config: (name, priority level, command, GPU-h estimate, lane note). Writes the trainer configs
    (stage 0, rounds 2) under `configs/<track>/`; prints nothing to the queue (the commands are for queue-commands.sh)."""
    python = python or "$PY"
    track = config["track"]
    levels = config["priority"]
    base, step = float(levels["base"]), float(levels["step"])
    level = lambda k: round(base + k * step, 6)                  # noqa: E731
    slots = levels.get("slots", {"stage0": 0, "stage4": 1, "stage5": 2, "evaluations": 3, "report": 4})
    cfg_root = ROOT / "configs" / track
    jobs: list[dict[str, Any]] = []
    cfg = config["_path"]

    def add(name: str, slot: str, command: list[str], hours: float, *, min_free_gb: int = 10, resume: bool = False) -> None:
        jobs.append({"name": name, "priority": level(slots[slot]), "command": command, "hours": hours, "min_free_gb": min_free_gb,
                     "resume": resume})

    families = sorted({host_family(config, h) for h in config["hosts"]})
    for family in families:
        add(f"e13-{track}-prep-{family}", "stage0", [python, "-m", "vsa_embed.experiments.e13_cycle", "prepare", "--config", cfg,
                                                     "--family", family], 0.15)
    for host, spec in config["hosts"].items():
        for seed in spec["seeds"]:
            run = stage0_config(config, host, seed)
            path = cfg_root / "stage0" / f"{stage0_dir(config, host, seed).name}.yaml"
            if write_configs:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(yaml.safe_dump(run, sort_keys=False))
            evals = 1 + len(_schedule(run))
            add(f"e13-{track}-{stage0_dir(config, host, seed).name}", "stage0",
                [python, "-m", "vsa_embed.training.lm", "--config", str(path), "--output", str(stage0_dir(config, host, seed))],
                estimate_hours(config, host, int(run["train"]["total_tokens"]), evals, windows=int(run["eval"]["windows"])),
                min_free_gb=20, resume=True)
    for host, spec in config["hosts"].items():
        for seed in spec["seeds"]:
            run_dir, cdir = stage0_dir(config, host, seed), cycle_dir(config, host, seed)
            learn_cfg = config.get("learn") or {}
            primary = learn_cfg.get("primary", "rule_closure")
            add(f"e13-{track}-{host}-s{seed}-learn", "stage0", [python, "-m", "vsa_embed.experiments.e13_cycle", "learn", "--config", cfg,
                                                               "--run", str(run_dir), "--output", str(cdir / "learn")],
                job_hours(config, f"learn_{primary}", host))
            for method in learn_cfg.get("secondary", []):           # e.g. rule closure next to TK-L's primary (T7-ROOD)
                add(f"e13-{track}-{host}-s{seed}-learn-{method}", "stage0",
                    [python, "-m", "vsa_embed.experiments.e13_cycle", "learn", "--config", cfg, "--run", str(run_dir), "--method", method,
                     "--output", str(cdir / f"learn-{method}")], job_hours(config, f"learn_{method}", host))
            add(f"e13-{track}-{host}-s{seed}-write", "stage0", [python, "-m", "vsa_embed.experiments.e13_cycle", "write", "--config", cfg,
                                                               "--run", str(run_dir), "--learned", str(cdir / "learn"), "--output",
                                                               str(cdir / "write")], job_hours(config, "write", host))
    tokens = int(config["round2"]["tokens"])
    for host, spec in config["hosts"].items():
        for seed in spec["seeds"]:
            batches = [(arm, None, "stage4") for arm in spec.get("round2_arms", PRIMARY_ARMS)]
            batches += [(arm, scheme, "stage5") for scheme, arms in (spec.get("frozen_arms") or {}).items() for arm in arms]
            for arm, scheme, slot in batches:
                run = round2_config(config, host, arm, seed, scheme=scheme)
                name = round2_name(host, arm, seed, scheme)
                path = cfg_root / "round2" / f"{name}.yaml"
                if write_configs:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(yaml.safe_dump(run, sort_keys=False))
                evals = 1 + len(_schedule(run))
                add(f"e13-{track}-{name}", slot, [python, "-m", "vsa_embed.experiments.e13_cycle", "round2", "--config", str(path),
                                                  "--output", str(run_root(config) / "round2" / name)],
                    estimate_hours(config, host, tokens, evals, windows=int(run["eval"]["windows"])) + 0.03, min_free_gb=20, resume=True)
    for host, spec in config["hosts"].items():
        for seed in spec["seeds"]:
            run_dir, cdir = stage0_dir(config, host, seed), cycle_dir(config, host, seed)
            add(f"e13-{track}-{host}-s{seed}-reason", "evaluations",
                [python, "-m", "vsa_embed.experiments.e13_cycle", "reason", "--config", cfg, "--run", str(run_dir), "--learned",
                 str(cdir / "learn"), "--written", str(cdir / "write"), "--output", str(cdir / "reason")],
                job_hours(config, "reason", host))
    add(f"e13-{track}-report", "report", [python, "-m", "vsa_embed.experiments.e13_cycle", "report", "--config", cfg, "--output",
                                          str(ROOT / "report" / track)], 0.0)
    return jobs


def _schedule(run: dict[str, Any]) -> list[int]:
    from ..training.lm import evaluation_schedule, resolve_config
    resolved = resolve_config(run)
    tokens_per_step = int(resolved["model"]["seq_len"]) * int(resolved["train"]["micro_batch"]) * int(resolved["train"]["grad_accum"])
    return evaluation_schedule(resolved, max(1, int(resolved["train"]["total_tokens"]) // tokens_per_step) * tokens_per_step)


def queue_lines(jobs: Sequence[dict[str, Any]]) -> list[str]:
    """`jobqueue add` lines (fractional priorities; run from the repository root with PY set); training jobs resume on
    retry, the others do not."""
    out = []
    for job in jobs:
        resume = "" if job["resume"] else " --no-resume"
        command = " ".join(job["command"])
        out.append(f"PYTHONPATH=src $PY -m vsa_embed.jobqueue add --name {job['name']} --priority {job['priority']} "
                   f"--min-free-gb {job['min_free_gb']}{resume} -- {command}   # ≈ {job['hours']:.2f} GPU-h")
    return out


# ---------------------------------------------------------------- smoke (CPU; SMOKE)


def smoke_config(data: Path, runs: Path, track: str = "t5") -> dict[str, Any]:
    """A track's config shrunk for a CPU smoke: SmolLM2-135M, 128-token windows, a few steps per stage, small corpora
    (T7-ROOD: the rounds build itself, 8 evaluation windows, a few hundred extraction windows, 40 probes)."""
    config = load_config(ROOT / ("t5.yaml" if track == "t5" else "t7-rood.yaml"))
    config.update(label="SMOKE", data_root=str(data), runs_root=str(runs), families={"SmolLM2-135M": "smollm2"},
                  hosts={"SmolLM2-135M": {"seeds": [1], "round2_arms": ["read", "noread", "fvt"],
                                          "frozen_arms": {"rtn": ["q4-read", "q4-noread", "qlora"]}}}, measure={})
    config["stage0"]["configs"] = {"SmolLM2-135M": f"experiments/e9-retrofit/configs/{e9_track(config)}/SmolLM2-135M-full-C5-s{{seed}}.yaml"}
    config["stage0"]["references"] = {}
    if track == "t5":
        config["text"].update(round2_domain_chars=150_000, validation_chars=400_000)
    else:
        config["stage0"]["eval_windows"] = {"smollm2": 8}
        config["learn"]["tkl"]["extract"] = {"splits": "eval_round1:300,train:300", "window": 128, "max_per_entry": 8, "batch": 4}
        config["learn"]["tkl"]["test"]["min_observations"] = 2
        config["reason"]["max_anchors"] = 4
    config["round2"].update(tokens=512, eval_points=[256], eval_windows=8, seq_len=128, general_windows=4, calibration_windows=2)
    config["learn"]["validation"].update(per_entry=3, length=64, min_windows=2)
    config["statistics"]["resamples"] = 200
    config["run_overrides"] = {"device": "cpu", "model": {"seq_len": 128},
                               "train": {"micro_batch": 1, "grad_accum": 2, "total_tokens": 768, "warmup_tokens": 256,
                                         "log_every": 1, "checkpoint_minutes": 600},
                               "eval": {"windows": 8, "batch": 4, "first_tokens": 256}}
    return config


def smoke(output: Path, *, data: Path | None = None, threads: int = 4, track: str = "t5") -> dict[str, Any]:
    """CPU smoke of the whole cycle on a track (SmolLM2-135M, labelled SMOKE): its round split, stage 0 for 3 steps,
    learn (and the secondary learn methods) / write / reason on a few terms, stage-4 arms read / noread / fvt and stage-5
    arms (RTN) q4-read / q4-noread / qlora for 2 steps, then the report. Run data go to `data` (default: a temporary
    folder); `output` gets the summary."""
    import tempfile
    from ..training.lm import train
    torch.set_num_threads(int(threads))
    data = Path(data) if data else Path(tempfile.mkdtemp(prefix="e13-smoke-"))
    config = smoke_config(data / "data", data / "runs", track)
    host, seed = "SmolLM2-135M", 1
    timings: dict[str, float] = {}
    out: dict[str, Any] = {"label": "SMOKE", "data": str(data), "host": host}

    def timed(name: str, function: Callable[[], Any]) -> Any:
        t0 = time.monotonic()
        value = function()
        timings[name] = round(time.monotonic() - t0, 1)
        return value

    out["prepare"] = timed("prepare", lambda: prepare(config, "smollm2", workers=2))
    stage0 = stage0_config(config, host, seed)
    run0 = stage0_dir(config, host, seed)
    out["stage0"] = timed("stage0", lambda: train(stage0, run0, resume=run0.exists()))
    cdir = cycle_dir(config, host, seed)
    out["learn"] = timed("learn", lambda: learn(config, run0, cdir / "learn", device="cpu", limit=24 if track == "t5" else 40))
    for method in (config.get("learn") or {}).get("secondary", []):
        out[f"learn-{method}"] = timed(f"learn-{method}", lambda: learn(config, run0, cdir / f"learn-{method}", device="cpu", limit=24,
                                                                        method=method))
    present = _round2_in_windows(config, stage0)              # read the round-2 terms the few smoke windows contain
    out["write"] = timed("write", lambda: write(config, run0, cdir / "learn", cdir / "write", device="cpu", limit=8, only=present))
    reasoned = timed("reason", lambda: reason(config, run0, cdir / "learn", cdir / "write", cdir / "reason", device="cpu", limit=4))
    out["reason"] = {k: v["items"]["by"] for k, v in reasoned["conditions"].items()}
    spec = config["hosts"][host]
    jobs = [(arm, None) for arm in spec["round2_arms"]] + [(arm, s) for s, arms in spec["frozen_arms"].items() for arm in arms]
    out["round2"] = {}
    for arm, scheme in jobs:
        run = round2_config(config, host, arm, seed, scheme=scheme)
        name = round2_name(host, arm, seed, scheme)
        path = data / "configs" / f"{name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(run, sort_keys=False))
        out["round2"][name] = timed(f"round2:{name}", lambda: run_round2(path, run_root(config) / "round2" / name,
                                                                         resume=(run_root(config) / "round2" / name).exists()))
    result = timed("report", lambda: report(config, Path(output) / "report"))
    out.update(verdicts=result["verdicts"], holm=result["holm"], timings=timings,
               step_seconds=_step_seconds(run0), cpu_threads=int(threads))
    _json(Path(output) / "smoke.json", out)
    return out


def _round2_in_windows(config: dict[str, Any], run: dict[str, Any]) -> set[int]:
    from ..data.corpus import TokenCorpus, eval_windows
    corpus = TokenCorpus.open(Path(run["data"]["eval"]))
    round2 = set(round2_entries(torch.load(run["data"]["ontology"], weights_only=False)))
    length = int(run["model"]["seq_len"])
    found: set[int] = set()
    for start in eval_windows(corpus, count=int(run["eval"]["windows"]), length=length):
        _, spans = corpus.window(int(start), length, min_subtokens=int(run["data"]["min_subtokens"]))
        found |= {int(e) for e in spans["entry"].tolist()} & round2
    return found


def _step_seconds(run_dir: Path) -> float | None:
    rows = [json.loads(line) for line in (Path(run_dir) / "metrics.jsonl").read_text().splitlines() if '"type": "train"' in line]
    speeds = [r["train_tokens_per_s"] for r in rows if r.get("train_tokens_per_s")]
    return None if not speeds else float(np.median([256 / s for s in speeds]))


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--config", type=Path, required=True); p.add_argument("--family", default="smollm2")
    p.add_argument("--workers", type=int, default=3)
    for name in ("learn", "write", "reason"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, required=True); p.add_argument("--run", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True); p.add_argument("--device", default=None)
        p.add_argument("--limit", type=int, default=None, help="smoke tests only")
        if name != "learn":
            p.add_argument("--learned", type=Path, default=None)
        else:
            p.add_argument("--method", choices=LEARN_METHODS, default=None, help="default: the config's learn.primary")
        if name == "reason":
            p.add_argument("--written", type=Path, required=True)
    p = sub.add_parser("items", help="understanding items of a rounds track's round-2 anchors (CPU)")
    p.add_argument("--config", type=Path, required=True); p.add_argument("--family", default="smollm2")
    p.add_argument("--output", type=Path, default=None)
    p = sub.add_parser("round2"); p.add_argument("--config", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true"); p.add_argument("--keep-checkpoint", action="store_true")
    p = sub.add_parser("report"); p.add_argument("--config", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("plan"); p.add_argument("--config", type=Path, required=True); p.add_argument("--no-write-configs", action="store_true")
    p = sub.add_parser("smoke"); p.add_argument("--output", type=Path, required=True); p.add_argument("--data", type=Path, default=None)
    p.add_argument("--threads", type=int, default=4); p.add_argument("--track", choices=("t5", "t7-rood"), default="t5")
    args = parser.parse_args(argv)
    if args.command == "smoke":
        print(json.dumps(smoke(args.output, data=args.data, threads=args.threads, track=args.track), indent=1, default=str))
        return
    if args.command == "round2":
        print(json.dumps(run_round2(args.config, args.output, resume=args.resume, keep_checkpoint=args.keep_checkpoint), default=str))
        return
    config = load_config(args.config)
    if args.command == "prepare":
        print(json.dumps(prepare(config, args.family, workers=args.workers), indent=1, default=str))
    elif args.command == "learn":
        print(json.dumps(learn(config, args.run, args.output, device=args.device, limit=args.limit, method=args.method), indent=1,
                         default=str))
    elif args.command == "items":
        print(json.dumps(build_round2_items(config, args.family, args.output), indent=1, default=str))
    elif args.command == "write":
        print(json.dumps(write(config, args.run, args.learned, args.output, device=args.device, limit=args.limit), indent=1, default=str))
    elif args.command == "reason":
        summary = reason(config, args.run, args.learned, args.written, args.output, device=args.device, limit=args.limit)
        print(json.dumps({k: v for k, v in summary.items() if k != "conditions"}, default=str))
    elif args.command == "report":
        report(config, args.output)
    elif args.command == "plan":
        jobs = plan(config, write_configs=not args.no_write_configs)
        print("\n".join(queue_lines(jobs)))
        print(f"# total ≈ {sum(j['hours'] for j in jobs):.1f} GPU-h over {len(jobs)} jobs", file=sys.stderr)


if __name__ == "__main__":
    main()
