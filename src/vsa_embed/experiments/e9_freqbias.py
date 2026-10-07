"""E9 frequency bias and the strict non-copy loss stratum (author request 2026-10-07; pre-registration
`experiments/e9-retrofit/preregistration-understanding.md`, families A1, A2 and B2). Evaluation only, on finished runs.

**A1 — frequency bias of the loss (`score`).** A finished run is re-scored on its own evaluation windows (the trainer's
per-token losses, bf16 autocast as `training.lm.evaluate`). Every after-span target gets the training frequency of the
term(s) it follows, binned in log-frequency bins (`--scheme`): the primary `halfdecade` bins 1–3, 4–9, 10–31, 32–99,
100–316, 317–999, 1,000–3,162, ≥ 3,163 are half-decades whose unions are exactly the trainer's `after_rare_seen` (1–9),
`after_mid` (10–99) and `after_frequent` (≥ 100) (checked on every batch); `log2` (1, 2–3, 4–7, …, ≥ 1,024) is the
sensitivity scheme. A target belongs to bin `b` when it is in the 8-token window after a span whose entry's training
frequency falls in `b` (the trainer's union rule: a target after two spans of different bins is in both). Held-out
and unseen terms keep the trainer's `after_heldout` and `after_unseen`. Each bin also has `_filler` (the targets that
belong to an alias of a filler of the frame, `e9_rescore`'s rule) and `_strict` (below).

**B2 — the strict non-copy stratum.** `X_strict` = targets of after-span stratum `X` (every stratum of
`training.lm.AFTER_STRATA` and every frequency bin) that belong to no occurrence, starting after the span, of a token
sequence that verbalizes (i) the term itself (its aliases), (ii) any filler of its frame or (iii) any filler of the frame
of a concept that fills its frame (two hops), or (iv) relation wording. A concept's verbalizations are its aliases and
lexicon text (`e9_rescore.atomic_surfaces`) **and each of their content words** (≥ 3 letters, not a stopword); relation
wording = the content words of the relation names, of the track lexicon's relation templates and (T5) of the
generator's fact templates. Every sequence is matched as written, lower-cased and capitalized, with and without a
leading space (the filler-table rule). The exclusion table is built once per track and tokenizer family (`exclusions`).

**Outputs** (`RUN/freqbias/`, run-folder contract): `windows.npz` (`strata`, `starts`, `count`, `sum_<variant>`:
strata × windows; the trainer's strata replay `eval_windows.npz` exactly — `ref_check`), `terms.npz` (one row per
evaluation window × entry with after-span targets: `window`, `entry`, `frequency`, `heldout`, `count`, `filler_count`,
`strict_count`, `sum_<variant>`, `filler_sum_<variant>`, `strict_sum_<variant>` — the union of the entry's own
after-span windows in that window; the unit of the term-cluster bootstrap), `freqbias.json` (per variant the stratified
losses; bins with target-weighted mean log2 frequency; the nesting check; `ref_check`), `resolved_config.yaml`,
`manifest.json`. Variants: `ref` (default) and `ref-off` (the channel switched off, `e9_rescore`'s rule).

**A2 — frequency decodability of concept rows (`rows`, CPU).** HRRBERT's t-SNE observation (unstructured embeddings
encode code frequency) made quantitative: a cross-validated ridge probe (5 folds, ridge strength by GCV inside each
training fold, `e5_zeroshot.ridge_map`) predicts log2 training frequency from each representation of the run's
concepts — the channel's own table before the projector (C5 and arms: the composed vector; C2: the free row; C6*: the
frozen source row), the injected row after the projector, the host input-embedding mean over the term's subtokens, and
two model-free references (frame degree; a 64-rank spectral embedding of the frame graph); the probe reads each
representation's coordinates plus its L2 norm and log norm (`probe_features`). Entries: not held out, training
frequency ≥ 1 (primary; `all` adds the unseen with log2(1 + f)). Out-of-fold predictions go to `RUN/freqrows/oof.npz`
so the report computes R² with entry-bootstrap intervals paired across representations, models and seeds; `tsne.npz`
and `tsne.png` hold an exact t-SNE (perplexity 30) of 1,500 frequency-stratified entries per representation, coloured
by log2 frequency (HRRBERT's figure).

    python -m vsa_embed.experiments.e9_freqbias exclusions --track t5 [--family smollm2]
    python -m vsa_embed.experiments.e9_freqbias score --run RUN [--variants ref ref-off] [--scheme halfdecade] [--resume]
    python -m vsa_embed.experiments.e9_freqbias rows --run RUN [--no-tsne]
    python -m vsa_embed.experiments.e9_freqbias queue --stage t5 --priority 55 [--models C0p C5 …] [--kind score rows]
        [--dry-run]
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import yaml

from ..data.corpus import TokenCorpus, collate_windows, eval_windows
from ..provenance import git_state, write_run_metadata
from ..row_sources import FillerIndex, frames_digest, load_filler_index
from ..span_channel import AliasTable, normalize_alias
from ..statistics import holm_adjust, paired_ratio_bootstrap
from ..training import lm

ROOT = Path("experiments/e9-retrofit")
EXCLUSION_ROOT = Path("~/data/vsa-llm/e9/exclusion-tables").expanduser()
SCORE_DIR, ROWS_DIR = "freqbias", "freqrows"
SCHEMES: dict[str, tuple[tuple[int, int | None], ...]] = {
    "halfdecade": ((1, 3), (4, 9), (10, 31), (32, 99), (100, 316), (317, 999), (1000, 3162), (3163, None)),
    "log2": ((1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 127), (128, 255), (256, 511), (512, 1023), (1024, None)),
}
PRIMARY_SCHEME = "halfdecade"
COARSE = (("after_rare_seen", 1, 9), ("after_mid", 10, 99), ("after_frequent", 100, None))
VARIANTS = ("ref", "ref-off")
MIN_WORD = 3
# A fixed English stopword list (function words; never content words of a name), so the strict stratum is reproducible
# without NLTK data.
STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before being below between both but
by can cannot could did do does doing down during each few for from further had has have having he her here hers herself
him himself his how i if in into is it its itself just let me more most my myself no nor not now of off on once only or
other ought our ours ourselves out over own same she should so some such than that the their theirs them themselves then
there these they this those through to too under until up upon very was we were what when where which while who whom why
will with would you your yours yourself yourselves one two three via per its it's
""".split())
WORD = re.compile(r"[A-Za-z]+")


# ---------------------------------------------------------------- bins


def bin_name(lo: int, hi: int | None) -> str:
    return f"after_f{lo}plus" if hi is None else f"after_f{lo}_{hi}"


def scheme_bins(scheme: str = PRIMARY_SCHEME) -> list[tuple[str, int, int | None]]:
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {sorted(SCHEMES)}")
    edges = SCHEMES[scheme]
    for (lo, hi), (lo2, _) in zip(edges, edges[1:]):
        if hi is None or lo2 != hi + 1:
            raise ValueError(f"{scheme}: bins must be contiguous")
    if edges[0][0] != 1 or edges[-1][1] is not None:
        raise ValueError(f"{scheme}: bins must cover 1 … ∞")
    return [(bin_name(lo, hi), lo, hi) for lo, hi in edges]


def bin_index(count: int, bins: Sequence[tuple[str, int, int | None]]) -> int | None:
    """Index of the bin holding training frequency `count` (None for 0)."""
    for i, (_, lo, hi) in enumerate(bins):
        if count >= lo and (hi is None or count <= hi):
            return i
    return None


def nesting(bins: Sequence[tuple[str, int, int | None]]) -> dict[str, list[str]]:
    """The bins whose union is each coarse trainer stratum (empty when the scheme does not nest it)."""
    out = {}
    for name, lo, hi in COARSE:
        members = [(b, blo, bhi) for b, blo, bhi in bins if blo >= lo and (hi is None or (bhi is not None and bhi <= hi))]
        ok = bool(members) and members[0][1] == lo and members[-1][2] == hi
        ok &= all(prev[2] is not None and nxt[1] == prev[2] + 1 for prev, nxt in zip(members, members[1:]))
        out[name] = [b for b, _, _ in members] if ok else []
    return out


def span_strata(entry: int, length: int, frequency: np.ndarray | None, heldout: set[int]) -> list[str]:
    """The after-span strata of one span — the rule of `training.lm.stratum_masks` (checked against it per batch)."""
    names = ["after", "after_len1" if length == 1 else "after_len2" if length == 2 else "after_len3plus"]
    if entry in heldout:
        names.append("after_heldout")
    elif frequency is not None:
        count = int(frequency[entry])
        names.append("after_rare" if count < 10 else "after_mid" if count < 100 else "after_frequent")
        if count < 10:
            names.append("after_unseen" if count == 0 else "after_rare_seen")
    return names


# ---------------------------------------------------------------- exclusion table (strict non-copy stratum)


def content_words(text: str) -> list[str]:
    """Content words of a surface (≥ 3 letters, not a stopword), as written."""
    return [w for w in WORD.findall(str(text)) if len(w) >= MIN_WORD and w.lower() not in STOPWORDS]


def relation_texts(track: str | None, ontology: dict[str, Any], lexicon: Any = None) -> list[str]:
    """Texts whose content words are relation wording: relation names, the lexicon's relation templates and (T5) the
    generator's fact templates."""
    texts = [str(r).replace("_", " ") for r in ontology.get("relation_names", ())]
    templates = getattr(lexicon, "templates", None)
    if templates:
        for spec in templates.values():
            texts += [*spec.prompts, spec.statement, spec.answer]
    elif lexicon is not None:                 # WordNet: the E5.4 templates and the held-out wordings
        from . import e5_zeroshot as zs
        from .e9_ontology_edit import STATEMENT_TEMPLATES
        texts += [t for group in (*zs.TEMPLATES.values(), *STATEMENT_TEMPLATES.values()) for t in group]
    if track == "t5":
        from ..benchmarks.glossary import FACT_TEMPLATES
        texts += [t for group in FACT_TEMPLATES.values() for t in group]
    return [re.sub(r"\{[^}]*\}", " ", t) for t in texts]


def _forms(surface: str) -> list[str]:
    out = []
    for form in dict.fromkeys((surface, surface.lower(), surface[:1].upper() + surface[1:])):
        out += [" " + form, form]
    return out


def tokenize_surfaces(surfaces: Sequence[str], tokenizer: Any, *, chunk: int = 4096) -> list[list[list[int]]]:
    """Per surface its token sequences (`e9_rescore.surface_sequences`' forms), batched."""
    forms = [(i, f) for i, s in enumerate(surfaces) for f in _forms(s)]
    out: list[set[tuple[int, ...]]] = [set() for _ in surfaces]
    for start in range(0, len(forms), chunk):
        part = forms[start:start + chunk]
        encoded = tokenizer([f for _, f in part], add_special_tokens=False)["input_ids"]
        for (i, _), ids in zip(part, encoded):
            if ids:
                out[i].add(tuple(int(t) for t in ids))
    return [[list(s) for s in sorted(seqs)] for seqs in out]


def entry_alias_displays(table: AliasTable, ontology: dict[str, Any]) -> list[list[str]]:
    """Per entry its aliases, in the concept name's casing where the alias is the name (the linker matches lower case)."""
    names = [str(n) for n in (ontology.get("concept_names") or [])]
    display = {normalize_alias(n): n for n in names}
    out: list[list[str]] = [[] for _ in table.entry_concepts]
    for alias, entry in sorted(table.alias_to_entry.items()):
        out[int(entry)].append(display.get(alias, alias))
    return out


def build_exclusion_table(ontology: dict[str, Any], table: AliasTable, tokenizer: Any, output: Path, *, lexicon: Any = None,
                          track: str | None = None, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """The strict stratum's verbalizations (module docstring): one vocabulary of surfaces (full aliases and their content
    words) with token sequences, and per atomic, per entry and for relation wording the surface ids; `atomic_entries`
    maps an atomic that names a concept to that concept's entries (the second hop)."""
    from .e9_rescore import atomic_surfaces
    vocabulary: dict[str, int] = {}

    def ids(texts: Iterable[str]) -> list[int]:
        found = set()
        for text in texts:
            for piece in [text, *content_words(text)]:
                piece = " ".join(str(piece).split())
                if len(piece) >= 2 and any(ch.isalnum() for ch in piece):
                    found.add(vocabulary.setdefault(piece, len(vocabulary)))
        return sorted(found)

    atomic = [ids(s) for s in atomic_surfaces(ontology, table, lexicon)]
    entries = [ids(a) for a in entry_alias_displays(table, ontology)]
    relations = sorted({vocabulary.setdefault(w, len(vocabulary)) for text in relation_texts(track, ontology, lexicon)
                        for w in content_words(text)})
    names = [str(n) for n in (ontology.get("concept_names") or [])]
    concept_of = {name: c for c, name in enumerate(names)}
    entries_of: dict[int, list[int]] = defaultdict(list)
    for e, concepts in enumerate(table.entry_concepts):
        for c in concepts:
            entries_of[int(c)].append(e)
    atomic_entries = []
    for atom in ontology["atomic_names"]:
        value = atom.split(":", 1)[1] if ":" in atom else atom
        concept = concept_of.get(value)
        atomic_entries.append(sorted(entries_of.get(concept, [])) if concept is not None else [])
    surfaces = sorted(vocabulary, key=vocabulary.get)
    sequences = tokenize_surfaces(surfaces, tokenizer)
    record = {"frames_sha256": frames_digest(ontology), "atomic_count": int(ontology["atomic_count"]),
              "entry_count": len(table.entry_concepts), "surfaces": surfaces, "sequences": sequences, "atomic_surfaces": atomic,
              "entry_surfaces": entries, "relation_surfaces": relations, "atomic_entries": atomic_entries,
              "meta": {**(meta or {}), "rule": "hop 0 (the term's aliases), hop 1 (its frame's fillers), hop 2 (fillers of the "
                       "frames of concepts that fill its frame): aliases and lexicon text and their content words (≥ 3 "
                       "letters, not a stopword); relation wording: content words of relation names and templates; as "
                       "written / lower / capitalized, with and without a leading space",
                       "surfaces": len(surfaces), "sequences": sum(map(len, sequences)),
                       "relation_words": [surfaces[i] for i in relations]}}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(output).with_suffix(".tmp")
    torch.save(record, temporary)
    temporary.replace(output)
    return record["meta"]


def exclusion_table_path(track: str, family: str = "smollm2") -> Path:
    from .e9_tracks import track_spec
    spec = track_spec(track, family)
    root = spec.alias_table_dir.parent / "exclusion-tables" if spec.licensed and spec.alias_table_dir else EXCLUSION_ROOT
    return root / f"{track}-{family}.pt"


def ensure_exclusion_table(track: str, family: str = "smollm2", output: Path | None = None, *, overwrite: bool = False) -> Path:
    """Build a track's exclusion table once (CPU: the alias table, the lexicon and the family's tokenizer)."""
    from transformers import AutoTokenizer

    from ..evaluation.channel_probes import load_alias_table, resolve_alias_table
    from .e9_tracks import FAMILY_TOKENIZERS, ensure_alias_table, lexicon_for, track_spec
    path = Path(output) if output else exclusion_table_path(track, family)
    if path.exists() and not overwrite:
        return path
    spec = track_spec(track, family)
    ontology = torch.load(spec.ontology, weights_only=False)
    alias_path = ensure_alias_table(spec)
    table = load_alias_table(alias_path) if alias_path else resolve_alias_table(ontology, spec.ontology)[0]
    tokenizer = AutoTokenizer.from_pretrained(FAMILY_TOKENIZERS[family], local_files_only=True)
    build_exclusion_table(ontology, table, tokenizer, path, lexicon=lexicon_for(spec, ontology), track=track,
                          meta={"track": track, "family": family, "tokenizer": FAMILY_TOKENIZERS[family], "ontology": str(spec.ontology)})
    return path


class StrictIndex:
    """Copy candidates of an entry: the token sequences of hops 0–2 and relation wording (see the module docstring)."""

    def __init__(self, data: dict[str, Any], ontology: dict[str, Any], *, hops: int = 2) -> None:
        if data.get("frames_sha256") != frames_digest(ontology):
            raise ValueError("the exclusion table was built for other ontology frames")
        self.offsets = np.asarray(ontology["offsets"], dtype=np.int64)
        self.fillers = np.asarray(ontology["fillers"], dtype=np.int64)
        self.sequences = [tuple(tuple(int(t) for t in s) for s in seqs) for seqs in data["sequences"]]
        self.atomic_surfaces = data["atomic_surfaces"]
        self.entry_surfaces = data["entry_surfaces"]
        self.relation_surfaces = list(data["relation_surfaces"])
        self.atomic_entries = data["atomic_entries"]
        self.hops = int(hops)
        self._index = functools.lru_cache(maxsize=65536)(self._build)

    def frame_atoms(self, entry: int) -> list[int]:
        if entry + 1 >= self.offsets.size:
            return []
        return self.fillers[self.offsets[entry]:self.offsets[entry + 1]].tolist()

    def closure(self, entry: int) -> set[int]:
        """Atomics within `hops` of the entry (hop 1 = its frame's fillers)."""
        atoms: set[int] = set()
        frontier = [entry]
        for _ in range(self.hops):
            found = {a for e in frontier for a in self.frame_atoms(e)} - atoms
            atoms |= found
            frontier = sorted({x for a in found for x in self.atomic_entries[a]})
        return atoms

    def surface_ids(self, entry: int) -> set[int]:
        out = set(self.relation_surfaces)
        if 0 <= entry < len(self.entry_surfaces):
            out.update(self.entry_surfaces[entry])
        for a in self.closure(entry):
            out.update(self.atomic_surfaces[a])
        return out

    def _build(self, entry: int) -> dict[int, tuple[tuple[int, ...], ...]]:
        by_first: dict[int, set[tuple[int, ...]]] = defaultdict(set)
        for i in self.surface_ids(entry):
            for seq in self.sequences[i]:
                if seq:
                    by_first[seq[0]].add(seq)
        return {k: tuple(sorted(v)) for k, v in by_first.items()}

    def target_hits(self, row: Sequence[int], entry: int, first: int, last: int) -> list[int]:
        """Targets `j ∈ [first, last]` covered by an occurrence of a copy candidate that starts inside `[first, last]`
        (the filler-table rule of `row_sources.FillerIndex.target_hits`)."""
        index = self._index(int(entry))
        hits: set[int] = set()
        for p in range(max(first, 0), min(last, len(row) - 1) + 1):
            for seq in index.get(row[p], ()):
                if tuple(row[p:p + len(seq)]) == seq:
                    hits.update(range(p, min(p + len(seq) - 1, last) + 1))
        return sorted(hits)


def load_strict_index(path: Path | str, ontology: dict[str, Any]) -> StrictIndex:
    return StrictIndex(torch.load(Path(path).expanduser(), weights_only=False, map_location="cpu"), ontology)


# ---------------------------------------------------------------- masks


def frequency_masks(ids: torch.Tensor, spans: dict[str, torch.Tensor], frequency: np.ndarray | None, heldout: set[int], *,
                    fillers: FillerIndex | None, strict: StrictIndex | None, bins: Sequence[tuple[str, int, int | None]]
                    ) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], dict[str, bool]]:
    """The trainer's strata (`lm.stratum_masks`, with the filler split when `fillers`), plus `X_strict` for every
    after-span stratum, the frequency bins (with `_filler` and `_strict`), and per (row, entry) the term masks.

    Returns (masks, terms, checks); `terms` rows are {"row", "entry", "own", "filler", "excluded"} boolean masks over the
    row's targets; `checks` records that the per-span strata replay the trainer's and that the bins nest the coarse
    strata."""
    base = lm.stratum_masks(ids, spans, frequency, heldout, fillers)
    batch, length = ids.shape
    shape = (batch, length - 1)
    zeros = lambda: torch.zeros(shape, dtype=torch.bool)
    replay = {name: zeros() for name in lm.AFTER_STRATA}
    excluded = {name: zeros() for name in lm.AFTER_STRATA}
    in_bin = {b: zeros() for b, _, _ in bins}
    bin_filler = {b: zeros() for b, _, _ in bins}
    bin_excluded = {b: zeros() for b, _, _ in bins}
    terms: dict[tuple[int, int], dict[str, Any]] = {}
    rows = ids.tolist()
    for b, s, e, entry, n in zip(spans["batch"].tolist(), spans["start"].tolist(), spans["end"].tolist(),
                                 spans["entry"].tolist(), spans["length"].tolist()):
        lo, hi = e, min(length - 1, e + lm.AFTER_WINDOW)
        if lo >= hi:
            continue
        names = span_strata(entry, n, frequency, heldout)
        filler_index = (torch.tensor(fillers.target_hits(rows[b], entry, lo + 1, hi), dtype=torch.long) - 1
                        if fillers is not None else torch.zeros(0, dtype=torch.long))
        strict_index = (torch.tensor(strict.target_hits(rows[b], entry, lo + 1, hi), dtype=torch.long) - 1
                        if strict is not None else torch.zeros(0, dtype=torch.long))
        for name in names:
            replay[name][b, lo:hi] = True
            excluded[name][b, strict_index] = True
        if entry not in heldout and frequency is not None:
            k = bin_index(int(frequency[entry]), bins)
            if k is not None:
                key = bins[k][0]
                in_bin[key][b, lo:hi] = True
                bin_filler[key][b, filler_index] = True
                bin_excluded[key][b, strict_index] = True
        term = terms.setdefault((b, int(entry)), {"row": b, "entry": int(entry), "own": torch.zeros(length - 1, dtype=torch.bool),
                                                  "filler": torch.zeros(length - 1, dtype=torch.bool),
                                                  "excluded": torch.zeros(length - 1, dtype=torch.bool)})
        term["own"][lo:hi] = True
        term["filler"][filler_index] = True
        term["excluded"][strict_index] = True
    checks = {"replay": all(torch.equal(replay[name], base[name]) for name in lm.AFTER_STRATA)}
    if not checks["replay"]:
        raise RuntimeError("the per-span strata do not replay training.lm.stratum_masks")
    masks = dict(base)
    if strict is not None:
        for name in lm.AFTER_STRATA:
            masks[f"{name}_strict"] = base[name] & ~excluded[name]
    for key, _, _ in bins:
        masks[key] = in_bin[key]
        if fillers is not None:
            masks[f"{key}_filler"] = in_bin[key] & bin_filler[key]
        if strict is not None:
            masks[f"{key}_strict"] = in_bin[key] & ~bin_excluded[key]
    nested = True
    for coarse, members in nesting(bins).items():
        if members:
            union = torch.zeros(shape, dtype=torch.bool)
            for key in members:
                union |= in_bin[key]
            nested &= torch.equal(union, base[coarse])
    checks["nested"] = bool(nested)
    for term in terms.values():
        term["filler"] &= term["own"]
        term["excluded"] &= term["own"]
    return masks, [terms[k] for k in sorted(terms)], checks


# ---------------------------------------------------------------- scoring a run


@torch.no_grad()
def evaluate_frequency(model: Any, corpus: TokenCorpus, starts: Sequence[int], config: dict[str, Any], frequency: np.ndarray | None,
                       heldout: set[int], device: torch.device, *, fillers: FillerIndex | None, strict: StrictIndex | None,
                       bins: Sequence[tuple[str, int, int | None]]) -> dict[str, Any]:
    """Per-window sums and counts of every stratum (strata × windows) and the term rows, with the per-token losses of
    `training.lm.evaluate` (same windows, batches and autocast)."""
    model.eval()
    length, batch = config["model"]["seq_len"], config["eval"]["batch"]
    sink: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    term_rows: list[tuple[int, int, int, int, int, float, float, float]] = []
    nested = True
    for i in range(0, len(starts), batch):
        windows = [corpus.window(s, length, min_subtokens=config["data"]["min_subtokens"]) for s in starts[i:i + batch]]
        ids, spans = collate_windows(windows)
        ids_d = ids.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            per_token = model(ids_d, spans={k: v.to(device) for k, v in spans.items()} if model.channel else None,
                              labels=ids_d, reduction="none")["loss"].float().cpu()
        masks, terms, checks = frequency_masks(ids, spans, frequency, heldout, fillers=fillers, strict=strict, bins=bins)
        nested &= checks["nested"]
        for name, mask in masks.items():
            sums, counts = sink.setdefault(name, ([], []))
            sums.append(per_token.double().masked_fill(~mask, 0.0).sum(1).numpy())
            counts.append(mask.sum(1).numpy().astype(np.int32))
        values = per_token.double()
        for term in terms:
            row = values[term["row"]]
            own, filler, strict_mask = term["own"], term["filler"], term["own"] & ~term["excluded"]
            term_rows.append((i + term["row"], term["entry"], int(own.sum()), int(filler.sum()), int(strict_mask.sum()),
                              float(row[own].sum()), float(row[filler].sum()), float(row[strict_mask].sum())))
    model.train()
    strata = list(sink)
    return {"strata": strata, "sums": np.stack([np.concatenate(sink[s][0]) for s in strata]),
            "counts": np.stack([np.concatenate(sink[s][1]) for s in strata]).astype(np.int32),
            "terms": np.asarray(term_rows, dtype=np.float64).reshape(-1, 8), "nested": bool(nested)}


def _atomic_write(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_scores(folder: Path) -> dict[str, Any] | None:
    """`{"strata", "starts", "count", "sums": {variant: strata × windows}, "terms": {...}, "record"}` or None."""
    folder = Path(folder)
    if not (folder / "windows.npz").exists():
        return None
    with np.load(folder / "windows.npz") as data:
        out: dict[str, Any] = {"strata": data["strata"].tolist(), "starts": data["starts"], "count": data["count"],
                               "sums": {k[4:]: data[k] for k in data.files if k.startswith("sum_")}}
    if (folder / "terms.npz").exists():
        with np.load(folder / "terms.npz") as data:
            out["terms"] = {k: data[k] for k in data.files}
    record_path = folder / "freqbias.json"
    out["record"] = json.loads(record_path.read_text()) if record_path.exists() else {}
    return out


def score_run(run_dir: Path, output: Path | None = None, *, variants: Sequence[str] = ("ref",), scheme: str = PRIMARY_SCHEME,
              fillers: Path | None | str = "auto", exclusions: Path | None | str = "auto", device: str = "cuda",
              eval_batch: int | None = None, resume: bool = False, overwrite: bool = False) -> dict[str, Any]:
    """Re-score `run_dir` on its evaluation windows with the trainer's strata, the filler split, the strict strata and the
    frequency bins (module docstring)."""
    from .e4_quant import reference_check
    from .e9_rescore import channel_off, ensure_filler_table
    run_dir = Path(run_dir)
    output = Path(output) if output else run_dir / SCORE_DIR
    for v in variants:
        if v not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
    final = torch.load(run_dir / "final.pt", weights_only=False, map_location="cpu")
    config = lm.resolve_config(final["config"])
    del final
    if eval_batch:
        config["eval"]["batch"] = int(eval_batch)
    target = torch.device(device if torch.cuda.is_available() else "cpu")
    if not config["data"].get("ontology"):
        raise ValueError("frequency rescoring needs the run's ontology")
    ontology = torch.load(config["data"]["ontology"], weights_only=False)
    track, family = config.get("e9_track"), config.get("e9_family", "smollm2")
    if fillers == "auto" or exclusions == "auto":
        if not track:
            raise ValueError("the run records no e9_track; pass --fillers and --exclusions (or --no-fillers / --no-strict)")
    if fillers == "auto":
        fillers = ensure_filler_table(track, family)
    if exclusions == "auto":
        exclusions = ensure_exclusion_table(track, family)
    filler_index = load_filler_index(Path(fillers), ontology) if fillers else None
    strict = load_strict_index(Path(exclusions), ontology) if exclusions else None
    heldout = {int(e) for e in ontology["heldout_entries"]}
    frequency = np.asarray(ontology["train_frequency"]) if ontology.get("train_frequency") is not None else None
    bins = scheme_bins(scheme)
    eval_corpus = TokenCorpus.open(Path(config["data"]["eval"]))
    starts = eval_windows(eval_corpus, count=config["eval"]["windows"], length=config["model"]["seq_len"])
    if overwrite and output.exists():
        for name in ("windows.npz", "terms.npz", "freqbias.json", "resolved_config.yaml", "manifest.json"):
            (output / name).unlink(missing_ok=True)
    if not resume and (output / "windows.npz").exists():
        raise FileExistsError(f"{output} holds a frequency rescoring; pass --resume or --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    git_at_start = git_state()
    existing = load_scores(output) if resume else None
    if existing and existing["record"].get("scheme") not in (None, scheme):
        raise ValueError(f"{output} was scored with scheme {existing['record']['scheme']}")
    sums = dict(existing["sums"]) if existing else {}
    terms = dict(existing.get("terms", {})) if existing else {}
    strata, counts = (existing["strata"], existing["count"]) if existing else (None, None)
    record = existing["record"] if existing and existing["record"] else {
        "run": str(run_dir), "id": f"{run_dir.parent.name}__{run_dir.name}", "variants": {}}
    has_channel = config["channel"]["mode"] != "none"
    record.update(channel=config["channel"]["mode"], scheme=scheme, bins=[{"name": b, "lo": lo, "hi": hi} for b, lo, hi in bins],
                  nesting=nesting(bins), fillers=str(fillers) if fillers else None, exclusions=str(exclusions) if exclusions else None,
                  frames_sha256=frames_digest(ontology), windows=len(starts), track=track, family=family)
    todo = [v for v in variants if v not in sums and (has_channel or v == "ref")]
    model = lm.load_final(run_dir / "final.pt", target) if todo else None
    for variant in todo:
        began = time.monotonic()
        with channel_off(model) if variant == "ref-off" else contextlib.nullcontext():
            found = evaluate_frequency(model, eval_corpus, starts, config, frequency, heldout, target, fillers=filler_index,
                                       strict=strict, bins=bins)
        if strata is None:
            strata, counts = found["strata"], found["counts"]
        elif found["strata"] != strata or not np.array_equal(found["counts"], counts):
            raise RuntimeError(f"{run_dir}: variant {variant} produced other strata or target counts")
        sums[variant] = found["sums"]
        rows = found["terms"]
        base_terms = {"window": rows[:, 0].astype(np.int32), "entry": rows[:, 1].astype(np.int64),
                      "count": rows[:, 2].astype(np.int32), "filler_count": rows[:, 3].astype(np.int32),
                      "strict_count": rows[:, 4].astype(np.int32)}
        if "window" in terms and not all(np.array_equal(terms[k], v) for k, v in base_terms.items()):
            raise RuntimeError(f"{run_dir}: variant {variant} produced other term rows")
        terms.update(base_terms)
        terms["frequency"] = (frequency[base_terms["entry"]] if frequency is not None else np.zeros(len(rows))).astype(np.int64)
        terms["heldout"] = np.asarray([int(e) in heldout for e in base_terms["entry"]], dtype=bool)
        terms.update({f"sum_{variant}": rows[:, 5], f"filler_sum_{variant}": rows[:, 6], f"strict_sum_{variant}": rows[:, 7]})
        total = found["sums"].sum(1)
        n = found["counts"].sum(1)
        record["variants"][variant] = {
            "strata": {s: {"loss": float(total[i] / n[i]) if n[i] else None, "tokens": int(n[i])} for i, s in enumerate(strata)},
            "channel_off": variant == "ref-off", "nested": found["nested"], "seconds": round(time.monotonic() - began, 2)}
        _atomic_write(output / "windows.npz", {"strata": np.asarray(strata), "starts": np.asarray(starts, dtype=np.int64),
                                                "count": counts, **{f"sum_{k}": v for k, v in sums.items()}})
        _atomic_write(output / "terms.npz", terms)
        (output / "freqbias.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
        print(json.dumps({"run": record["id"], "variant": variant, "after": record["variants"][variant]["strata"]["after"]["loss"],
                          "seconds": record["variants"][variant]["seconds"]}), flush=True)
    if "ref" in sums and strata is not None:
        record["ref_check"] = reference_check(run_dir, strata, list(starts), sums["ref"], counts)
    if strata is not None and "window" in terms:
        record["bin_support"] = bin_support(strata, counts, terms, bins)
    if not has_channel:
        record["off_variants"] = "a run without a channel answers ref-off with ref"
    (output / "freqbias.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    settings = {"run": str(run_dir), "variants": list(variants), "scheme": scheme, "fillers": str(fillers) if fillers else None,
                "exclusions": str(exclusions) if exclusions else None, "eval_batch": eval_batch, "device": device}
    write_run_metadata(output, settings, git_at_start=git_at_start, device=target, variants=sorted(sums))
    return record


def bin_support(strata: Sequence[str], counts: np.ndarray, terms: dict[str, np.ndarray],
                bins: Sequence[tuple[str, int, int | None]]) -> dict[str, dict[str, Any]]:
    """Per bin (and the zero-frequency strata): targets, distinct entries and the target-weighted mean log2 frequency of
    the entries' own after-span targets (model-independent)."""
    out = {}
    freq, held = terms["frequency"], terms["heldout"].astype(bool)
    for name, lo, hi in [*bins, ("after_unseen", 0, 0), ("after_heldout", None, None)]:
        if name not in strata:
            continue
        if name == "after_heldout":
            member = held
        else:
            member = ~held & (freq >= lo) & ((freq <= hi) if hi is not None else True)
        weight = terms["count"][member].astype(np.float64)
        x = np.log2(np.maximum(freq[member], 1)).astype(np.float64)
        out[name] = {"targets": int(counts[list(strata).index(name)].sum()), "entries": int(np.unique(terms["entry"][member]).size),
                     "mean_log2_frequency": float((weight * x).sum() / weight.sum()) if weight.sum() and name not in
                     {"after_unseen", "after_heldout"} else None}
    return out


# ---------------------------------------------------------------- analysis (used by e9_report --freqbias)


def _pool(scores: dict[int, dict[str, Any]], seeds: Sequence[int], variant: str, strata: Sequence[str]
          ) -> tuple[np.ndarray, np.ndarray] | None:
    """Window sums and counts (len(strata) × windows) summed over `seeds` (a single-run model stands for every seed)."""
    picks = [scores[min(scores)]] * len(seeds) if len(scores) == 1 and set(seeds) - set(scores) else [scores.get(s) for s in seeds]
    if not picks or any(p is None for p in picks):
        return None
    total_s = total_n = 0
    for found in picks:
        name = variant if variant in found["sums"] else ("ref" if variant == "ref-off" and found["record"].get("channel") == "none"
                                                       else None)
        if name is None or any(s not in found["strata"] for s in strata):
            return None
        index = [found["strata"].index(s) for s in strata]
        total_s = total_s + found["sums"][name][index]
        total_n = total_n + found["count"][index].astype(np.float64)
    return total_s, total_n


def _term_pool(scores: dict[int, dict[str, Any]], seeds: Sequence[int], variant: str, part: str = "") -> dict[str, np.ndarray] | None:
    """Per entry (one row per entry) the sums and counts of its own after-span targets, summed over windows and seeds."""
    picks = [scores[min(scores)]] * len(seeds) if len(scores) == 1 and set(seeds) - set(scores) else [scores.get(s) for s in seeds]
    if not picks or any(p is None or "terms" not in p for p in picks):
        return None
    prefix = f"{part}_" if part else ""
    key = f"{prefix}sum_{variant}"
    count_key = f"{part}_count" if part else "count"
    first = picks[0]["terms"]
    entries, inverse = np.unique(first["entry"], return_inverse=True)
    total_s = np.zeros(entries.size)
    total_n = np.zeros(entries.size)
    for found in picks:
        terms = found["terms"]
        name = key if key in terms else (f"{prefix}sum_ref" if variant == "ref-off" and found["record"].get("channel") == "none" else None)
        if name is None or not np.array_equal(terms["entry"], first["entry"]) or not np.array_equal(terms["window"], first["window"]):
            return None
        total_s += np.bincount(inverse, weights=terms[name], minlength=entries.size)
        total_n += np.bincount(inverse, weights=terms[count_key].astype(np.float64), minlength=entries.size)
    frequency = np.zeros(entries.size, dtype=np.int64)
    heldout = np.zeros(entries.size, dtype=bool)
    frequency[inverse] = first["frequency"]
    heldout[inverse] = first["heldout"].astype(bool)
    return {"entry": entries, "sum": total_s, "count": total_n, "frequency": frequency, "heldout": heldout}


def _ols_slope(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS slopes of each row of `y` (… × bins) on `x` (bins); NaN with fewer than two bins."""
    if x.size < 2:
        return np.full(y.shape[:-1], np.nan)
    xc = x - x.mean()
    return ((y - y.mean(-1, keepdims=True)) * xc).sum(-1) / (xc ** 2).sum()


def _member_mask(frequency: np.ndarray, heldout: np.ndarray, lo: int | None, hi: int | None) -> np.ndarray:
    if lo is None:
        return heldout
    return ~heldout & (frequency >= lo) & ((frequency <= hi) if hi is not None else True)


def _interval(draws: np.ndarray, point: float) -> dict[str, float]:
    draws = draws[np.isfinite(draws)]
    if not draws.size:
        return {"mean": point, "ci_low": None, "ci_high": None, "p_value": None}
    p = float(min(1.0, 2 * (min((draws <= 0).sum(), (draws >= 0).sum()) + 1) / (draws.size + 1)))
    return {"mean": point, "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)), "p_value": p}


def bias_measures(curves: dict[str, np.ndarray], x: np.ndarray, slope_bins: Sequence[int], rare: int, frequent: int) -> dict[str, np.ndarray]:
    """Slope (nats per doubling of training frequency) over `slope_bins`, absolute and relative rare–frequent gap, from
    loss curves (… × rows; rows = bins + the coarse strata)."""
    out = {}
    for model, y in curves.items():
        out[model] = {"slope": _ols_slope(x[list(slope_bins)], y[..., list(slope_bins)]),
                      "gap": y[..., rare] - y[..., frequent],
                      "gap_relative": (y[..., rare] - y[..., frequent]) / y[..., frequent]}
    return out


def _common_seeds(scores: dict[str, dict[int, Any]], a: str, b: str) -> list[int]:
    """Seeds of a paired comparison; a model with a single run (P0) stands for every seed of the other."""
    if len(scores[b]) == 1 and len(scores[a]) > 1:
        return sorted(scores[a])
    if len(scores[a]) == 1 and len(scores[b]) > 1:
        return sorted(scores[b])
    return sorted(set(scores[a]) & set(scores[b]))


def frequency_analysis(scores: dict[str, dict[int, dict[str, Any]]], *, candidates: Sequence[str], reference: str = "C0'",
                       variant: str = "ref", resamples: int = 10_000, seed: int = 0, min_targets: int = 1000,
                       min_entries: int = 20, extra_pairs: Sequence[tuple[str, str]] = ()) -> dict[str, Any]:
    """A1 (pre-registration §A1): per-bin losses per model, candidate − reference per bin (window and term-cluster
    bootstraps, Holm over bins), and per model the frequency-bias slope and the rare–frequent gap with their changes.

    `scores`: model → seed → `load_scores` output. Seeds are pooled per window (all seeds share the evaluation windows);
    a model with one run (P0) stands for every seed. Bins enter the slope when they hold ≥ `min_targets` targets and ≥
    `min_entries` distinct entries (one seed's windows; support is model-independent). The first candidate's gap change
    is the primary endpoint."""
    scores = {m: by_seed for m, by_seed in scores.items() if by_seed}
    if reference not in scores:
        return {"available": False, "detail": f"no {reference} frequency rescoring"}
    first = next(iter(scores[reference].values()))
    record = first["record"]
    bins = [(b["name"], b["lo"], b["hi"]) for b in record.get("bins", [])]
    support = record.get("bin_support") or {}
    rows = [r for r in [b for b, _, _ in bins] + ["after_unseen", "after_heldout", "after_rare_seen", "after_frequent"]
            if r in first["strata"]]
    if "after_rare_seen" not in rows or "after_frequent" not in rows:
        return {"available": False, "detail": "the rescoring lacks the coarse strata"}
    rare, frequent = rows.index("after_rare_seen"), rows.index("after_frequent")
    slope_bins = [rows.index(b) for b, _, _ in bins if b in rows and support.get(b, {}).get("targets", 0) >= min_targets
                  and support.get(b, {}).get("entries", 0) >= min_entries]
    x = np.asarray([support.get(r, {}).get("mean_log2_frequency") or np.nan for r in rows], dtype=np.float64)
    windows = first["count"].shape[1]
    weights = np.random.default_rng(seed).multinomial(windows, np.full(windows, 1.0 / windows), size=resamples).astype(np.float64)
    out: dict[str, Any] = {"available": True, "reference": reference, "variant": variant, "scheme": record.get("scheme"),
                           "rows": [{"name": r, **support.get(r, {})} for r in rows], "slope_bins": [rows[i] for i in slope_bins],
                           "min_targets": min_targets, "min_entries": min_entries, "losses": {}, "bias": {}, "comparisons": {},
                           "changes": {}, "seeds": {}}
    for model, by_seed in scores.items():
        pooled = _pool(by_seed, sorted(by_seed), variant, rows)
        if pooled is None:
            continue
        s, n = pooled
        with np.errstate(invalid="ignore", divide="ignore"):
            point = s.sum(1) / n.sum(1)
            curve = (weights @ s.T) / (weights @ n.T)
        out["losses"][model] = {r: float(point[i]) for i, r in enumerate(rows)}
        out["seeds"][model] = sorted(by_seed)
        p, d = bias_measures({"m": point}, x, slope_bins, rare, frequent)["m"], bias_measures({"m": curve}, x, slope_bins, rare, frequent)["m"]
        with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            out["bias"][model] = {k: {"mean": float(p[k]), "ci_low": float(np.nanquantile(d[k], 0.025)),
                                      "ci_high": float(np.nanquantile(d[k], 0.975))} for k in p}
    pairs = [(c, reference) for c in candidates if c in scores and c != reference] + [
        p for p in extra_pairs if p[0] in scores and p[1] in scores]
    for cand, ref in pairs:
        common = _common_seeds(scores, cand, ref)
        a, b = _pool(scores[cand], common, variant, rows), _pool(scores[ref], common, variant, rows)
        if a is None or b is None or not np.array_equal(a[1], b[1]):
            continue
        label = f"{cand} − {ref}"
        block: dict[str, Any] = {"seeds": common, "bins": {}}
        for i, r in enumerate(rows):
            if a[1][i].sum() == 0:
                continue
            boot = paired_ratio_bootstrap(a[0][i] - b[0][i], a[1][i], b[0][i], resamples=resamples, seed=seed)
            block["bins"][r] = {k: boot.get(k) for k in ("mean", "ci_low", "ci_high", "p_value", "relative", "relative_ci_low",
                                                       "relative_ci_high")} | {"targets": int(a[1][i].sum())}
        tested = [block["bins"][r] for r in rows if r in block["bins"] and r not in {"after_rare_seen", "after_frequent"}]
        for entry, adjusted in zip(tested, holm_adjust([e["p_value"] for e in tested]) if tested else []):
            entry.update(holm_p=adjusted, significant=adjusted < 0.05)
        ta, tb = _term_pool(scores[cand], common, variant), _term_pool(scores[ref], common, variant)
        terms_ok = ta is not None and tb is not None and np.array_equal(ta["entry"], tb["entry"])
        if terms_ok:                       # term-cluster bootstrap per bin (entries resampled; the entry-sum estimand)
            for r in rows:
                lo, hi = _row_limits(r, bins)
                member = _member_mask(ta["frequency"], ta["heldout"], lo, hi)
                if member.sum() < 2 or ta["count"][member].sum() == 0:
                    continue
                boot = paired_ratio_bootstrap(ta["sum"][member] - tb["sum"][member], ta["count"][member], tb["sum"][member],
                                              resamples=resamples, seed=seed)
                block["bins"].setdefault(r, {})["term"] = {k: boot.get(k) for k in ("mean", "ci_low", "ci_high", "p_value", "relative",
                                                                                    "relative_ci_low", "relative_ci_high")} | {
                    "entries": int(member.sum())}
        out["comparisons"][label] = block
        # the slope needs ≥ 2 supported bins (else NaN); the gap needs none
        with np.errstate(invalid="ignore", divide="ignore"):
            pa, pb = a[0].sum(1) / a[1].sum(1), b[0].sum(1) / b[1].sum(1)
            ca, cb = (weights @ a[0].T) / (weights @ a[1].T), (weights @ b[0].T) / (weights @ b[1].T)
        point = bias_measures({"a": pa, "b": pb}, x, slope_bins, rare, frequent)
        draws = bias_measures({"a": ca, "b": cb}, x, slope_bins, rare, frequent)
        change = {k: _interval(draws["a"][k] - draws["b"][k], float(point["a"][k] - point["b"][k]))
                  for k in ("slope", "gap", "gap_relative")}
        with np.errstate(invalid="ignore", divide="ignore"):
            change["gap_cut"] = _interval(1 - draws["a"]["gap"] / draws["b"]["gap"], float(1 - point["a"]["gap"] / point["b"]["gap"]))
        change["gap_cut"]["p_value"] = None              # a share, not a test against 0
        if terms_ok:
            change["term"] = _term_changes(ta, tb, bins, slope_bins, rows, x, resamples=resamples, seed=seed)
        change["seeds"] = common
        out["changes"][label] = change
    primary = out["changes"].get(f"{candidates[0]} − {reference}") if candidates else None
    if primary:
        primary["gap"]["primary"] = True
    return out


def _row_limits(row: str, bins: Sequence[tuple[str, int, int | None]]) -> tuple[Any, Any]:
    for name, lo, hi in bins:
        if name == row:
            return lo, hi
    return {"after_unseen": (0, 0), "after_heldout": (None, None), "after_rare_seen": (1, 9),
            "after_frequent": (100, None)}.get(row, ("skip", None))


def _term_changes(ta: dict[str, np.ndarray], tb: dict[str, np.ndarray], bins, slope_bins, rows, x, *, resamples: int, seed: int
                  ) -> dict[str, Any]:
    """Slope and gap changes under the term-cluster bootstrap (entries resampled; the entry-sum estimand)."""
    members = np.stack([_member_mask(ta["frequency"], ta["heldout"], *_row_limits(r, bins)) if _row_limits(r, bins)[0] != "skip"
                        else np.zeros(ta["entry"].size, dtype=bool) for r in rows]).astype(np.float64)      # rows × entries
    weights = np.random.default_rng(seed + 1).multinomial(ta["entry"].size, np.full(ta["entry"].size, 1.0 / ta["entry"].size),
                                                         size=resamples).astype(np.float64)
    def curve(t: dict[str, np.ndarray], w: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return ((w * t["sum"]) @ members.T) / ((w * t["count"]) @ members.T)
    one = np.ones((1, ta["entry"].size))
    rare, frequent = rows.index("after_rare_seen"), rows.index("after_frequent")
    point = bias_measures({"a": curve(ta, one)[0], "b": curve(tb, one)[0]}, x, slope_bins, rare, frequent)
    draws = bias_measures({"a": curve(ta, weights), "b": curve(tb, weights)}, x, slope_bins, rare, frequent)
    return {k: _interval(draws["a"][k] - draws["b"][k], float(point["a"][k] - point["b"][k])) for k in ("slope", "gap", "gap_relative")}


STRICT_STRATA = ("after_heldout", "after_rare_seen", "after_unseen", "after_len3plus", "after")


def strict_analysis(scores: dict[str, dict[int, dict[str, Any]]], *, candidate: str, references: Sequence[str], variant: str = "ref",
                    resamples: int = 10_000, seed: int = 0) -> dict[str, Any]:
    """B2 (pre-registration §B2): candidate − reference on `X`, `X_filler`, `X_nonfiller` and `X_strict` (window bootstrap,
    pooled over common seeds; Holm over the strict tests), with the strict share of each stratum's targets."""
    if candidate not in scores:
        return {"available": False}
    out: dict[str, Any] = {"available": True, "rows": {}}
    tested = []
    for reference in [r for r in references if r in scores and r != candidate]:
        common = _common_seeds(scores, candidate, reference)
        for base in STRICT_STRATA:
            parts = {}
            for part in ("", "_filler", "_nonfiller", "_strict"):
                a, b = _pool(scores[candidate], common, variant, [base + part]), _pool(scores[reference], common, variant, [base + part])
                if a is None or b is None or not np.array_equal(a[1], b[1]) or a[1].sum() == 0:
                    continue
                boot = paired_ratio_bootstrap(a[0][0] - b[0][0], a[1][0], b[0][0], resamples=resamples, seed=seed)
                parts[part.lstrip("_") or "total"] = {k: boot.get(k) for k in ("mean", "ci_low", "ci_high", "p_value", "relative",
                                                                               "relative_ci_low", "relative_ci_high")} | {
                    "targets": int(a[1][0].sum()), "seeds": common}
            if "strict" in parts:
                tested.append(parts["strict"])
                parts["strict_share"] = parts["strict"]["targets"] / parts["total"]["targets"] if parts.get("total") else None
            if parts:
                out["rows"].setdefault(base, {})[reference] = parts
    for entry, adjusted in zip(tested, holm_adjust([t["p_value"] for t in tested]) if tested else []):
        entry.update(holm_p=adjusted, significant=adjusted < 0.05)
    return out


# ---------------------------------------------------------------- A2: frequency decodability of concept rows


def probe_features(matrix: np.ndarray) -> np.ndarray:
    """A representation's coordinates plus its L2 norm and log norm (a linear probe cannot read a norm, and a rarely
    updated free row differs from a frequently updated one mostly in norm); one-column references are used as they are."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[1] <= 1:
        return matrix
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.concatenate([matrix, norm, np.log(norm + 1e-8)], axis=1)


def ridge_cv(x: np.ndarray, y: np.ndarray, *, folds: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold predictions of a ridge probe (standardized inputs, strength by GCV in each training fold)."""
    from .e5_zeroshot import ridge_map
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    order = np.random.default_rng(seed).permutation(len(y))
    fold = np.empty(len(y), dtype=np.int64)
    fold[order] = np.arange(len(y)) % folds
    out = np.empty(len(y))
    for k in range(folds):
        train, test = fold != k, fold == k
        if x.shape[1] == 0 or np.allclose(x[train].std(0), 0):
            out[test] = y[train].mean()
            continue
        predict, _ = ridge_map(x[train], y[train][:, None])
        out[test] = predict(x[test])[:, 0]
    return out


def r2(y: np.ndarray, prediction: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray | float:
    """R² of predictions; with `weights` (resamples × n) one value per resample."""
    if weights is None:
        return float(1 - ((y - prediction) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    total = weights.sum(1, keepdims=True)
    mean = (weights @ y)[:, None] / total
    sse = weights @ ((y - prediction) ** 2)
    sst = (weights * (y[None, :] - mean) ** 2).sum(1)
    return 1 - sse / sst


@torch.no_grad()
def concept_representations(run_dir: Path, *, device: str = "cpu", alias_table: Path | None = None,
                            tokenizer_name: str | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Entries × dimension arrays of a run's concept representations (module docstring) and their description."""
    from ..evaluation.channel_probes import load_alias_table, resolve_alias_table
    from .e5_common import entry_rows
    from .e9_tracks import ensure_alias_table, track_spec
    from transformers import AutoTokenizer
    final = torch.load(Path(run_dir) / "final.pt", weights_only=False, map_location="cpu")
    config = lm.resolve_config(final["config"])
    del final
    ontology = torch.load(config["data"]["ontology"], weights_only=False)
    model = lm.load_final(Path(run_dir) / "final.pt", device)
    entries = torch.arange(int(ontology["entry_count"]))
    reps: dict[str, np.ndarray] = {}
    info: dict[str, Any] = {"channel": config["channel"]["mode"]}
    channel = model.channel
    if channel is not None:
        if channel.mode == "compose":
            parts = [channel.composer.compose(part.to(next(channel.parameters()).device), None).float().cpu() for part in entries.split(4096)]
            reps["table"] = torch.cat(parts).numpy()
            info["table"] = "composed vector (before the projector)"
        elif channel.mode == "free":
            reps["table"] = channel.table.weight.detach().float().cpu().numpy()
            info["table"] = "free row (before the projector)"
        elif channel.mode == "source":
            reps["table"] = channel.source_rows.detach().float().cpu().numpy()
            info["table"] = "frozen source row (before the projector)"
        if channel.mode in {"compose", "free", "source"}:
            reps["row"] = entry_rows(channel, entries).numpy()
            info["row"] = "injected row (after the projector, host scale included)"
    track, family = config.get("e9_track"), config.get("e9_family", "smollm2")
    if alias_table is None and track:
        alias_table = ensure_alias_table(track_spec(track, family))
    table = load_alias_table(alias_table) if alias_table else resolve_alias_table(ontology, Path(config["data"]["ontology"]))[0]
    from .e9_tracks import FAMILY_TOKENIZERS
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or FAMILY_TOKENIZERS.get(family, config["model"].get("pretrained")),
                                              local_files_only=True)
    embedding = model.model.get_input_embeddings().weight.detach().float().cpu()
    displays = entry_alias_displays(table, ontology)
    flat = [(e, " " + a) for e, aliases in enumerate(displays) for a in aliases]
    means = torch.zeros(len(displays), embedding.shape[1])
    counts = torch.zeros(len(displays))
    for start in range(0, len(flat), 4096):
        part = flat[start:start + 4096]
        for (e, _), ids in zip(part, tokenizer([t for _, t in part], add_special_tokens=False)["input_ids"]):
            if ids:
                means[e] += embedding[torch.tensor(ids)].mean(0)
                counts[e] += 1
    reps["host_subtoken_mean"] = (means / counts.clamp(min=1)[:, None]).numpy()
    info["host_subtoken_mean"] = "the host's input-embedding rows averaged over each alias's subtokens, then over aliases"
    offsets = np.asarray(ontology["offsets"])
    reps["degree"] = np.diff(offsets).astype(np.float64)[:, None]
    info["degree"] = "frame degree (model-free reference)"
    from .e5_zeroshot import graph_features
    heldout = {int(e) for e in ontology["heldout_entries"]}
    fit = [e for e in range(len(offsets) - 1) if e not in heldout]
    reps["frame_graph"] = np.asarray(graph_features(ontology, fit, rank=64)(list(range(len(offsets) - 1))), dtype=np.float64)
    info["frame_graph"] = "64-rank spectral embedding of the typed frame graph (model-free reference)"
    del model
    return reps, {**info, "track": track, "family": family, "config": config}


def probe_entries(ontology: dict[str, Any], *, subset: str = "seen") -> tuple[np.ndarray, np.ndarray]:
    """(entries, target): `seen` = not held out, frequency ≥ 1, target log2 f; `all` = not held out, log2(1 + f)."""
    frequency = np.asarray(ontology["train_frequency"], dtype=np.float64)
    heldout = np.zeros(frequency.size, dtype=bool)
    heldout[[int(e) for e in ontology["heldout_entries"]]] = True
    if subset == "seen":
        keep = np.flatnonzero(~heldout & (frequency >= 1))
        return keep, np.log2(frequency[keep])
    keep = np.flatnonzero(~heldout)
    return keep, np.log2(1 + frequency[keep])


def tsne(x: np.ndarray, *, perplexity: float = 30.0, iterations: int = 750, seed: int = 0, pca: int = 50) -> np.ndarray:
    """Exact t-SNE (van der Maaten & Hinton 2008: perplexity calibration by bisection, early exaggeration 12 for 250
    iterations, momentum 0.5 → 0.8, adaptive gains, learning rate max(n / 48, 50) as scikit-learn's "auto"), on the first
    `pca` principal components; torch on the CPU."""
    gen = torch.Generator().manual_seed(seed)
    data = torch.as_tensor(np.asarray(x, dtype=np.float64))
    data = data - data.mean(0)
    if data.shape[1] > pca:
        _, _, v = torch.linalg.svd(data, full_matrices=False)
        data = data @ v[:pca].T
    n = data.shape[0]
    d2 = torch.cdist(data, data).pow(2)
    target = math.log(perplexity)
    p = torch.zeros(n, n, dtype=torch.float64)
    beta = torch.ones(n, dtype=torch.float64)
    lo, hi = torch.full((n,), 0.0, dtype=torch.float64), torch.full((n,), float("inf"), dtype=torch.float64)
    eye = torch.eye(n, dtype=torch.bool)
    scale = d2[~eye].median().clamp(min=1e-12)
    d2n = d2 / scale
    for _ in range(64):
        logits = -d2n * beta[:, None]
        logits = logits.masked_fill(eye, -float("inf"))
        p = torch.softmax(logits, 1)
        entropy = -(p * torch.log(p.clamp(min=1e-300))).sum(1)
        too_flat = entropy > target
        lo = torch.where(too_flat, beta, lo)
        hi = torch.where(too_flat, hi, beta)
        beta = torch.where(torch.isinf(hi), beta * 2, (lo + hi) / 2)
    p = (p + p.T) / (2 * n)
    p = p.clamp(min=1e-12)
    y = torch.randn(n, 2, generator=gen, dtype=torch.float64) * 1e-4
    rate = max(n / 48.0, 50.0)
    update = torch.zeros_like(y)
    gains = torch.ones_like(y)
    for it in range(iterations):
        exaggeration = 12.0 if it < 250 else 1.0
        num = 1.0 / (1.0 + torch.cdist(y, y).pow(2))
        num = num.masked_fill(eye, 0.0)
        q = (num / num.sum()).clamp(min=1e-12)
        pq = (exaggeration * p - q) * num
        grad = 4 * (pq.sum(1, keepdim=True) * y - pq @ y)
        momentum = 0.5 if it < 250 else 0.8
        gains = torch.where(torch.sign(grad) != torch.sign(update), gains + 0.2, gains * 0.8).clamp(min=0.01)
        update = momentum * update - rate * gains * grad
        y = y + update
        y = y - y.mean(0)
    return y.numpy()


def tsne_sample(entries: np.ndarray, target: np.ndarray, *, size: int = 1500, seed: int = 0) -> np.ndarray:
    """Indices into `entries` of up to `size` entries, equal numbers per integer log2-frequency level (seeded; the same
    sample for every run of a track)."""
    rng = np.random.default_rng(seed)
    levels = np.floor(target).astype(int)
    groups = [np.flatnonzero(levels == k) for k in np.unique(levels)]
    per = max(1, size // max(1, len(groups)))
    picked = np.concatenate([rng.choice(g, min(per, g.size), replace=False) for g in groups])
    if picked.size < size:
        rest = np.setdiff1d(np.arange(entries.size), picked)
        picked = np.concatenate([picked, rng.choice(rest, min(size - picked.size, rest.size), replace=False)])
    return np.sort(picked)


SEQUENTIAL = ("#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b")
INK, MUTED = "#0b0b0b", "#52514e"


def plot_tsne(panels: dict[str, tuple[np.ndarray, np.ndarray]], path: Path, *, title: str) -> None:
    """One panel per representation, points coloured by log2 training frequency (the dataviz sequential blue ramp,
    steps 250 → 700, light = rare)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("freq", SEQUENTIAL)
    keys = list(panels)
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys) + 0.8, 3.4), squeeze=False)
    vmax = max(float(v.max()) for _, v in panels.values())
    for ax, key in zip(axes[0], keys):
        coords, values = panels[key]
        order = np.argsort(values)
        scatter = ax.scatter(coords[order, 0], coords[order, 1], c=values[order], cmap=cmap, vmin=0, vmax=vmax, s=5, linewidths=0)
        ax.set_title(key, fontsize=9, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#e6e5e0")
    bar = fig.colorbar(scatter, ax=axes[0].tolist(), shrink=0.8, pad=0.02)
    bar.set_label("log2 training frequency", fontsize=8, color=MUTED)
    bar.ax.tick_params(labelsize=7, colors=MUTED)
    fig.suptitle(title, fontsize=10, color=INK)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def probe_run(run_dir: Path, output: Path | None = None, *, folds: int = 5, seed: int = 0, tsne_points: int = 1500,
              tsne_iterations: int = 750, with_tsne: bool = True, device: str = "cpu", overwrite: bool = False,
              alias_table: Path | None = None, tokenizer_name: str | None = None) -> dict[str, Any]:
    """A2 for one run: out-of-fold ridge predictions of log2 frequency per representation (`oof.npz`), point R², and the
    t-SNE coordinates and figure."""
    run_dir = Path(run_dir)
    output = Path(output) if output else run_dir / ROWS_DIR
    if (output / "oof.npz").exists() and not overwrite:
        raise FileExistsError(f"{output} holds a row probe; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    git_at_start = git_state()
    began = time.monotonic()
    reps, info = concept_representations(run_dir, device=device, alias_table=alias_table, tokenizer_name=tokenizer_name)
    config = info.pop("config")
    ontology = torch.load(config["data"]["ontology"], weights_only=False)
    arrays: dict[str, np.ndarray] = {}
    record: dict[str, Any] = {"run": str(run_dir), "id": f"{run_dir.parent.name}__{run_dir.name}", "representations": {},
                              "folds": folds, "seed": seed, "subsets": {}}
    for subset in ("seen", "all"):
        entries, y = probe_entries(ontology, subset=subset)
        arrays[f"{subset}_entries"] = entries
        arrays[f"{subset}_target"] = y
        record["subsets"][subset] = {"entries": int(entries.size)}
        for name, matrix in reps.items():
            prediction = ridge_cv(probe_features(matrix[entries]), y, folds=folds, seed=seed)
            arrays[f"{subset}_oof_{name}"] = prediction.astype(np.float32)
            record["representations"].setdefault(name, {"description": info.get(name), "dimension": int(matrix.shape[1])})
            record["representations"][name][f"r2_{subset}"] = r2(y, prediction)
    if with_tsne:
        entries, y = arrays["seen_entries"], arrays["seen_target"]
        pick = tsne_sample(entries, y, size=tsne_points, seed=seed)
        coords = {}
        for name in [n for n in ("table", "row", "host_subtoken_mean") if n in reps]:
            coords[name] = tsne(reps[name][entries[pick]], seed=seed, iterations=tsne_iterations)
        np.savez_compressed(output / "tsne.npz", entries=entries[pick], target=y[pick], **{f"coords_{k}": v for k, v in coords.items()})
        labels = {"table": info.get("table", "table"), "row": "injected row", "host_subtoken_mean": "host subtoken mean"}
        plot_tsne({labels[k]: (v, y[pick]) for k, v in coords.items()}, output / "tsne.png", title=record["id"])
        record["tsne"] = {"points": int(pick.size), "perplexity": 30, "iterations": tsne_iterations, "representations": sorted(coords)}
    np.savez_compressed(output / "oof.npz", **arrays)
    record["seconds"] = round(time.monotonic() - began, 1)
    (output / "probe.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    write_run_metadata(output, {"run": str(run_dir), "folds": folds, "seed": seed, "tsne_points": tsne_points,
                                "tsne_iterations": tsne_iterations, "with_tsne": with_tsne, "device": device},
                       git_at_start=git_at_start, device=device)
    return record


def load_probe(folder: Path) -> dict[str, Any] | None:
    folder = Path(folder)
    if not (folder / "oof.npz").exists():
        return None
    with np.load(folder / "oof.npz") as data:
        arrays = {k: data[k] for k in data.files}
    record = json.loads((folder / "probe.json").read_text()) if (folder / "probe.json").exists() else {}
    tsne_data = None
    if (folder / "tsne.npz").exists():
        with np.load(folder / "tsne.npz") as data:
            tsne_data = {k: data[k] for k in data.files}
    return {"arrays": arrays, "record": record, "tsne": tsne_data}


def probe_analysis(probes: dict[str, dict[int, dict[str, Any]]], *, subset: str = "seen", resamples: int = 2000, seed: int = 0,
                   primary: tuple[tuple[str, str], tuple[str, str]] = (("C2", "table"), ("C5", "table"))) -> dict[str, Any]:
    """A2 across models and seeds: R² per (model, representation) averaged over seeds with an entry-bootstrap interval
    (the same entry resamples for every run, so differences are paired), and the pre-registered contrast
    R²(C2 free row) − R²(C5 composed vector) (`primary`)."""
    runs = [(m, s, p) for m, by_seed in probes.items() for s, p in by_seed.items() if p is not None]
    if not runs:
        return {"available": False}
    entries = runs[0][2]["arrays"][f"{subset}_entries"]
    y = runs[0][2]["arrays"][f"{subset}_target"]
    weights = np.random.default_rng(seed).multinomial(entries.size, np.full(entries.size, 1.0 / entries.size),
                                                      size=resamples).astype(np.float64)
    draws: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    points: dict[tuple[str, str], list[float]] = defaultdict(list)
    seeds: dict[tuple[str, str], list[int]] = defaultdict(list)
    for model, s, p in runs:
        arrays = p["arrays"]
        if not np.array_equal(arrays[f"{subset}_entries"], entries):
            continue
        for key in arrays:
            if key.startswith(f"{subset}_oof_"):
                rep = key[len(f"{subset}_oof_"):]
                points[(model, rep)].append(r2(y, arrays[key]))
                draws[(model, rep)].append(r2(y, arrays[key], weights))
                seeds[(model, rep)].append(s)
    out: dict[str, Any] = {"available": True, "subset": subset, "entries": int(entries.size), "r2": {}, "contrasts": {}}
    for (model, rep), values in points.items():
        mean_draw = np.mean(draws[(model, rep)], axis=0)
        out["r2"].setdefault(model, {})[rep] = {"mean": float(np.mean(values)), "per_seed": dict(zip(seeds[(model, rep)], values)),
                                               "ci_low": float(np.quantile(mean_draw, 0.025)),
                                               "ci_high": float(np.quantile(mean_draw, 0.975))}
    (ma, ra), (mb, rb) = primary
    if (ma, ra) in points and (mb, rb) in points:
        diff = np.mean(draws[(ma, ra)], axis=0) - np.mean(draws[(mb, rb)], axis=0)
        point = float(np.mean(points[(ma, ra)]) - np.mean(points[(mb, rb)]))
        out["contrasts"][f"{ma} {ra} − {mb} {rb}"] = {**_interval(diff, point), "primary": True}
    for model in out["r2"]:
        if model != mb and (model, "table") in points and (mb, rb) in points and (model, "table") != (ma, ra):
            diff = np.mean(draws[(model, "table")], axis=0) - np.mean(draws[(mb, rb)], axis=0)
            out["contrasts"][f"{model} table − {mb} {rb}"] = _interval(diff, float(np.mean(points[(model, "table")]) - np.mean(points[(mb, rb)])))
    return out


# ---------------------------------------------------------------- queueing


def score_command(run_dir: Path, *, python: str = sys.executable, variants: Sequence[str] = ("ref",), batch_size: int | None = None,
                  scheme: str = PRIMARY_SCHEME) -> list[str]:
    return [python, "-m", "vsa_embed.experiments.e9_freqbias", "score", "--run", str(run_dir), "--variants", *variants,
            "--scheme", scheme, "--resume", *(["--eval-batch", str(batch_size)] if batch_size else [])]


def rows_command(run_dir: Path, *, python: str = sys.executable, with_tsne: bool = True) -> list[str]:
    return [python, "-m", "vsa_embed.experiments.e9_freqbias", "rows", "--run", str(run_dir), "--overwrite",
            *([] if with_tsne else ["--no-tsne"])]


def queue_stage(stage: str, *, priority: int = 55, kinds: Sequence[str] = ("score", "rows"), models: Sequence[str] | None = None,
                seeds: Sequence[int] | None = None, root: Path = ROOT, queue_dir: Path | None = None, python: str | None = None,
                dry_run: bool = False, licensed: bool = False) -> list[dict[str, Any]]:
    """One `score` job (GPU lane, at `priority`) and one `rows` job (CPU lane, at `priority` + 1) per config of `stage`
    (run folders need not exist yet); named `<stage>-<stem>-freqbias` / `-freqrows` (idempotent). Licensed stages
    (T1c) get no t-SNE figure."""
    from vsa_embed.jobqueue import DEFAULT_DIR, add

    from .cpt_plan import pinned_python
    from .e9_plan import EVAL_JOB_BATCH, _config_host, stage_python
    from .e9_rescore import model_of
    configs = sorted((Path(root) / "configs" / stage).glob("*.yaml"))
    hosts = [h for h in (_config_host(yaml.safe_load(p.read_text())) for p in configs) if h]
    python = python or (stage_python(hosts) if hosts else pinned_python())
    jobs = []
    for path in configs:
        model = model_of(path.stem)
        seed_match = re.search(r"-s(\d+)$", path.stem)
        if (models and model not in models) or (seeds and seed_match and int(seed_match[1]) not in seeds):
            continue
        config = yaml.safe_load(path.read_text())
        run_dir = Path(root) / "runs" / stage / path.stem
        if "score" in kinds:
            jobs.append({"name": f"{stage}-{path.stem}-freqbias", "priority": priority, "lane": None, "min_free_gb": 5,
                         "command": score_command(run_dir, python=python, batch_size=EVAL_JOB_BATCH.get(_config_host(config) or ""))})
        if "rows" in kinds:
            jobs.append({"name": f"{stage}-{path.stem}-freqrows", "priority": priority + 1, "lane": "cpu", "min_free_gb": 0,
                         "command": rows_command(run_dir, python=python, with_tsne=not licensed)})
    if dry_run:
        return jobs
    queued = []
    for job in jobs:
        try:
            add(queue_dir or DEFAULT_DIR, job["command"], name=job["name"], priority=job["priority"], min_free_gb=job["min_free_gb"],
                env={"PYTHONPATH": "src"}, resume_args=[], lane=job["lane"])
            queued.append(job)
        except FileExistsError:
            pass
    return queued


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("exclusions", help="build a track's exclusion table (strict non-copy stratum)")
    ex.add_argument("--track", required=True); ex.add_argument("--family", default="smollm2")
    ex.add_argument("--output", type=Path, default=None); ex.add_argument("--overwrite", action="store_true")
    score = sub.add_parser("score", help="frequency-bin and strict rescoring of one finished run")
    score.add_argument("--run", type=Path, required=True); score.add_argument("--output", type=Path, default=None)
    score.add_argument("--variants", nargs="+", default=["ref"], choices=list(VARIANTS))
    score.add_argument("--scheme", default=PRIMARY_SCHEME, choices=sorted(SCHEMES))
    score.add_argument("--fillers", type=Path, default=None); score.add_argument("--no-fillers", action="store_true")
    score.add_argument("--exclusions", type=Path, default=None); score.add_argument("--no-strict", action="store_true")
    score.add_argument("--device", default="cuda"); score.add_argument("--eval-batch", type=int, default=None)
    score.add_argument("--resume", action="store_true"); score.add_argument("--overwrite", action="store_true")
    rows = sub.add_parser("rows", help="A2: frequency decodability of a run's concept rows (CPU)")
    rows.add_argument("--run", type=Path, required=True); rows.add_argument("--output", type=Path, default=None)
    rows.add_argument("--folds", type=int, default=5); rows.add_argument("--seed", type=int, default=0)
    rows.add_argument("--tsne-points", type=int, default=1500); rows.add_argument("--tsne-iterations", type=int, default=750)
    rows.add_argument("--no-tsne", action="store_true"); rows.add_argument("--device", default="cpu")
    rows.add_argument("--overwrite", action="store_true")
    queue = sub.add_parser("queue", help="queue score (GPU) and rows (CPU) jobs for a stage's configs")
    queue.add_argument("--stage", required=True); queue.add_argument("--priority", type=int, default=55)
    queue.add_argument("--models", nargs="*", default=None); queue.add_argument("--seeds", type=int, nargs="*", default=None)
    queue.add_argument("--kind", nargs="+", default=["score", "rows"], choices=["score", "rows"])
    queue.add_argument("--root", type=Path, default=ROOT); queue.add_argument("--licensed", action="store_true")
    queue.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "exclusions":
        print(json.dumps({"path": str(ensure_exclusion_table(args.track, args.family, args.output, overwrite=args.overwrite))}))
    elif args.command == "score":
        fillers: Path | None | str = None if args.no_fillers else (args.fillers or "auto")
        exclusions: Path | None | str = None if args.no_strict else (args.exclusions or "auto")
        record = score_run(args.run, args.output, variants=args.variants, scheme=args.scheme, fillers=fillers, exclusions=exclusions,
                           device=args.device, eval_batch=args.eval_batch, resume=args.resume, overwrite=args.overwrite)
        print(json.dumps({"run": record["id"], "variants": sorted(record["variants"]), "ref_check": record.get("ref_check")}, default=str))
    elif args.command == "rows":
        record = probe_run(args.run, args.output, folds=args.folds, seed=args.seed, tsne_points=args.tsne_points,
                           tsne_iterations=args.tsne_iterations, with_tsne=not args.no_tsne, device=args.device, overwrite=args.overwrite)
        print(json.dumps({"run": record["id"], **{k: v.get("r2_seen") for k, v in record["representations"].items()}}, default=str))
    else:
        jobs = queue_stage(args.stage, priority=args.priority, kinds=args.kind, models=args.models, seeds=args.seeds, root=args.root,
                           dry_run=args.dry_run, licensed=args.licensed)
        print(json.dumps([{k: j[k] for k in ("name", "priority", "lane")} | {"command": " ".join(j["command"])} for j in jobs], indent=2))


if __name__ == "__main__":
    main()
