"""Lexical-semantic probes for trained channel models (WP-probe; experiments.md §0.11, E4.2, E4.3).

`ChannelModelAdapter` loads a trained run (`final.pt` + resolved config + the run's ontology and
alias table) and exposes the `ModelAdapter` interface (`word_state`, `_forward`, `token_logprobs`)
with the span channel live. Probe texts are tokenized with the run's tokenizer and linked by the
causal linker at the run's `min_subtokens` with the **full** alias table (held-out aliases included,
exactly as the evaluation corpus is linked), so held-out concepts are composed at zero update.
Without a channel (C0) the adapter is a plain model.

Probes, in prompting/similarity and linear-probe form where meaningful (§0.11):

- LAMBADA: last-word accuracy and loss (prompting; as `probes.lambada`).
- WiC: cosine threshold (prompting) and the logistic probe on `[|h₁ − h₂|, h₁ ⊙ h₂]` (`probes.wic`).
- CARD-660, Rare Words: Spearman of word-state cosine (`probes.word_similarity`).
- WSD (Raganato ALL): per-lemma#POS multinomial L2-logistic probe on frozen states trained on SemCor
  (at most `wsd_train_cap` sampled occurrences per lemma#POS), nearest sense centroid (similarity
  form), SemCor MFS and WordNet first-sense baselines. No gloss prompting: definitions never appear
  in prompts (§0.1).
- BLESS: 4-way {hyper, coord, mero, random-n} multinomial probe on `[h_x; h_y]` with a
  concept-disjoint split, plus a relatum-only control and an interaction-feature probe on
  `[|h_x − h_y|; h_x ⊙ h_y]` (concatenation lets a probe memorize prototypical relata); prompting: PMI of "The x is a kind of y" against "The thing is a kind of
  y" (per-concept average precision of hypernyms, pooled hyper-vs-rest AUC).
- HyperLex: Spearman of cosine, of the prompting PMI, and of a ridge probe on `[h_x; h_y]` trained
  on the lexical split (ridge strength chosen on its dev part from a fixed grid).

Every item records the link status of the entry injected at the position the probe reads (`heldout`;
`rare`/`mid`/`frequent` by its training frequency, as the trainer's strata; `unlinked`), and each
metric is also reported per status subset (E4.3). Per-item values go to a gzipped sidecar next to
the output so two runs can be compared item by item (`--compare`, paired bootstrap).

Runs saved with `train.save_trainable_only` (frozen or LoRA hosts, evaluation-only C0') hold no
frozen host weights; the host is rebuilt from the Hugging Face cache and the trainable state loaded
on top (`training.lm.load_model_state`, as `load_final`). `--quantize int8|int4` evaluates the run
after the post-training weight-only quantization of `e4_quant` (output head FP; `--quantize-channel`
= its variant B), so every probe can be repeated on the quantized model (E9).

    python -m vsa_embed.evaluation.channel_probes --run RUN [--probes all|lambada,wic,…] [--fast] --output RUN/probes.json
    python -m vsa_embed.evaluation.channel_probes --run RUN --quantize int4 [--quantize-channel] --output RUN/probes-int4.json
    python -m vsa_embed.evaluation.channel_probes --host gpt2 --output OUT/probes.json
    python -m vsa_embed.evaluation.channel_probes --compare A/probes.json B/probes.json [--output diff.json]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
import yaml
from scipy.stats import rankdata
from torch.nn import functional as F

from ..compose import FrameComposer, FrameSchedule
from ..data.semcor import WSDInstance, load_wsd_instances
from ..span_channel import AliasTable, CausalLinker, link_batch
from ..statistics import paired_bootstrap_ci
from .probes import (ModelAdapter, _fit_logistic, _spearman, _token_char_span, _wic_features, _wic_split, load_card660,
                     load_rare_words)

PROBES = ("lambada", "wic", "card660", "rare_words", "wsd", "bless", "hyperlex")
SCHEMA = "channel-probes/1"
WORD_PREFIX = "The word"           # as `probes.word_similarity`
STATUS_ORDER = ("heldout", "rare", "mid", "frequent", "seen", "unlinked")
SUBSETS = {"heldout": ("heldout",), "seen": ("rare", "mid", "frequent", "seen"), "unlinked": ("unlinked",),
           "rare": ("rare",), "mid": ("mid",), "frequent": ("frequent",)}
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class ProbeSettings:
    """Everything that fixes items, splits and probe fits; recorded in the output."""

    lambada_limit: int | None = None
    wic_train_limit: int | None = None
    wsd_dataset: str = "ALL"
    wsd_train_cap: int = 100          # SemCor occurrences per lemma#POS used to fit the WSD probe
    wsd_steps: int = 200              # L-BFGS iterations of the WSD probe (on GPT-2, predictions equal 400's)
    seed: int = 0                     # train-occurrence sampling and the BLESS concept split
    l2: float = 1e-2                  # as `probes._fit_logistic`
    steps: int = 400
    bless_test_fraction: float = 0.3
    hyperlex_split: str = "lexical"
    ridge_grid: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0, 10000.0)
    noun_template: str = "The {x} is a kind of {y}"
    noun_null: str = "thing"
    verb_template: str = "To {x} is a way to {y}"
    verb_null: str = "do"

    @classmethod
    def fast(cls) -> "ProbeSettings":
        """Screening subset: first 1,000 LAMBADA passages, 20 SemCor occurrences per lemma#POS."""
        return cls(lambada_limit=1000, wsd_train_cap=20)


# -- the adapter --------------------------------------------------------------------------------

def read_index(offsets: Sequence[tuple[int, int]], start: int, end: int) -> int:
    """Token whose state `ModelAdapter.word_state` reads: the last one overlapping `[start, end)`."""
    return max((i for i, (s, e) in enumerate(offsets) if s < end and e > start), default=len(offsets) - 1)


def entry_status(entries: Iterable[int], heldout: frozenset[int] | set[int], frequency: np.ndarray | None) -> str:
    """Status of the entries injected at one read position (bins as `training.lm.stratum_masks`)."""
    entries = list(entries)
    if not entries:
        return "unlinked"
    if any(e in heldout for e in entries):
        return "heldout"
    if frequency is None:
        return "seen"
    count = min(int(frequency[e]) for e in entries)
    return "rare" if count < 10 else "mid" if count < 100 else "frequent"


def combine_status(*statuses: str | None) -> str | None:
    """Status of a multi-word item: the first of `STATUS_ORDER` present (held-out dominates)."""
    present = [s for s in statuses if s is not None]
    return min(present, key=STATUS_ORDER.index) if present else None


@dataclass
class ChannelModelAdapter(ModelAdapter):
    """`ModelAdapter` over a trained `ChannelLM` whose spans come from the run's causal linker."""

    linker: CausalLinker | None = None
    heldout_entries: frozenset[int] = frozenset()
    train_frequency: np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.spans_fn is None and self.linker is not None and getattr(self.model, "channel", None) is not None:
            self.spans_fn = self.link_spans

    def link_spans(self, texts: Sequence[str], offsets: Sequence[Sequence[tuple[int, int]]]) -> dict[str, torch.Tensor]:
        spans = link_batch(self.linker, texts, offsets)
        # Corpora store confidences as float16; round the same way so the gate sees training values.
        spans["confidence"] = spans["confidence"].half().float()
        return spans

    def link_targets(self, texts: Sequence[str], char_spans: Sequence[tuple[int, int]]) -> list[list[int]]:
        """Entries injected at the position `word_state` reads, per (text, character span)."""
        if self.linker is None:
            return [[] for _ in texts]
        found: list[list[int]] = []
        for start in range(0, len(texts), 256):
            batch = list(texts[start:start + 256])
            encoded = self.tokenizer(batch, return_offsets_mapping=True, add_special_tokens=False, truncation=True,
                                     max_length=self.max_length)
            for text, offsets, (s, e) in zip(batch, encoded["offset_mapping"], char_spans[start:start + 256]):
                offsets = [tuple(o) for o in offsets]
                index = read_index(offsets, s, e)
                found.append(sorted({span.entry for span in self.linker.link(text, offsets) if span.inject_token == index}))
        return found

    def statuses(self, texts: Sequence[str], char_spans: Sequence[tuple[int, int]]) -> list[str]:
        return [entry_status(e, self.heldout_entries, self.train_frequency) for e in self.link_targets(texts, char_spans)]


def item_statuses(adapter: ModelAdapter, texts: Sequence[str], char_spans: Sequence[tuple[int, int]]) -> list[str | None]:
    """Link status per item, or `None` for adapters without a linker (untouched hosts)."""
    if isinstance(adapter, ChannelModelAdapter) and adapter.linker is not None:
        return adapter.statuses(texts, char_spans)
    return [None] * len(texts)


# -- alias table --------------------------------------------------------------------------------

def save_alias_table(table: AliasTable, path: Path) -> None:
    record = {"alias_to_entry": table.alias_to_entry, "entry_concepts": table.entry_concepts,
              "holdout": sorted(table.holdout), "sha256": table.digest()}
    if table.normalization != "default":          # identifier-mode tables (T2) record their mode
        record["normalization"] = table.normalization
    path.write_text(json.dumps(record) + "\n")


def load_alias_table(path: Path) -> AliasTable:
    data = json.loads(Path(path).read_text())
    return AliasTable({k: int(v) for k, v in data["alias_to_entry"].items()},
                      [tuple(int(c) for c in cs) for cs in data["entry_concepts"]],
                      frozenset(int(c) for c in data.get("holdout", ())), data.get("normalization", "default"))


def _holdout_concepts_from_names(ontology: dict[str, Any], candidates: Iterable[Path]) -> tuple[list[int] | None, str | None]:
    """Held-out concept ids from a `holdout_concepts.txt` whose hash matches `holdout_sha256`."""
    expected = ontology.get("holdout_sha256")
    if not expected:
        return None, None
    index = {name: i for i, name in enumerate(ontology["concept_names"])}
    for path in candidates:
        if path is None or not Path(path).is_file():
            continue
        names = Path(path).read_text().splitlines()
        if hashlib.sha256("\n".join(names).encode()).hexdigest() == expected:
            return sorted(index[n] for n in names), str(path)
    return None, None


def wordnet_alias_table(ontology: dict[str, Any], *, ontology_path: Path | None = None,
                        holdout_names: Path | None = None, wordnet: Any = None) -> tuple[AliasTable, dict[str, Any]]:
    """Rebuild the C3 alias table: every lemma name of every concept (a WordNet synset, in the
    ontology's concept order) is an alias, exactly as `build_wordnet_ontology` emits them."""
    if wordnet is None:
        from nltk.corpus import wordnet
    by_name = {s.name(): s for s in wordnet.all_synsets()}
    pairs = [(lemma.replace("_", " "), i) for i, name in enumerate(ontology["concept_names"])
             for lemma in by_name[name].lemma_names()]
    candidates = [holdout_names, ontology_path and Path(ontology_path).with_name("holdout_concepts.txt"),
                  *sorted((REPO_ROOT / "experiments").glob("*/runs/*/holdout_concepts.txt"))]
    holdout, source = _holdout_concepts_from_names(ontology, candidates)
    table = AliasTable.from_pairs(pairs, holdout=holdout or (), include_holdout=True)
    return table, {"source": "wordnet", "holdout_names": source, "holdout_known": holdout is not None}


def resolve_alias_table(ontology: dict[str, Any], ontology_path: Path | None = None, *, alias_table: Path | None = None,
                        holdout_names: Path | None = None) -> tuple[AliasTable, dict[str, Any]]:
    """The run's full (evaluation) alias table, checked against the ontology.

    Sources, in order: an explicit JSON (`save_alias_table`), `alias_to_entry` stored in the
    ontology, an `alias_table.json` next to `ontology.pt`, or a WordNet rebuild (C3). The table must
    reproduce the ontology's `entry_concepts` and, when its holdout is known, `alias_table_sha256`
    and `heldout_entries`; a mismatch raises.
    """
    sidecar = Path(ontology_path).with_name("alias_table.json") if ontology_path else None
    if alias_table is not None:
        table, info = load_alias_table(alias_table), {"source": str(alias_table), "holdout_known": True}
    elif ontology.get("alias_to_entry") is not None:
        table = AliasTable(dict(ontology["alias_to_entry"]), [tuple(c) for c in ontology["entry_concepts"]],
                           frozenset(ontology.get("holdout_concepts", ())))
        info = {"source": "ontology", "holdout_known": "holdout_concepts" in ontology}
    elif sidecar is not None and sidecar.is_file():
        table, info = load_alias_table(sidecar), {"source": str(sidecar), "holdout_known": True}
    elif ontology.get("concept_names") is not None:
        table, info = wordnet_alias_table(ontology, ontology_path=ontology_path, holdout_names=holdout_names)
    else:
        raise ValueError("cannot resolve the alias table: pass alias_table or store alias_to_entry in the ontology")
    if len(table.entry_concepts) != int(ontology["entry_count"]):
        raise ValueError(f"alias table has {len(table.entry_concepts)} entries, ontology {ontology['entry_count']}")
    checks: dict[str, bool | None] = {"entry_concepts": None, "digest": None, "heldout_entries": None}
    if ontology.get("entry_concepts") is not None:
        if [tuple(c) for c in ontology["entry_concepts"]] != [tuple(c) for c in table.entry_concepts]:
            raise ValueError("alias table entries differ from the ontology's entry_concepts")
        checks["entry_concepts"] = True
    if info["holdout_known"]:
        if ontology.get("alias_table_sha256"):
            if table.digest() != ontology["alias_table_sha256"]:
                raise ValueError("alias table digest differs from the ontology's alias_table_sha256")
            checks["digest"] = True
        if table.holdout:
            if table.heldout_entries() != {int(e) for e in ontology["heldout_entries"]}:
                raise ValueError("alias table held-out entries differ from the ontology's heldout_entries")
            checks["heldout_entries"] = True
    return table, {**info, "checks": checks, "aliases": len(table.alias_to_entry), "sha256": table.digest()}


# -- loading a run --------------------------------------------------------------------------------

def restore_composer_schedule(composer: FrameComposer, saved: dict[str, Any]) -> None:
    """Grow a freshly built composer to a checkpoint's dictionary (M3 growth) before loading weights."""
    extra = int(saved["atomics"]) - composer.atomics.shape[0]
    if extra > 0:
        composer.add_atomics(torch.zeros(extra, composer.atomics.shape[1]))
    extra_relations = int(saved["relations_count"]) - composer.relation_count
    if extra_relations > 0:
        composer.add_relation_copies(torch.zeros(extra_relations, dtype=torch.long))
    composer.set_schedule(FrameSchedule(saved["offsets"], saved["relations"], saved["fillers"]))


def run_tokenizer(config: dict[str, Any]) -> tuple[str, str]:
    """Tokenizer name and linker boundary of a run: the corpus manifest's tokenizer (it produced the
    training ids; hosts of one family share it, e.g. SmolLM2-360M runs on SmolLM2-135M-tokenized
    corpora), else the pretrained host's, else GPT-2 BPE (the from-scratch track)."""
    name, boundary = None, "prefix"
    for key in ("eval", "train"):
        path = config["data"].get(key)
        manifest = Path(path) / "manifest.json" if path else None
        if manifest is not None and manifest.is_file():
            data = json.loads(manifest.read_text())
            name, boundary = data.get("tokenizer"), data.get("boundary", boundary)
            break
    return name or config["model"].get("pretrained") or "gpt2", boundary


QUANTIZE_BITS = {"int8": 8, "int4": 4}


def quantize_model(model: torch.nn.Module, quantize: str | int, *, channel: bool = False,
                   group_size: str | int = "auto") -> dict[str, Any]:
    """Post-training weight-only quantization of a loaded `ChannelLM`, exactly as `e4_quant` (D4.3):
    LoRA merged, the host's linear layers INT8 (per row) or INT4 (group-wise, tile-packed; CUDA
    only) with torchao, input embedding and output head kept 16-bit; `channel=True` is variant B
    (the span channel and P1 context quantized too, simulated round-to-nearest), else variant A
    (channel FP16). Returns the record `e4_quant` keeps per variant (bits, group size, layers)."""
    from ..experiments.e4_quant import quantize_channel, quantize_host
    from .quantization import auto_group_size
    bits = QUANTIZE_BITS[quantize] if isinstance(quantize, str) else int(quantize)
    if bits not in (8, 4):
        raise ValueError("quantize must be int8 or int4")
    if bits == 4 and next(model.parameters()).device.type != "cuda":
        raise RuntimeError("INT4 (tile-packed) needs the model on CUDA; use int8 on CPU")
    size = auto_group_size(model.model) if group_size == "auto" else int(group_size)
    info: dict[str, Any] = {"variant": f"int{bits}-{'B' if channel else 'A'}", "bits": bits, "group_size": size,
                            "channel_quantized": bool(channel), **quantize_host(model, bits, size)}
    if channel:
        info["channel_tensors"] = sorted(quantize_channel(model, bits, size))
    return info


def load_run(run_dir: Path, *, checkpoint: str = "final.pt", device: torch.device | str | None = None,
             ontology_path: Path | None = None, alias_table: Path | None = None, holdout_names: Path | None = None,
             batch_size: int = 32, max_length: int = 256, layer: int = -1, quantize: str | int | None = None,
             quantize_channel: bool = False, group_size: str | int = "auto") -> ChannelModelAdapter:
    """Rebuild a trained run's `ChannelLM` (any condition) with its tokenizer and evaluation linker.

    `quantize` ("int8" / "int4") applies `quantize_model` after loading (`quantize_channel`: variant
    B); the record is kept under `info["quantization"]` (None = the run as trained, bf16 autocast)."""
    if quantize_channel and not quantize:
        raise ValueError("quantize_channel needs quantize")
    from transformers import AutoTokenizer

    from ..integrations.transformers import ChannelLM
    from ..training.lm import build_channel, build_model, load_model_state, resolve_config

    run_dir = Path(run_dir)
    state = torch.load(run_dir / checkpoint, weights_only=False, map_location="cpu")
    config = resolve_config(state.get("config") or yaml.safe_load((run_dir / "resolved_config.yaml").read_text()))
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ontology_source = ontology_path or config["data"].get("ontology")
    ontology_path = Path(ontology_source) if ontology_source else None
    ontology = torch.load(ontology_path, weights_only=False) if ontology_path else None
    base = build_model(config)
    channel, context = build_channel(config, ontology, base.get_input_embeddings().weight.shape[1])
    if channel is not None and ontology is not None:
        channel.set_unseen(ontology["heldout_entries"])
    if channel is not None and channel.composer is not None and state.get("composer_schedule"):
        restore_composer_schedule(channel.composer, state["composer_schedule"])
    host_mode = config["model"]["host_mode"] if config["model"]["pretrained"] else "train"
    model = ChannelLM(base, channel, context=context, host_mode=host_mode, lora_rank=int(config["model"]["lora_rank"]))
    # A trainable-only state (frozen or LoRA hosts, `train.save_trainable_only`; an evaluation-only C0')
    # omits the frozen host weights, which `build_model` has just reloaded from the Hugging Face cache.
    load_model_state(model, state["model"], trainable_only=bool(state.get("trainable_only", False)))
    model.to(device).eval()
    quantization = quantize_model(model, quantize, channel=quantize_channel, group_size=group_size) if quantize else None
    tokenizer_name, boundary = run_tokenizer(config)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    linker, heldout, frequency, table_info = None, frozenset(), None, None
    if ontology is not None:
        table, table_info = resolve_alias_table(ontology, ontology_path, alias_table=alias_table, holdout_names=holdout_names)
        linker = CausalLinker(table, boundary=boundary, min_subtokens=int(config["data"]["min_subtokens"]))
        heldout = frozenset(int(e) for e in ontology["heldout_entries"])
        if ontology.get("train_frequency") is not None:
            frequency = np.asarray(ontology["train_frequency"])
    if not config["model"]["pretrained"]:
        max_length = min(max_length, int(config["model"]["seq_len"]))   # learned positions end at seq_len
    info = {"run": str(run_dir), "checkpoint": checkpoint, "experiment": config.get("experiment"), "seed": config["seed"],
            "channel": config["channel"]["mode"], "channel_settings": config["channel"], "size": config["model"]["size"],
            "pretrained": config["model"]["pretrained"], "host_mode": host_mode, "tokenizer": tokenizer_name,
            "boundary": boundary, "min_subtokens": int(config["data"]["min_subtokens"]),
            "ontology": str(ontology_path) if ontology_path else None, "alias_table": table_info,
            "heldout_entries": len(heldout), "parameters": sum(p.numel() for p in model.parameters())}
    if state.get("trainable_only"):          # new keys only where they apply (headers of earlier outputs unchanged)
        info["trainable_only_checkpoint"] = True
    if quantization:
        info["quantization"] = quantization
    return ChannelModelAdapter(model, tokenizer, device, layer=layer, batch_size=batch_size, max_length=max_length,
                               linker=linker, heldout_entries=heldout, train_frequency=frequency, info=info)


# -- shared pieces --------------------------------------------------------------------------------

def states_at(adapter: ModelAdapter, texts: Sequence[str], spans: Sequence[Sequence[tuple[int, int]]]) -> list[torch.Tensor]:
    """States at several character spans per text (one forward per text), read as `word_state`.

    Same batches and model call as `ModelAdapter._forward`, but only the read rows leave the device.
    """
    out: list[torch.Tensor] = []
    for start in range(0, len(texts), adapter.batch_size):
        batch = list(texts[start:start + adapter.batch_size])
        encoded, outputs, _ = adapter._run_batch(batch)
        rows, columns, counts = [], [], []
        for r, (offsets, mask, targets) in enumerate(zip(encoded["offset_mapping"].tolist(), encoded["attention_mask"].tolist(),
                                                         spans[start:start + len(batch)])):
            offsets = [tuple(o) for o in offsets[:sum(mask)]]
            for s, e in targets:
                rows.append(r); columns.append(read_index(offsets, s, e))
            counts.append(len(targets))
        picked = outputs.hidden_states[adapter.layer][rows, columns].float().cpu()
        out += list(torch.split(picked, counts))
    return out


def word_states(adapter: ModelAdapter, words: Sequence[str]) -> tuple[torch.Tensor, list[str | None]]:
    """States of isolated words ("The word <w>", as `probes.word_similarity`) and their link status."""
    texts = [f"{WORD_PREFIX} {w.replace('_', ' ')}" for w in words]
    spans = [(len(WORD_PREFIX) + 1, len(t)) for t in texts]
    return adapter.word_state(texts, spans), item_statuses(adapter, texts, spans)


def continuation_logprob(adapter: ModelAdapter, prefixes: Sequence[str], continuations: Sequence[str]) -> np.ndarray:
    """`log p(continuation | prefix)` summed over the tokens overlapping the continuation."""
    texts = [p + c for p, c in zip(prefixes, continuations)]
    scores, offsets = adapter.token_logprobs(texts)
    out = np.zeros(len(texts))
    for i, (score, offs, prefix) in enumerate(zip(scores, offsets, prefixes)):
        out[i] = sum(float(score[t - 1]) for t in range(1, len(offs)) if offs[t][1] > len(prefix))
    return out


def template_pmi(adapter: ModelAdapter, pairs: Sequence[tuple[str, str]], template: str, null: str) -> np.ndarray:
    """`log p(y | template(x)) − log p(y | template(null))`: how much `x` raises `y` in the pattern."""
    head = template[:template.index("{y}")]
    prefixes = [head.format(x=x) for x, _ in pairs]
    ys = [y for _, y in pairs]
    distinct = sorted(set(ys))
    null_scores = dict(zip(distinct, continuation_logprob(adapter, [head.format(x=null)] * len(distinct), distinct)))
    return continuation_logprob(adapter, prefixes, ys) - np.asarray([null_scores[y] for y in ys])


def standardize(train: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    mean, std = train.mean(0), train.std(0) + 1e-6
    return [(x - mean) / std for x in (train, *others)]


def fit_grouped_softmax(x: torch.Tensor, groups: Sequence[int], labels: Sequence[int], class_counts: Sequence[int], *,
                        l2: float = 1e-2, steps: int = 400, device: torch.device | str = "cpu",
                        max_elements: int = 1 << 25, max_parameters: int = 1 << 22,
                        history: int = 10) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Independent L2 multinomial logistic regressions, one per group (e.g. one per lemma).

    Group `g` minimises mean cross-entropy over its rows + `l2·‖W_g‖²` (the objective of
    `probes._fit_logistic`, multinomial). Groups are solved jointly in padded chunks by full-batch
    L-BFGS; the joint objective is a sum of independent strictly convex terms, so its minimiser is
    the per-group minimisers. `x` (rows, d) is expected standardized. Chunks hold at most
    `max_elements` padded inputs and `max_parameters` weights (L-BFGS keeps `history` copies of them).
    Returns `(W_g, b_g)` per group.
    """
    rows: list[list[int]] = [[] for _ in class_counts]
    for i, g in enumerate(groups):
        rows[g].append(i)
    order = sorted(range(len(class_counts)), key=lambda g: (class_counts[g], len(rows[g]), g))
    labels_t = torch.as_tensor(list(labels), dtype=torch.long)
    d = x.shape[1]
    results: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(class_counts)
    position = 0
    while position < len(order):
        chunk = [order[position]]
        while position + len(chunk) < len(order):
            nxt = order[position + len(chunk)]
            size = (len(chunk) + 1) * max(len(rows[g]) for g in (*chunk, nxt)) * d
            weights = (len(chunk) + 1) * max(class_counts[g] for g in (*chunk, nxt)) * d
            if size > max_elements or weights > max_parameters:
                break
            chunk.append(nxt)
        position += len(chunk)
        k_max, n_max = max(class_counts[g] for g in chunk), max(max(1, len(rows[g])) for g in chunk)
        xp = torch.zeros(len(chunk), n_max, d); yp = torch.zeros(len(chunk), n_max, dtype=torch.long)
        mask = torch.zeros(len(chunk), n_max); valid = torch.zeros(len(chunk), k_max, dtype=torch.bool)
        for j, g in enumerate(chunk):
            index = torch.as_tensor(rows[g], dtype=torch.long)
            xp[j, :len(index)] = x[index].float(); yp[j, :len(index)] = labels_t[index]
            mask[j, :len(index)] = 1.0; valid[j, :class_counts[g]] = True
        xp, yp, mask, valid = (t.to(device) for t in (xp, yp, mask, valid))
        counts = mask.sum(1).clamp_min(1.0)
        w = torch.zeros(len(chunk), k_max, d, device=device, requires_grad=True)
        b = torch.zeros(len(chunk), k_max, device=device, requires_grad=True)
        optimizer = torch.optim.LBFGS([w, b], max_iter=steps, history_size=history, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            logits = (torch.bmm(xp, w.transpose(1, 2)) + b[:, None, :]).masked_fill(~valid[:, None, :], -1e9)
            ce = F.cross_entropy(logits.reshape(-1, k_max), yp.reshape(-1), reduction="none").view(len(chunk), n_max)
            loss = ((ce * mask).sum(1) / counts).sum() + l2 * w.square().sum()
            loss.backward()
            return loss

        optimizer.step(closure)
        for j, g in enumerate(chunk):
            results[g] = (w.detach()[j, :class_counts[g]].cpu(), b.detach()[j, :class_counts[g]].cpu())
    return results  # type: ignore[return-value]


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Ridge regression on standardized features: `(w, b)` minimising `‖Xw + b − y‖² + α‖w‖²`."""
    b = float(y.mean())
    w = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ (y - b))
    return w, b


# -- metrics ------------------------------------------------------------------------------------

def spearman_tied(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Spearman correlation with average ranks for ties (`probes._spearman` uses ordinal ranks)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size < 3:
        return None
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denominator = math.sqrt(float((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / denominator) if denominator else None


def macro_f1(predicted: Sequence[Any], gold: Sequence[Any], classes: Sequence[Any] | None = None) -> float | None:
    predicted, gold = list(predicted), list(gold)
    if not gold:
        return None
    classes = list(classes) if classes is not None else sorted(set(gold) | set(predicted))
    scores = []
    for c in classes:
        tp = sum(p == c and g == c for p, g in zip(predicted, gold))
        fp = sum(p == c and g != c for p, g in zip(predicted, gold))
        fn = sum(p != c and g == c for p, g in zip(predicted, gold))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(scores))


def roc_auc(scores: Sequence[float], positive: Sequence[bool]) -> float | None:
    scores, positive = np.asarray(scores, dtype=float), np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return None
    ranks = rankdata(scores)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(scores: Sequence[float], positive: Sequence[bool]) -> float | None:
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    hits, total = 0, 0.0
    for rank, i in enumerate(order, 1):
        if positive[i]:
            hits += 1; total += hits / rank
    return total / hits if hits else None


# Per probe table: metric → (kind, value columns, row-mask column or None). Subset metrics and the
# paired comparison are both computed from these, so they always agree with each other.
METRICS: dict[str, dict[str, tuple[str, tuple[str, ...], str | None]]] = {
    "lambada": {"accuracy": ("mean", ("correct",), None), "word_loss": ("mean", ("loss",), None)},
    "wic": {"prompt_accuracy": ("mean", ("prompt_correct",), None), "probe_accuracy": ("mean", ("probe_correct",), None)},
    "card660": {"spearman_tied": ("spearman", ("cosine", "gold"), None)},
    "rare_words": {"spearman_tied": ("spearman", ("cosine", "gold"), None)},
    "wsd": {"probe_f1": ("mean", ("probe_correct",), None), "centroid_f1": ("mean", ("centroid_correct",), None),
            "mfs_f1": ("mean", ("mfs_correct",), None), "wn1_f1": ("mean", ("wn1_correct",), None)},
    "bless": {"probe_accuracy": ("mean", ("probe_correct",), "test"),
              "probe_macro_f1": ("macro_f1", ("probe_pred", "relation"), "test"),
              "pair_probe_accuracy": ("mean", ("pair_probe_correct",), "test"),
              "pair_probe_macro_f1": ("macro_f1", ("pair_probe_pred", "relation"), "test"),
              "prompt_auc": ("auc", ("pmi", "is_hyper"), None)},
    "bless_concepts": {"prompt_map": ("mean", ("ap",), None)},
    "hyperlex": {"cosine_spearman": ("spearman", ("cosine", "gold"), None),
                 "prompt_spearman": ("spearman", ("pmi", "gold"), None),
                 "probe_spearman": ("spearman", ("probe_pred", "gold"), "test")},
}


def compute_metric(kind: str, columns: Sequence[Sequence[Any]], rows: Sequence[int]) -> float | None:
    values = [[column[i] for i in rows] for column in columns]
    if kind == "mean":
        return float(np.mean(np.asarray(values[0], dtype=float))) if rows else None
    if kind == "spearman":
        return spearman_tied(*values)
    if kind == "macro_f1":
        return macro_f1(*values)
    if kind == "auc":
        return roc_auc(*values)
    raise ValueError(f"unknown metric kind {kind}")


def _metric_rows(table: dict[str, list[Any]], mask: str | None, members: Sequence[str] | None = None) -> list[int]:
    status = table.get("status") or [None] * len(table["id"])
    return [i for i in range(len(table["id"])) if (mask is None or table[mask][i])
            and (members is None or status[i] in members)]


def subset_metrics(name: str, table: dict[str, list[Any]]) -> dict[str, dict[str, Any]]:
    """Every registered metric of a table on each link-status subset (E4.3)."""
    if not any(s is not None for s in table.get("status") or []):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for subset, members in SUBSETS.items():
        entry: dict[str, Any] = {}
        for metric, (kind, columns, mask) in METRICS[name].items():
            rows = _metric_rows(table, mask, members)
            entry[metric] = {"value": compute_metric(kind, [table[c] for c in columns], rows) if rows else None, "n": len(rows)}
        out[subset] = entry
    return out


# -- probes -------------------------------------------------------------------------------------

def probe_lambada(adapter: ModelAdapter, root: Path, settings: ProbeSettings, **_: Any):
    """`probes.lambada` with per-item outcomes; the aggregate numbers are identical."""
    path = root / "lambada" / "data" / "lambada_test_en.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()][:settings.lambada_limit]
    texts, word_ids_all, context_lengths, spans = [], [], [], []
    for row in rows:
        context, word = row["text"].rsplit(" ", 1)
        context_lengths.append(len(adapter.tokenizer(context, add_special_tokens=False)["input_ids"]))
        word_ids_all.append(adapter.tokenizer(" " + word, add_special_tokens=False)["input_ids"])
        texts.append(context + " " + word)
        spans.append((len(context) + 1, len(context) + 1 + len(word)))
    adapter.need_logits = True
    kept, correct, losses = [], [], []
    try:
        for start in range(0, len(texts), adapter.batch_size):
            _, _, logits_batch, ids_batch = adapter._forward(texts[start:start + adapter.batch_size])
            for k, (logits, ids) in enumerate(zip(logits_batch, ids_batch)):
                i = start + k
                word_ids, n_context = word_ids_all[i], context_lengths[i]
                positions = list(range(n_context - 1, n_context - 1 + len(word_ids)))
                if positions[-1] >= logits.shape[0] or ids[n_context:n_context + len(word_ids)] != word_ids:
                    continue
                predicted = logits[positions].argmax(-1).tolist()
                log_probs = torch.log_softmax(logits[positions], -1)
                kept.append(i); correct.append(int(predicted == word_ids))
                losses.append(-float(log_probs[torch.arange(len(word_ids)), torch.tensor(word_ids)].mean()))
    finally:
        adapter.need_logits = False
    metrics = {"accuracy": sum(correct) / max(1, len(losses)), "word_loss": float(np.mean(losses)), "n": len(losses),
               "skipped": len(rows) - len(kept)}
    status = item_statuses(adapter, [texts[i] for i in kept], [spans[i] for i in kept])
    return metrics, {"lambada": {"id": kept, "correct": correct, "loss": losses, "status": status}}, {}


def probe_wic(adapter: ModelAdapter, root: Path, settings: ProbeSettings, **_: Any):
    """`probes.wic` with per-item outcomes on the dev split (train → threshold and probe)."""
    train, dev = _wic_split(root / "wic", "train")[:settings.wic_train_limit], _wic_split(root / "wic", "dev")
    t1, t2, ty = _wic_features(adapter, train)
    d1, d2, dy = _wic_features(adapter, dev)
    train_cos = F.cosine_similarity(t1, t2).numpy(); dev_cos = F.cosine_similarity(d1, d2).numpy()
    thresholds = np.unique(train_cos)
    accuracies = [((train_cos >= t) == ty).mean() for t in thresholds]
    threshold = float(thresholds[int(np.argmax(accuracies))])
    features = lambda a, b: torch.cat([(a - b).abs(), a * b], -1).numpy()
    w, b, mean, std = _fit_logistic(features(t1, t2), ty.astype(np.float64))
    dev_pred = (((features(d1, d2) - mean) / std) @ w + b) > 0
    prompt_correct, probe_correct = (dev_cos >= threshold) == dy, dev_pred == dy
    metrics = {"prompt_accuracy": float(prompt_correct.mean()), "probe_accuracy": float(probe_correct.mean()),
               "majority": float(max(dy.mean(), 1 - dy.mean())), "n_dev": int(len(dy)), "threshold": threshold}
    s1 = item_statuses(adapter, [it["s1"] for it in dev], [_token_char_span(it["s1"], it["i1"]) for it in dev])
    s2 = item_statuses(adapter, [it["s2"] for it in dev], [_token_char_span(it["s2"], it["i2"]) for it in dev])
    table = {"id": list(range(len(dev))), "prompt_correct": prompt_correct.astype(int).tolist(),
             "probe_correct": probe_correct.astype(int).tolist(), "cosine": dev_cos.astype(float).tolist(),
             "status": [combine_status(a, b) for a, b in zip(s1, s2)]}
    split = {"train": "wic/train" + (f"[:{settings.wic_train_limit}]" if settings.wic_train_limit else ""), "test": "wic/dev"}
    return metrics, {"wic": table}, {"split": split}


def _similarity(adapter: ModelAdapter, pairs: list[tuple[str, str, float]], name: str):
    """`probes.word_similarity` with per-pair cosines (same batches, so the same numbers)."""
    words = sorted({w for a, b, _ in pairs for w in (a, b)})
    states, status = word_states(adapter, words)
    index = {w: i for i, w in enumerate(words)}
    cos = np.asarray([float(F.cosine_similarity(states[index[a]], states[index[b]], dim=0)) for a, b, _ in pairs])
    gold = np.asarray([s for _, _, s in pairs])
    metrics = {"spearman": _spearman(cos, gold), "spearman_tied": spearman_tied(cos, gold), "n": len(pairs)}
    table = {"id": list(range(len(pairs))), "cosine": cos.tolist(), "gold": gold.tolist(),
             "status": [combine_status(status[index[a]], status[index[b]]) for a, b, _ in pairs]}
    return metrics, {name: table}, {}


def probe_card660(adapter: ModelAdapter, root: Path, settings: ProbeSettings, **_: Any):
    return _similarity(adapter, load_card660(root / "card660.tsv"), "card660")


def probe_rare_words(adapter: ModelAdapter, root: Path, settings: ProbeSettings, **_: Any):
    return _similarity(adapter, load_rare_words(root / "rw" / "rw" / "rw.txt"), "rare_words")


WN_POS = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}


def probe_wsd(adapter: ModelAdapter, root: Path, settings: ProbeSettings, *, wordnet: Any = None, device: Any = None, **_: Any):
    """All-words WSD on Raganato ALL with a per-lemma#POS linear probe trained on SemCor states."""
    if wordnet is None:
        from nltk.corpus import wordnet
    base = root / "wsd" / "WSD_Evaluation_Framework"
    semcor = load_wsd_instances(base / "Training_Corpora" / "SemCor" / "semcor.data.xml",
                                base / "Training_Corpora" / "SemCor" / "semcor.gold.key.txt")
    name = settings.wsd_dataset
    test = load_wsd_instances(base / "Evaluation_Datasets" / name / f"{name}.data.xml",
                              base / "Evaluation_Datasets" / name / f"{name}.gold.key.txt")
    resolved: dict[str, str | None] = {}

    def synset(key: str) -> str | None:
        if key not in resolved:
            try:
                resolved[key] = wordnet.lemma_from_key(key).synset().name()
            except Exception:
                resolved[key] = None
        return resolved[key]

    senses_cache: dict[tuple[str, str], list[str]] = {}

    def senses(key: tuple[str, str]) -> list[str]:
        if key not in senses_cache:
            senses_cache[key] = [s.name() for s in wordnet.synsets(key[0], WN_POS[key[1]])]
        return senses_cache[key]

    key_of = lambda inst: (inst.lemma.lower(), inst.pos)
    test_keys = {key_of(t) for t in test}
    # Only lemma#POS of the test set matter (probe, centroids and MFS); first gold key, as load_semcor.
    train = [(inst, synset(inst.keys[0])) for inst in semcor if key_of(inst) in test_keys]
    train = [(inst, s) for inst, s in train if s is not None]
    counts: dict[tuple[str, str], Counter] = {}
    by_key: dict[tuple[str, str], list[int]] = {}
    for i, (inst, s) in enumerate(train):
        counts.setdefault(key_of(inst), Counter())[s] += 1
        by_key.setdefault(key_of(inst), []).append(i)
    rng = np.random.default_rng(settings.seed)
    chosen: list[int] = []
    for k in sorted(by_key):
        ids = by_key[k]
        chosen += ids if len(ids) <= settings.wsd_train_cap else sorted(rng.choice(ids, settings.wsd_train_cap, replace=False).tolist())

    def sentence_states(instances: Sequence[WSDInstance]) -> torch.Tensor:
        order: dict[str, int] = {}
        texts, spans, where = [], [], []
        for inst in instances:
            if inst.sentence_id not in order:
                order[inst.sentence_id] = len(texts); texts.append(inst.text); spans.append([])
            j = order[inst.sentence_id]
            where.append((j, len(spans[j]))); spans[j].append((inst.start, inst.end))
        states = states_at(adapter, texts, spans)
        return torch.stack([states[j][r] for j, r in where])

    x_train = sentence_states([train[i][0] for i in chosen]).numpy()
    x_test = sentence_states(test).numpy()
    x_train, x_test = standardize(x_train, x_test)
    rank = lambda k, s: senses(k).index(s) if s in senses(k) else len(senses(k))
    chosen_keys, chosen_senses = [key_of(train[i][0]) for i in chosen], [train[i][1] for i in chosen]
    classes: dict[tuple[str, str], list[str]] = {}
    for k, s in zip(chosen_keys, chosen_senses):
        if s not in classes.setdefault(k, []):
            classes[k].append(s)
    for k in classes:
        classes[k].sort(key=lambda s: (rank(k, s), s))
    ambiguous = sorted(k for k, c in classes.items() if len(c) > 1)
    group_of = {k: g for g, k in enumerate(ambiguous)}
    rows = [j for j, k in enumerate(chosen_keys) if k in group_of]
    fits = fit_grouped_softmax(torch.from_numpy(x_train[rows]).float(), [group_of[chosen_keys[j]] for j in rows],
                               [classes[chosen_keys[j]].index(chosen_senses[j]) for j in rows],
                               [len(classes[k]) for k in ambiguous], l2=settings.l2, steps=settings.wsd_steps,
                               device=device or "cpu")
    unit = x_train / np.linalg.norm(x_train, axis=1, keepdims=True)
    centroids: dict[tuple[str, str], np.ndarray] = {}
    members: dict[tuple[str, str], list[list[int]]] = {k: [[] for _ in classes[k]] for k in ambiguous}
    for j in rows:
        members[chosen_keys[j]][classes[chosen_keys[j]].index(chosen_senses[j])].append(j)
    for k in ambiguous:
        c = np.stack([unit[m].mean(0) for m in members[k]])
        centroids[k] = c / np.linalg.norm(c, axis=1, keepdims=True)

    def mfs(k: tuple[str, str]) -> str | None:
        if k in counts:
            return max(counts[k], key=lambda s: (counts[k][s], -rank(k, s), s))
        return senses(k)[0] if senses(k) else None

    table: dict[str, list[Any]] = {c: [] for c in ("id", "dataset", "pos", "probe_correct", "centroid_correct", "mfs_correct",
                                                   "wn1_correct", "ambiguous", "backoff", "gold_resolved")}
    for t, inst in enumerate(test):
        k = key_of(inst)
        gold = {s for s in (synset(key) for key in inst.keys) if s is not None}
        wn1 = senses(k)[0] if senses(k) else None
        if k in group_of:
            w, b = fits[group_of[k]]
            probe = classes[k][int(torch.argmax(torch.from_numpy(x_test[t]).float() @ w.T + b))]
            centroid = classes[k][int(np.argmax(centroids[k] @ (x_test[t] / np.linalg.norm(x_test[t]))))]
        elif k in classes:
            probe = centroid = classes[k][0]
        else:
            probe = centroid = wn1
        table["id"].append(inst.instance_id); table["dataset"].append(inst.instance_id.split(".")[0])
        table["pos"].append(inst.pos)
        for column, prediction in (("probe_correct", probe), ("centroid_correct", centroid), ("mfs_correct", mfs(k)),
                                   ("wn1_correct", wn1)):
            table[column].append(int(prediction in gold))
        table["ambiguous"].append(k in group_of); table["backoff"].append(k not in classes)
        table["gold_resolved"].append(bool(gold))
    table["status"] = item_statuses(adapter, [inst.text for inst in test], [(inst.start, inst.end) for inst in test])

    def scores(rows: list[int]) -> dict[str, Any]:
        return {"n": len(rows), **{m: compute_metric("mean", [table[c]], rows) for m, (_, (c,), _) in METRICS["wsd"].items()}}

    metrics: dict[str, Any] = scores(list(range(len(test))))
    for column in ("dataset", "pos"):
        metrics[f"by_{column}"] = {v: scores([i for i, x in enumerate(table[column]) if x == v]) for v in sorted(set(table[column]))}
    metrics["ambiguous"] = scores([i for i, a in enumerate(table["ambiguous"]) if a])
    metrics["backoff_items"] = int(sum(table["backoff"])); metrics["unresolved_gold"] = int(len(test) - sum(table["gold_resolved"]))
    split = {"train": "SemCor", "test": name, "train_cap_per_lemma_pos": settings.wsd_train_cap, "seed": settings.seed,
             "train_occurrences": len(chosen), "lemma_pos_probes": len(ambiguous), "lemma_pos_in_test": len(test_keys),
             "train_sentences": len({train[i][0].sentence_id for i in chosen})}
    return metrics, {"wsd": table}, {"split": split}


BLESS_RELATIONS = ("hyper", "coord", "mero", "random-n")


def load_bless(path: Path) -> list[dict[str, str]]:
    """BLESS tuples: concept, class, relation, relatum (word and POS from the `-n/-v/-j` suffix)."""
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        concept, klass, relation, relatum = parts
        word, pos = relatum.rsplit("-", 1)
        rows.append({"concept": concept.rsplit("-", 1)[0], "class": klass, "relation": relation, "relatum": word, "pos": pos})
    return rows


def probe_bless(adapter: ModelAdapter, root: Path, settings: ProbeSettings, *, device: Any = None, **_: Any):
    """Hypernym vs co-hyponym / meronym / random noun: probe on `[h_x; h_y]` and template PMI."""
    rows = [r for r in load_bless(root / "bless" / "bless-gems" / "BLESS.txt")
            if r["relation"] in BLESS_RELATIONS and r["pos"] == "n"]
    concepts = sorted({r["concept"] for r in rows})
    permutation = np.random.default_rng(settings.seed).permutation(len(concepts))
    test_concepts = sorted(concepts[i] for i in permutation[:round(len(concepts) * settings.bless_test_fraction)])
    test_set = set(test_concepts)
    words = sorted({r["concept"] for r in rows} | {r["relatum"] for r in rows})
    states, status = word_states(adapter, words)
    index = {w: i for i, w in enumerate(words)}
    hx, hy = states[[index[r["concept"]] for r in rows]], states[[index[r["relatum"]] for r in rows]]
    is_test = np.asarray([r["concept"] in test_set for r in rows])
    labels = np.asarray([BLESS_RELATIONS.index(r["relation"]) for r in rows])

    def probe(features: torch.Tensor) -> np.ndarray:
        x = features.numpy().astype(np.float64)
        x_train, x_test = standardize(x[~is_test], x[is_test])
        (w, b), = fit_grouped_softmax(torch.from_numpy(x_train).float(), [0] * len(x_train), labels[~is_test].tolist(),
                                      [len(BLESS_RELATIONS)], l2=settings.l2, steps=settings.steps, device=device or "cpu")
        return (torch.from_numpy(x_test).float() @ w.T + b).argmax(-1).numpy()

    predicted = probe(torch.cat([hx, hy], -1))
    # Controls: the relatum alone (lexical memorization of prototypical hypernyms, Levy et al. 2015),
    # and WiC-style interaction features, which cannot memorize the relatum by position.
    relatum_only = probe(hy)
    interaction = probe(torch.cat([(hx - hy).abs(), hx * hy], -1))
    pmi = template_pmi(adapter, [(r["concept"], r["relatum"]) for r in rows], settings.noun_template, settings.noun_null)
    test_rows = np.flatnonzero(is_test)
    probe_pred: list[str | None] = [None] * len(rows); probe_correct: list[int | None] = [None] * len(rows)
    pair_pred: list[str | None] = [None] * len(rows); pair_correct: list[int | None] = [None] * len(rows)
    for j, i in enumerate(test_rows):
        probe_pred[i] = BLESS_RELATIONS[predicted[j]]; probe_correct[i] = int(predicted[j] == labels[i])
        pair_pred[i] = BLESS_RELATIONS[interaction[j]]; pair_correct[i] = int(interaction[j] == labels[i])
    pair_status = [combine_status(status[index[r["concept"]]], status[index[r["relatum"]]]) for r in rows]
    table = {"id": [f"{r['concept']}|{r['relation']}|{r['relatum']}" for r in rows], "relation": [r["relation"] for r in rows],
             "test": is_test.tolist(), "probe_pred": probe_pred, "probe_correct": probe_correct,
             "pair_probe_pred": pair_pred, "pair_probe_correct": pair_correct, "pmi": pmi.tolist(),
             "is_hyper": [r["relation"] == "hyper" for r in rows], "status": pair_status}
    concept_table: dict[str, list[Any]] = {"id": [], "ap": [], "test": [], "status": []}
    for concept in concepts:
        members = [i for i, r in enumerate(rows) if r["concept"] == concept]
        ap = average_precision(pmi[members], [rows[i]["relation"] == "hyper" for i in members])
        if ap is None:
            continue
        concept_table["id"].append(concept); concept_table["ap"].append(ap)
        concept_table["test"].append(concept in test_set); concept_table["status"].append(status[index[concept]])
    train_labels = labels[~is_test]
    majority = int(np.bincount(train_labels, minlength=len(BLESS_RELATIONS)).argmax())
    gold_test = [BLESS_RELATIONS[labels[i]] for i in test_rows]
    pred_test = [probe_pred[i] for i in test_rows]
    metrics = {
        "probe_accuracy": float(np.mean([probe_correct[i] for i in test_rows])),
        "probe_macro_f1": macro_f1(pred_test, gold_test, BLESS_RELATIONS),
        "probe_f1_by_relation": {c: macro_f1([p == c for p in pred_test], [g == c for g in gold_test], [True]) for c in BLESS_RELATIONS},
        "majority_accuracy": float(np.mean(labels[is_test] == majority)),
        "relatum_only_accuracy": float(np.mean(relatum_only == labels[is_test])),
        "relatum_only_macro_f1": macro_f1([BLESS_RELATIONS[p] for p in relatum_only], gold_test, BLESS_RELATIONS),
        "pair_probe_accuracy": float(np.mean(interaction == labels[is_test])),
        "pair_probe_macro_f1": macro_f1([BLESS_RELATIONS[p] for p in interaction], gold_test, BLESS_RELATIONS),
        "prompt_map": float(np.mean(concept_table["ap"])),
        "prompt_map_test_concepts": float(np.mean([a for a, t in zip(concept_table["ap"], concept_table["test"]) if t])),
        "prompt_auc": roc_auc(pmi, table["is_hyper"]),
        "n_pairs": len(rows), "n_test_pairs": int(is_test.sum()),
    }
    split = {"unit": "concept", "seed": settings.seed, "test_fraction": settings.bless_test_fraction,
             "test_concepts": test_concepts, "relations": list(BLESS_RELATIONS),
             "features": {"probe": "[h_x; h_y]", "pair_probe": "[|h_x − h_y|; h_x ⊙ h_y]", "relatum_only": "h_y"},
             "template": settings.noun_template, "null": settings.noun_null}
    return metrics, {"bless": table, "bless_concepts": concept_table}, {"split": split}


def load_hyperlex(path: Path) -> list[dict[str, Any]]:
    """HyperLex pairs (`WORD1 WORD2 POS TYPE AVG_SCORE …`; score 0–6, "to what degree is X a type of Y")."""
    rows = []
    for line in Path(path).read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 5:
            rows.append({"x": parts[0], "y": parts[1], "pos": parts[2], "type": parts[3], "score": float(parts[4])})
    return rows


def hyperlex_split(root: Path, kind: str = "lexical") -> dict[tuple[str, str, str], str]:
    """(x, y, POS) → train | dev | test from the official split files."""
    folder = root / "splits" / kind
    split = {}
    for part, name in (("train", "training"), ("dev", "dev"), ("test", "test")):
        for row in load_hyperlex(folder / f"hyperlex_{name}_all_{kind}.txt"):
            split[(row["x"], row["y"], row["pos"])] = part
    return split


def probe_hyperlex(adapter: ModelAdapter, root: Path, settings: ProbeSettings, **_: Any):
    """Graded hypernymy: cosine, template PMI and a ridge probe on `[h_x; h_y]`, all by Spearman."""
    folder = root / "hyperlex"
    rows = load_hyperlex(folder / "hyperlex-all.txt")
    split_of = hyperlex_split(folder, settings.hyperlex_split)
    parts = [split_of.get((r["x"], r["y"], r["pos"]), "none") for r in rows]
    words = sorted({w for r in rows for w in (r["x"], r["y"])})
    states, status = word_states(adapter, words)
    index = {w: i for i, w in enumerate(words)}
    hx, hy = states[[index[r["x"]] for r in rows]], states[[index[r["y"]] for r in rows]]
    cosine = F.cosine_similarity(hx, hy).numpy().astype(float)
    gold = np.asarray([r["score"] for r in rows])
    pmi = np.zeros(len(rows))
    for pos, template, null in (("N", settings.noun_template, settings.noun_null), ("V", settings.verb_template, settings.verb_null)):
        members = [i for i, r in enumerate(rows) if r["pos"] == pos]
        if members:
            pmi[members] = template_pmi(adapter, [(rows[i]["x"], rows[i]["y"]) for i in members], template, null)
    features = torch.cat([hx, hy], -1).numpy().astype(np.float64)
    masks = {p: np.asarray([q == p for q in parts]) for p in ("train", "dev", "test")}
    x_train, x_dev, x_test = standardize(features[masks["train"]], features[masks["dev"]], features[masks["test"]])
    dev_scores = []
    for alpha in settings.ridge_grid:
        w, b = fit_ridge(x_train, gold[masks["train"]], alpha)
        dev_scores.append(spearman_tied(x_dev @ w + b, gold[masks["dev"]]) or -1.0)
    alpha = settings.ridge_grid[int(np.argmax(dev_scores))]
    w, b = fit_ridge(x_train, gold[masks["train"]], alpha)
    probe_pred: list[float | None] = [None] * len(rows)
    for j, i in enumerate(np.flatnonzero(masks["test"])):
        probe_pred[i] = float(x_test[j] @ w + b)
    test_rows = np.flatnonzero(masks["test"])
    table = {"id": list(range(len(rows))), "pos": [r["pos"] for r in rows], "split": parts, "test": masks["test"].tolist(),
             "gold": gold.tolist(), "cosine": cosine.tolist(), "pmi": pmi.tolist(), "probe_pred": probe_pred,
             "status": [combine_status(status[index[r["x"]]], status[index[r["y"]]]) for r in rows]}
    metrics = {"cosine_spearman": spearman_tied(cosine, gold), "prompt_spearman": spearman_tied(pmi, gold),
               "probe_spearman": spearman_tied([probe_pred[i] for i in test_rows], gold[test_rows]),
               "cosine_spearman_test": spearman_tied(cosine[test_rows], gold[test_rows]),
               "prompt_spearman_test": spearman_tied(pmi[test_rows], gold[test_rows]),
               "ridge_alpha": alpha, "ridge_dev_spearman": dict(zip(map(str, settings.ridge_grid), dev_scores)),
               "n": len(rows), "n_train": int(masks["train"].sum()), "n_dev": int(masks["dev"].sum()), "n_test": len(test_rows)}
    split = {"probe": f"hyperlex/splits/{settings.hyperlex_split}", "features": "[h_x; h_y]",
             "templates": {"N": [settings.noun_template, settings.noun_null], "V": [settings.verb_template, settings.verb_null]}}
    return metrics, {"hyperlex": table}, {"split": split}


RUNNERS: dict[str, Callable[..., Any]] = {
    "lambada": probe_lambada, "wic": probe_wic, "card660": probe_card660, "rare_words": probe_rare_words,
    "wsd": probe_wsd, "bless": probe_bless, "hyperlex": probe_hyperlex,
}


def run_channel_probes(adapter: ModelAdapter, root: Path, *, probes: Sequence[str] = PROBES,
                       settings: ProbeSettings | None = None, wordnet: Any = None,
                       fit_device: torch.device | str | None = None) -> tuple[dict[str, Any], dict[str, dict[str, list[Any]]]]:
    """Run the probes; returns (per-probe results with subset metrics, per-item tables)."""
    settings = settings or ProbeSettings()
    fit_device = fit_device or adapter.device
    results: dict[str, Any] = {}
    tables: dict[str, dict[str, list[Any]]] = {}
    for name in probes:
        if name not in RUNNERS:
            raise ValueError(f"unknown probe {name}; choose from {', '.join(PROBES)}")
        started = time.monotonic()
        metrics, probe_tables, extra = RUNNERS[name](adapter, Path(root), settings, wordnet=wordnet, device=fit_device)
        tables.update(probe_tables)
        results[name] = {"metrics": metrics, **extra,
                         "subsets": {t: subset_metrics(t, table) for t, table in probe_tables.items()},
                         "seconds": round(time.monotonic() - started, 2)}
    return results, tables


# -- output and paired comparison ---------------------------------------------------------------

def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(v) for v in value]
    if isinstance(value, np.ndarray):
        return _native(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def items_path(output: Path) -> Path:
    return Path(output).with_name(Path(output).stem + ".items.json.gz")


def write_outputs(output: Path, results: dict[str, Any], tables: dict[str, Any], header: dict[str, Any]) -> dict[str, Any]:
    """`output` (summary, settings, subsets) plus `<stem>.items.json.gz` (per-item tables, for pairing)."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:   # mtime 0: identical bytes for identical items
        handle.write(json.dumps({"schema": SCHEMA, "tables": _native(tables)}, sort_keys=True).encode())
    items = items_path(output)
    items.write_bytes(buffer.getvalue())
    summary = {f"{probe}/{metric}": value for probe, result in results.items()
               for metric, value in result["metrics"].items() if isinstance(value, (int, float)) or value is None}
    document = {"schema": SCHEMA, **header, "summary": summary, "probes": results,
                "items_file": items.name, "items_sha256": hashlib.sha256(buffer.getvalue()).hexdigest()}
    output.write_text(json.dumps(_native(document), indent=2) + "\n")
    return document


def load_items(output: Path) -> dict[str, dict[str, list[Any]]]:
    document = json.loads(Path(output).read_text())
    raw = (Path(output).parent / document["items_file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != document["items_sha256"]:
        raise ValueError(f"{document['items_file']} does not match the hash recorded in {output}")
    return json.loads(gzip.decompress(raw))["tables"]


def _spearman_resampled(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    rx, ry = rankdata(x, axis=1), rankdata(y, axis=1)
    rx -= rx.mean(1, keepdims=True); ry -= ry.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):    # constant resamples give NaN; dropped by the caller
        return (rx * ry).sum(1) / np.sqrt((rx * rx).sum(1) * (ry * ry).sum(1))


def paired_bootstrap_metric(kind: str, columns_a: Sequence[Sequence[Any]], columns_b: Sequence[Sequence[Any]], *,
                            resamples: int = 2000, seed: int = 0) -> dict[str, float | int | None]:
    """Percentile 95% interval of `metric(A) − metric(B)`, resampling items jointly for both runs.

    Mean-type metrics use `statistics.paired_bootstrap_ci`; correlation, F1 and AUC metrics are
    recomputed on every resample.
    """
    n = len(columns_a[0])
    if kind == "mean":
        return paired_bootstrap_ci(np.asarray(columns_a[0], dtype=float), np.asarray(columns_b[0], dtype=float),
                                   resamples=resamples, seed=seed)
    point_a, point_b = compute_metric(kind, columns_a, range(n)), compute_metric(kind, columns_b, range(n))
    if point_a is None or point_b is None:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": n}
    index = np.random.default_rng(seed).integers(0, n, size=(resamples, n))
    if kind == "spearman":
        gold_a, gold_b = np.asarray(columns_a[1], dtype=float), np.asarray(columns_b[1], dtype=float)
        diffs = (_spearman_resampled(np.asarray(columns_a[0], dtype=float)[index], gold_a[index])
                 - _spearman_resampled(np.asarray(columns_b[0], dtype=float)[index], gold_b[index]))
    else:
        diffs = []
        for rows in index:
            a, b = compute_metric(kind, columns_a, rows), compute_metric(kind, columns_b, rows)
            if a is not None and b is not None:
                diffs.append(a - b)
        diffs = np.asarray(diffs)
    diffs = diffs[np.isfinite(diffs)]
    return {"mean": float(point_a - point_b), "ci_low": float(np.quantile(diffs, 0.025)) if diffs.size else None,
            "ci_high": float(np.quantile(diffs, 0.975)) if diffs.size else None, "n": n}


def compare_outputs(output_a: Path, output_b: Path, *, resamples: int = 2000, seed: int = 0,
                    allow_different_settings: bool = False) -> dict[str, Any]:
    """Paired comparison of two probe outputs over identical items: A − B per metric and subset.

    Both outputs must come from the same probe settings (items, splits, caps, batch size) and have
    the same item ids. Subsets use A's link statuses (B's when A has none, e.g. an untouched host vs
    a channel run); if both have statuses and they differ, only the full item set is compared.
    """
    settings_a, settings_b = (json.loads(Path(o).read_text()).get("settings") for o in (output_a, output_b))
    if settings_a != settings_b and not allow_different_settings:
        raise ValueError("the two outputs were produced with different probe settings")
    a, b = load_items(output_a), load_items(output_b)
    comparison: dict[str, Any] = {"a": str(output_a), "b": str(output_b), "resamples": resamples, "seed": seed, "tables": {}}
    for name in sorted(set(a) & set(b) & set(METRICS)):
        ta, tb = a[name], b[name]
        if ta["id"] != tb["id"]:
            raise ValueError(f"{name}: the two outputs do not have the same items")
        sa, sb = ta.get("status"), tb.get("status")
        has_a, has_b = any(s is not None for s in sa or []), any(s is not None for s in sb or [])
        entry: dict[str, Any] = {}
        if has_a and has_b and sa != sb:
            status, entry["subsets_from"] = None, None
            entry["warning"] = "link statuses differ between the runs; subsets not compared"
        else:
            status, entry["subsets_from"] = (sa, "a") if has_a else (sb, "b") if has_b else (None, None)
        for metric, (kind, columns, mask) in METRICS[name].items():
            if mask is not None and ta[mask] != tb[mask]:
                raise ValueError(f"{name}: the probe splits ({mask}) differ")
            per_subset = {}
            for subset, members in [("all", None), *SUBSETS.items()] if status is not None else [("all", None)]:
                rows = [i for i in range(len(ta["id"])) if (mask is None or ta[mask][i])
                        and (members is None or status[i] in members)]
                if not rows:
                    continue
                cols_a = [[ta[c][i] for i in rows] for c in columns]
                cols_b = [[tb[c][i] for i in rows] for c in columns]
                ci = paired_bootstrap_metric(kind, cols_a, cols_b, resamples=resamples, seed=seed)
                per_subset[subset] = {"a": compute_metric(kind, cols_a, range(len(rows))),
                                      "b": compute_metric(kind, cols_b, range(len(rows))),
                                      "difference": ci["mean"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "n": len(rows)}
            entry[metric] = per_subset
        comparison["tables"][name] = entry
    return comparison


# -- CLI ----------------------------------------------------------------------------------------

def host_adapter(name: str, device: torch.device, *, batch_size: int = 32, layer: int = -1) -> ModelAdapter:
    """An untouched pretrained host, loaded as in B8 (bf16 weights, plain `ModelAdapter`)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(name, local_files_only=True, torch_dtype=torch.bfloat16).to(device).eval()
    return ModelAdapter(model, tokenizer, device, layer=layer, batch_size=batch_size)


def data_hashes(root: Path, probes: Sequence[str], settings: ProbeSettings) -> dict[str, str]:
    files = {"lambada": ["lambada/data/lambada_test_en.jsonl"],
             "wic": ["wic/train/train.data.txt", "wic/train/train.gold.txt", "wic/dev/dev.data.txt", "wic/dev/dev.gold.txt"],
             "card660": ["card660.tsv"], "rare_words": ["rw/rw/rw.txt"],
             "wsd": ["wsd/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.data.xml",
                     "wsd/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor.gold.key.txt",
                     f"wsd/WSD_Evaluation_Framework/Evaluation_Datasets/{settings.wsd_dataset}/{settings.wsd_dataset}.data.xml",
                     f"wsd/WSD_Evaluation_Framework/Evaluation_Datasets/{settings.wsd_dataset}/{settings.wsd_dataset}.gold.key.txt"],
             "bless": ["bless/bless-gems/BLESS.txt"],
             "hyperlex": ["hyperlex/hyperlex-all.txt"] + [f"hyperlex/splits/{settings.hyperlex_split}/hyperlex_{p}_all_{settings.hyperlex_split}.txt"
                                                         for p in ("training", "dev", "test")]}
    return {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for p in probes for f in files[p]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path, help="trained run folder (final.pt + resolved_config.yaml)")
    source.add_argument("--host", help="untouched Hugging Face host (plain adapter, as B8)")
    source.add_argument("--compare", nargs=2, type=Path, metavar=("A", "B"), help="paired comparison of two probe outputs")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", default="final.pt")
    parser.add_argument("--probes", default="all", help="all or a comma-separated subset of " + ",".join(PROBES))
    parser.add_argument("--fast", action="store_true", help="screening subset (ProbeSettings.fast)")
    parser.add_argument("--probes-root", type=Path, default=Path("~/data/vsa-llm/probes").expanduser())
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--ontology", type=Path); parser.add_argument("--alias-table", type=Path)
    parser.add_argument("--holdout-names", type=Path)
    parser.add_argument("--resamples", type=int, default=2000); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-different-settings", action="store_true", help="--compare outputs of different settings")
    parser.add_argument("--quantize", choices=sorted(QUANTIZE_BITS), default=None,
                        help="post-training weight-only quantization of the run as in e4_quant (int4 needs CUDA)")
    parser.add_argument("--quantize-channel", action="store_true", help="with --quantize: variant B (channel quantized too)")
    parser.add_argument("--group-size", default="auto", help="INT4 group size, or auto (as e4_quant)")
    args = parser.parse_args(argv)
    if (args.quantize or args.quantize_channel) and not args.run:
        parser.error("--quantize applies to --run")
    if args.quantize_channel and not args.quantize:
        parser.error("--quantize-channel needs --quantize")
    if args.compare:
        comparison = compare_outputs(*args.compare, resamples=args.resamples, seed=args.seed,
                                     allow_different_settings=args.allow_different_settings)
        text = json.dumps(_native(comparison), indent=2) + "\n"
        if args.output:
            args.output.write_text(text)
        print(text)
        return
    if args.output is None:
        parser.error("--output is required")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")
    from ..provenance import build_manifest
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    probes = PROBES if args.probes == "all" else tuple(p.strip() for p in args.probes.split(",") if p.strip())
    settings = ProbeSettings.fast() if args.fast else ProbeSettings()
    started = time.monotonic()
    if args.run:
        adapter = load_run(args.run, checkpoint=args.checkpoint, device=device, ontology_path=args.ontology,
                           alias_table=args.alias_table, holdout_names=args.holdout_names, batch_size=args.batch_size,
                           layer=args.layer, quantize=args.quantize, quantize_channel=args.quantize_channel,
                           group_size=args.group_size)
        model_info = adapter.info
    else:
        adapter = host_adapter(args.host, device, batch_size=args.batch_size, layer=args.layer)
        model_info = {"host": args.host, "weights": "bfloat16", "adapter": "ModelAdapter"}
    loaded = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    results, tables = run_channel_probes(adapter, args.probes_root, probes=probes, settings=settings)
    header = {"model": model_info, "probes_run": list(probes), "fast": args.fast,
              "settings": {**asdict(settings), "batch_size": args.batch_size, "max_length": adapter.max_length,
                           "layer": args.layer, "word_prefix": WORD_PREFIX},
              "probes_root": str(args.probes_root), "data_sha256": data_hashes(args.probes_root, probes, settings),
              "timing": {"load_seconds": round(loaded - started, 2), "probe_seconds": round(time.monotonic() - loaded, 2),
                         "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None},
              "provenance": build_manifest({"settings": asdict(settings), "probes": list(probes)}, device=device)}
    document = write_outputs(args.output, results, tables, header)
    print(json.dumps(document["summary"], indent=2))


if __name__ == "__main__":
    main()
