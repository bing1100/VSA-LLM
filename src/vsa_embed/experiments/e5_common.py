"""Shared pieces of the E5 experiments (WP-E5; experiments.md E5.1–E5.5).

- `open_run`: a trained run (`final.pt` + resolved config) as a `channel_probes.ChannelModelAdapter`
  (full evaluation linker, held-out aliases included) plus its config, ontology and alias table.
- `entry_rows`: context-free channel rows (model width, after projection) of any entries.
- `override_rows`: within the block, the rows of listed entries come from a given table (a zero row
  is exactly "no injection": the channel adds `gate · row`). Used by the zero-shot baselines.
- `edge_weight_hook` / `ablation_transform`: within the block, the composer's edge weights pass
  through a function — zeroing the top-, bottom- or random-`k` edges of each occurrence (E5.1), or
  recording them (E5.2, E5.3). A removed edge is erased from the weighted sum with the other
  weights unchanged; because the bundle is normalized, this equals renormalizing the softmax over
  the remaining edges. Removing every edge gives a zero row, i.e. removes the channel.
- readable names for ontology atoms and relations (judge items), canonical surface forms per entry,
  document parity of eval-corpus positions (example contexts vs test contexts in E5.4).

Nothing here changes the behaviour of existing code outside the `with` blocks.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np
import torch
import yaml
from torch import Tensor

from ..compose import FrameComposer
from ..data.corpus import TokenCorpus
from ..evaluation import channel_probes as cp
from ..provenance import prepare_output_dir, write_run_metadata
from ..span_channel import AliasTable, SpanChannel

NAME = re.compile(r"(?:^|-)(?P<size>[^-]+)-(?P<condition>[^-]+)-s(?P<seed>\d+)$")   # as e4_report
STATUSES = ("heldout", "rare", "mid", "frequent")
RELATION_PHRASES = {
    "hypernym": "is a kind of", "instance_hypernym": "is an instance of", "part_meronym": "has part",
    "member_meronym": "has member", "substance_meronym": "is made of", "part_holonym": "is part of",
    "member_holonym": "is a member of", "substance_holonym": "is a substance in", "attribute": "has attribute",
    "similar_to": "is similar to", "topic_domain": "belongs to the domain", "entailment": "entails", "cause": "causes",
    "antonym": "is the opposite of", "lexname": "has category", "pos": "is a",
    "kind": "is a", "belongs_to": "belongs to", "returns": "returns", "takes": "takes", "raises": "raises",
    "calls": "calls", "inherits": "inherits from", "category": "is used for",
}
POS_WORDS = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}


# -- runs ---------------------------------------------------------------------------------------

@dataclass
class E5Run:
    """A trained run opened for E5: the probe adapter plus config, ontology and alias table."""

    path: Path
    config: dict[str, Any]
    adapter: cp.ChannelModelAdapter
    ontology: dict[str, Any] | None
    table: AliasTable | None
    checkpoint: str = "final.pt"

    @property
    def model(self):
        return self.adapter.model

    @property
    def channel(self) -> SpanChannel | None:
        return self.adapter.model.channel

    @property
    def composer(self) -> FrameComposer | None:
        channel = self.channel
        return None if channel is None else channel.composer

    @property
    def mode(self) -> str:
        return "none" if self.channel is None else self.channel.mode

    @property
    def device(self) -> torch.device:
        return self.adapter.device

    @property
    def tokenizer(self):
        return self.adapter.tokenizer

    @property
    def min_subtokens(self) -> int:
        return int(self.config["data"]["min_subtokens"])

    @property
    def condition(self) -> str:
        match = NAME.search(str(self.config.get("experiment") or "")) or NAME.search(self.path.name)
        return match["condition"] if match else self.path.name

    @property
    def seed(self) -> int:
        match = NAME.search(str(self.config.get("experiment") or "")) or NAME.search(self.path.name)
        return int(match["seed"]) if match else int(self.config.get("seed", 0))

    @property
    def size(self) -> str:
        model = self.config["model"]
        return f"{model['pretrained']}/{model.get('host_mode', 'train')}" if model.get("pretrained") else str(model["size"])

    def describe(self) -> dict[str, Any]:
        """Source-run record for manifests and summaries."""
        manifest = self.path / "manifest.json"
        git_sha = json.loads(manifest.read_text()).get("git_sha") if manifest.exists() else None
        return {"run": str(self.path), "checkpoint": self.checkpoint, "condition": self.condition, "seed": self.seed,
                "size": self.size, "channel_mode": self.mode, "channel": self.config.get("channel"),
                "experiment": self.config.get("experiment"), "run_git_sha": git_sha,
                "ontology": self.config["data"].get("ontology"), "eval_corpus": self.config["data"].get("eval"),
                "min_subtokens": self.min_subtokens, "tokenizer": self.adapter.info.get("tokenizer"),
                **({"quantization": self.adapter.info["quantization"]} if self.adapter.info.get("quantization") else {})}


def run_config(run_dir: Path, checkpoint: str = "final.pt") -> dict[str, Any]:
    """The run's resolved config (`resolved_config.yaml`, else the one stored in the checkpoint)."""
    from ..training.lm import resolve_config
    path = Path(run_dir) / "resolved_config.yaml"
    if path.exists():
        return resolve_config(yaml.safe_load(path.read_text()))
    return resolve_config(torch.load(Path(run_dir) / checkpoint, weights_only=False, map_location="cpu")["config"])


def open_run(run_dir: Path, *, checkpoint: str = "final.pt", device: torch.device | str | None = None,
             batch_size: int = 32, max_length: int = 256, ontology_path: Path | None = None,
             alias_table: Path | None = None, holdout_names: Path | None = None, quantize: str | None = None,
             quantize_channel: bool = False, group_size: str | int = "auto") -> E5Run:
    """`quantize` ("int8"/"int4", `quantize_channel` = variant B) evaluates the run after the
    post-training quantization of `e4_quant` (`channel_probes.quantize_model`)."""
    run_dir = Path(run_dir)
    adapter = cp.load_run(run_dir, checkpoint=checkpoint, device=device, ontology_path=ontology_path,
                          alias_table=alias_table, holdout_names=holdout_names, batch_size=batch_size,
                          max_length=max_length, quantize=quantize, quantize_channel=quantize_channel,
                          group_size=group_size)
    config = run_config(run_dir, checkpoint)
    source = ontology_path or config["data"].get("ontology")
    ontology = torch.load(source, weights_only=False) if source else None
    table = adapter.linker.table if adapter.linker is not None else None
    return E5Run(run_dir, config, adapter, ontology, table, checkpoint)


# -- rows and overrides ----------------------------------------------------------------------------

@torch.no_grad()
def entry_rows(channel: SpanChannel, entries: Tensor | Sequence[int], *, chunk: int = 8192) -> Tensor:
    """Context-free rows (model width, float32, on the CPU) the channel would inject for `entries`
    (free-table fallback included). Hashed memories (C1h) have no per-entry row."""
    if channel.mode == "hashed":
        raise ValueError("a hashed span memory has no per-entry rows")
    entries = torch.as_tensor(entries, dtype=torch.long)
    device = next(channel.parameters()).device
    parts = [channel.rows({"entry": part.to(device)}).float().cpu() for part in entries.split(chunk)]
    width = channel.gate.in_features // 2
    return torch.cat(parts) if parts else torch.zeros(0, width)


@contextlib.contextmanager
def override_rows(channel: SpanChannel, rows: dict[int, Tensor] | None) -> Iterator[None]:
    """Within the block, entries in `rows` get the given vectors (model width) instead of the
    channel's own rows; every other entry is unchanged. A zero vector is no injection."""
    if not rows:
        yield
        return
    ids = sorted(int(e) for e in rows)
    id_tensor = torch.tensor(ids, dtype=torch.long)
    table = torch.stack([torch.as_tensor(rows[e], dtype=torch.float32) for e in ids])
    original = channel.rows

    def patched(spans: dict[str, Tensor], input_ids: Tensor | None = None, context: Tensor | None = None) -> Tensor:
        entries = spans["entry"]
        device = entries.device
        hit = torch.isin(entries, id_tensor.to(device))
        if bool(hit.all()):
            out = torch.zeros(entries.numel(), table.shape[1], device=device, dtype=table.dtype)
        else:
            keep = ~hit
            subset = {k: v[keep] for k, v in spans.items()}
            base = original(subset, input_ids, None if context is None else context[keep])
            out = base.new_zeros(entries.numel(), base.shape[1])
            out[keep] = base
        if bool(hit.any()):
            position = torch.searchsorted(id_tensor.to(device), entries[hit])
            out[hit] = table.to(device, out.dtype)[position]
        return out

    channel.rows = patched
    try:
        yield
    finally:
        del channel.rows


# -- edge weights -------------------------------------------------------------------------------------

@contextlib.contextmanager
def edge_weight_hook(composer: FrameComposer, transform: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]) -> Iterator[None]:
    """Within the block, `composer.edge_weights` returns `transform(weights, edge_index, segments, concept_ids)`."""
    original = composer.edge_weights

    def patched(edge_index: Tensor, segments: Tensor, concept_ids: Tensor, bound: Tensor, context: Tensor | None) -> Tensor:
        return transform(original(edge_index, segments, concept_ids, bound, context), edge_index, segments, concept_ids)

    composer.edge_weights = patched
    try:
        yield
    finally:
        del composer.edge_weights


def segment_ranks(keys: Tensor, segments: Tensor, count: int, *, descending: bool, tiebreak: Tensor) -> Tensor:
    """Rank (0 = first) of each element within its segment by `keys`, ties broken by `tiebreak`
    (ascending), so equal keys (a uniform-weight composer) are ordered at random when `tiebreak` is."""
    order = torch.argsort(tiebreak, stable=True)
    order = order[torch.argsort(keys[order], descending=descending, stable=True)]
    order = order[torch.argsort(segments[order], stable=True)]
    sizes = torch.bincount(segments, minlength=count)
    starts = torch.cumsum(sizes, 0) - sizes
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(order.numel(), device=order.device) - starts[segments[order]]
    return ranks


@dataclass(frozen=True)
class EdgePolicy:
    """`full`; `remove_all`; `remove_<order><k>` / `keep_<order><k>` with order ∈ {top, bottom, random}
    (draw `d` of a random policy: `remove_random2_d1`)."""

    action: str = "full"
    order: str = "top"
    k: int = 0
    draw: int = 0

    @property
    def name(self) -> str:
        if self.action in {"full", "remove_all"}:
            return self.action
        return f"{self.action}_{self.order}{self.k}" + (f"_d{self.draw}" if self.order == "random" else "")

    @property
    def family(self) -> str:
        """Name without the random draw (draws are averaged)."""
        return self.name.split("_d")[0] if self.order == "random" else self.name

    @classmethod
    def parse(cls, name: str) -> "EdgePolicy":
        if name in {"full", "remove_all"}:
            return cls(name)
        match = re.fullmatch(r"(remove|keep)_(top|bottom|random)(\d+)(?:_d(\d+))?", name)
        if not match:
            raise ValueError(f"unknown edge policy {name!r}")
        return cls(match[1], match[2], int(match[3]), int(match[4] or 0))


def ablation_transform(policy: EdgePolicy, degrees: Tensor, *, seed: int) -> Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]:
    """Edge-weight transform for `policy`. Only occurrences with more than `k` edges are touched.
    Random choices and tie-breaks come from a generator seeded once per block (deterministic for a
    fixed sequence of compositions)."""
    generator = torch.Generator().manual_seed(int(seed))

    def transform(weights: Tensor, edge_index: Tensor, segments: Tensor, concept_ids: Tensor) -> Tensor:
        if policy.action == "full":
            return weights
        if policy.action == "remove_all":
            return torch.zeros_like(weights)
        count = concept_ids.numel()
        degree = degrees.to(concept_ids.device)[concept_ids][segments]
        eligible = degree > policy.k
        tiebreak = torch.rand(weights.numel(), generator=generator).to(weights.device)
        if policy.order == "random":
            ranks = segment_ranks(tiebreak, segments, count, descending=False, tiebreak=tiebreak)
        else:
            ranks = segment_ranks(weights.detach().float(), segments, count, descending=policy.order == "top",
                                  tiebreak=tiebreak)
        chosen = (ranks < policy.k) & eligible
        drop = chosen if policy.action == "remove" else (~chosen & eligible)
        return weights.masked_fill(drop, 0.0)

    return transform


@torch.no_grad()
def span_contexts(model, ids: Tensor, spans: dict[str, Tensor]) -> Tensor | None:
    """The P1 context query at each span's injection position, exactly as `ChannelLM.embed` builds it."""
    if getattr(model, "context", None) is None or spans["entry"].numel() == 0:
        return None
    embeddings = model.model.get_input_embeddings()(ids)
    pooled = model.context(embeddings)
    return pooled[spans["batch"].to(ids.device), spans["inject"].to(ids.device)]


@torch.no_grad()
def occurrence_weights(composer: FrameComposer, entries: Tensor, context: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
    """Per occurrence: (edge weights, schedule edge ids, occurrence index of each edge)."""
    _, weights, edge_index = composer.compose(entries, context, return_weights=True)
    degrees = composer.schedule.degrees[entries]
    segments = torch.repeat_interleave(torch.arange(entries.numel(), device=entries.device), degrees)
    return weights.float(), edge_index, segments


@torch.no_grad()
def forward_tokens(run: "E5Run", ids: Tensor, spans: dict[str, Tensor] | None) -> tuple[Tensor, Tensor]:
    """Per-target-token loss (batch × (T − 1), float32, CPU) and final hidden states (on the device),
    computed as the trainer's evaluation does (bf16 autocast on CUDA)."""
    device = run.device
    ids_d = ids.to(device)
    spans_d = None if spans is None else {k: v.to(device) for k, v in spans.items()}
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = run.model(ids_d, spans=spans_d, labels=ids_d, reduction="none")
    return out["loss"].float().cpu(), out["hidden"]


# -- names ----------------------------------------------------------------------------------------------

def readable_atomic(name: str) -> str:
    """`synset:city.n.01` → `city`; `lexname:noun.animal` → `animal`; `pos:n` → `noun`; `type:int` → `int`."""
    kind, _, value = str(name).partition(":")
    if not value:
        return str(name).replace("_", " ")
    if kind == "synset":
        return value.rsplit(".", 2)[0].replace("_", " ")
    if kind == "lexname":
        return value.split(".", 1)[-1].replace("_", " ")
    if kind == "pos":
        return POS_WORDS.get(value, value)
    return value


def edge_text(relation: str, atomic: str) -> str:
    return f"{RELATION_PHRASES.get(relation, relation.replace('_', ' '))} {readable_atomic(atomic)}"


def entry_aliases(table: AliasTable) -> dict[int, list[str]]:
    aliases: dict[int, list[str]] = {}
    for alias, entry in table.alias_to_entry.items():
        aliases.setdefault(entry, []).append(alias)
    return aliases


def canonical_surfaces(table: AliasTable, tokenizer: Any, min_subtokens: int,
                       entries: Iterable[int] | None = None) -> dict[int, dict[str, Any]]:
    """Per entry, the surface form shown in prompts and judge items: the shortest alias (then
    alphabetical) with ≥ `min_subtokens` subtokens mid-sentence (so the linker injects the entry);
    if none qualifies, the alias with the most subtokens, marked `linkable: False`."""
    aliases = entry_aliases(table)
    wanted = sorted(aliases) if entries is None else sorted({int(e) for e in entries if int(e) in aliases})
    flat = [(e, a) for e in wanted for a in aliases[e]]
    lengths: list[int] = []
    for start in range(0, len(flat), 4096):
        chunk = [" " + a for _, a in flat[start:start + 4096]]
        lengths += [len(ids) for ids in tokenizer(chunk, add_special_tokens=False)["input_ids"]] if chunk else []
    by_entry: dict[int, list[tuple[str, int]]] = {}
    for (entry, alias), n in zip(flat, lengths):
        by_entry.setdefault(entry, []).append((alias, n))
    out = {}
    for entry, options in by_entry.items():
        good = sorted((len(a), a, n) for a, n in options if n >= min_subtokens)
        if good:
            out[entry] = {"surface": good[0][1], "subtokens": good[0][2], "linkable": True}
        else:
            alias, n = max(options, key=lambda x: (x[1], -len(x[0]), x[0]))
            out[entry] = {"surface": alias, "subtokens": n, "linkable": False}
    return out


# -- corpus helpers -------------------------------------------------------------------------------------------

def eos_positions(corpus: TokenCorpus) -> np.ndarray:
    """Positions of the EOS tokens that separate a corpus's documents."""
    return np.flatnonzero(np.asarray(corpus.tokens) == int(corpus.manifest.get("eos_id", 50256)))


def document_index(corpus: TokenCorpus, positions: np.ndarray | Sequence[int], boundaries: np.ndarray | None = None) -> np.ndarray:
    """Document number of each token position (documents are separated by the corpus's EOS id)."""
    boundaries = eos_positions(corpus) if boundaries is None else boundaries
    return np.searchsorted(boundaries, np.asarray(positions), side="left")


def status_of(entry: int, heldout: set[int] | frozenset[int], frequency: np.ndarray | None) -> str:
    return cp.entry_status([int(entry)], heldout, frequency)


# -- run folders --------------------------------------------------------------------------------------------

RESULT_FILES = ("resolved_config.yaml", "manifest.json", "summary.json", "report.md", "predictions.jsonl")


def clear_output(output: Path) -> list[str]:
    """Remove a previous (possibly interrupted) evaluation's result files from `output` so that it
    can be written again (opt-in `--overwrite`); any other file in the folder is left and still
    makes `start_output` refuse it."""
    removed = []
    for name in RESULT_FILES:
        path = Path(output) / name
        if path.is_file():
            path.unlink(); removed.append(name)
    return removed


def start_output(output: Path, config: dict[str, Any]) -> dict[str, Any]:
    git_at_start = prepare_output_dir(Path(output))
    (Path(output) / "resolved_config.yaml").write_text(yaml.safe_dump(json_ready(config), sort_keys=False))
    return git_at_start


def finish_output(output: Path, config: dict[str, Any], *, git_at_start: dict[str, Any] | None,
                  device: torch.device | str | None, **extra: Any) -> None:
    write_run_metadata(Path(output), json_ready(config), git_at_start=git_at_start, device=device, **json_ready(extra))


def json_ready(value: Any) -> Any:
    """JSON-safe copy (numpy scalars and arrays, tuples, Paths, non-finite floats → None)."""
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_ready(v) for v in (sorted(value) if isinstance(value, (set, frozenset)) else value)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return json_ready(value.tolist())
    return cp._native(value)


def write_json(path: Path, value: Any) -> None:
    Path(path).write_text(json.dumps(json_ready(value), indent=2) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def fmt(value: float | None, digits: int = 4, *, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def fmt_ci(entry: dict[str, Any] | None, digits: int = 4) -> str:
    if not entry or entry.get("mean") is None:
        return "—"
    low, high = entry.get("ci_low"), entry.get("ci_high")
    interval = f" [{fmt(low, digits, signed=True)}, {fmt(high, digits, signed=True)}]" if low is not None else ""
    return f"{fmt(entry['mean'], digits, signed=True)}{interval}"
