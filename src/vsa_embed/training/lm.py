"""From-scratch causal-LM training with the span channel (B7), and stratified evaluation.

One run = one condition × seed. Conditions differ only in the channel; the batches are a pure
function of `(data.seed, step)`, so runs are paired. The run folder holds `resolved_config.yaml`,
`manifest.json`, `metrics.jsonl` (training loss and stratified evaluations at log-spaced token
counts), `checkpoint.pt` (latest, rewritten every `checkpoint_minutes`) and `final.pt`. With
`eval.save_window_losses: true` (opt-in) every evaluation also writes per-window, per-stratum loss
sums and target counts to `eval_windows.npz` (`strata`, `starts`, `sum_<tokens>` and
`count_<tokens>`, strata × windows); the windows are identical across conditions, so condition
differences are paired by window. Evaluation rows carry the training tokens under `tokens` and the
stratum's target count under `stratum_tokens` (rows written before that key existed have the count
under `tokens`). `--resume` continues from `checkpoint.pt`; a run that died before its first
checkpoint starts over, its partial files kept as `<stem>.aborted-<n><suffix>`.

Evaluation strata (per target token `j`, predicted from position `j − 1`):
`all`; `unlinked` (not inside or within 8 tokens after a linked span); `inside` (subtokens 2..ℓ of
a span); `after` (the 8 tokens after a span) and its splits by entry status — `after_heldout`
(held-out concepts, never linked in training), `after_rare` / `after_mid` / `after_frequent`
(training frequency 1–9 / 10–99 / ≥ 100; `after_rare` also holds entries never linked in training, which
`after_unseen` (frequency 0, not held out) and `after_rare_seen` (1–9) separate) — and by span length
(`after_len1`, `after_len2`, `after_len3plus`).

Continued pretraining of pretrained hosts (`model.pretrained`, E4.6) adds optional keys, read with
defaults so that from-scratch configs resolve exactly as before:

- `train.eval_only` (false): evaluate without training and log the result at every evaluation point
  (C0' on a frozen host, which has no trainable parameters).
- `train.host_lr` (null): learning rate of host parameters (LoRA adapters, or a fully trained host)
  when it should differ from `train.lr` (which then applies to the channel only).
- `train.save_trainable_only` (false): checkpoints and `final.pt` hold trainable parameters plus the
  channel's state, not the frozen host weights (reloaded from the hub id); see `load_final`.

For pretrained hosts the corpora are checked against the host: token ids must fit its embedding
table, and a corpus that records its tokenizer fingerprint (host corpora do) must match the host's.

Two further opt-in keys (E7 self-authoring; absent keys change nothing):

- `train.init_from` (null): a state file `{"model": state, "trainable_only": bool}` (e.g. a previous
  run's `final.pt`, or one prepared from it with a grown dictionary) loaded before training starts;
  a resumed run takes its checkpoint instead.
- `eval.reference_strata` (null): an `.npz` written by `save_reference_strata` with boolean target
  masks over the evaluation windows; each mask becomes an extra stratum (`ref_<name>`) in
  `metrics.jsonl` and `eval_windows.npz`. The masks are fixed by the caller, so they are identical
  across conditions whose linkers differ (paired comparisons on the same target tokens).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import GPT2Config, GPT2LMHeadModel

from ..compose import FrameComposer, FrameSchedule
from ..context import CausalLocalContext
from ..data.corpus import TokenCorpus, collate_windows, eval_windows, sample_batch, tokenizer_fingerprint
from ..developmental import DevelopmentalConfig, DevelopmentalDictionary
from ..integrations.transformers import ChannelLM
from ..provenance import git_state, prepare_output_dir, write_run_metadata
from ..span_channel import SpanChannel

MODEL_SIZES = {
    "tiny": dict(n_layer=2, n_embd=64, n_head=2),
    "50M": dict(n_layer=8, n_embd=512, n_head=8),
    "125M": dict(n_layer=12, n_embd=768, n_head=12),
    "350M": dict(n_layer=24, n_embd=1024, n_head=16),
}
AFTER_WINDOW = 8


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    c = copy.deepcopy(config)
    c.setdefault("seed", 0)
    model = c.setdefault("model", {})
    model.setdefault("size", "50M"); model.setdefault("vocab_size", 50257); model.setdefault("seq_len", 1024)
    model.setdefault("gradient_checkpointing", False)
    model.setdefault("pretrained", None)        # HF id of a pretrained host (continued pretraining)
    model.setdefault("host_mode", "train")      # train | frozen | lora (pretrained hosts)
    model.setdefault("lora_rank", 16)
    train = c.setdefault("train", {})
    for key, value in {"micro_batch": 16, "grad_accum": 32, "total_tokens": 300_000_000, "lr": 1e-3,
                       "min_lr_ratio": 0.1, "warmup_tokens": 10_000_000, "weight_decay": 0.1,
                       "beta1": 0.9, "beta2": 0.95, "grad_clip": 1.0, "log_every": 20,
                       "checkpoint_minutes": 30, "semantic_weight": 0.0, "delta_weight": 1e-3}.items():
        train.setdefault(key, value)
    data = c.setdefault("data", {})
    data.setdefault("seed", 1234); data.setdefault("min_subtokens", 2)
    evaluation = c.setdefault("eval", {})
    evaluation.setdefault("windows", 256); evaluation.setdefault("batch", 16)
    evaluation.setdefault("first_tokens", 10_000_000)
    channel = c.setdefault("channel", {})
    channel.setdefault("mode", "none")
    for key, value in {"operator": "hrr", "composition": "attentive", "dimension": 256, "key_dimension": 32,
                       "context_window": 0, "gate_bias": -2.0, "hashed_buckets": 0, "developmental": None,
                       "concept_factor": "induced", "free_dimension": 0, "frames": "ontology"}.items():
        channel.setdefault(key, value)
    c.setdefault("device", "cuda")
    return c


def eval_token_schedule(first: int, total: int) -> list[int]:
    """Log-spaced evaluation points: first, 2·first, 4·first, … and the end."""
    points, t = [], first
    while t < total:
        points.append(t); t *= 2
    return points + [total]


def build_model(config: dict[str, Any]):
    if config["model"]["pretrained"]:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(config["model"]["pretrained"], local_files_only=True,
                                                     torch_dtype=torch.float32, attn_implementation="sdpa")
        if config["model"]["gradient_checkpointing"]:
            model.gradient_checkpointing_enable(); model.config.use_cache = False
        return model
    size = config["model"]["size"]
    gpt2 = GPT2Config(vocab_size=config["model"]["vocab_size"], n_positions=config["model"]["seq_len"], **MODEL_SIZES[size])
    gpt2._attn_implementation = "sdpa"
    model = GPT2LMHeadModel(gpt2)
    if config["model"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(); model.config.use_cache = False
    return model


def build_channel(config: dict[str, Any], ontology: dict[str, Any] | None, width: int) -> tuple[SpanChannel | None, CausalLocalContext | None]:
    settings = config["channel"]
    mode = settings["mode"]
    if mode == "none":
        return None, None
    if ontology is None:
        raise ValueError("channel conditions need data.ontology")
    entries = int(ontology["entry_count"])
    if mode != "compose":
        return SpanChannel(None, width, entry_count=entries, mode=mode, hashed_buckets=int(settings["hashed_buckets"]),
                           gate_bias=float(settings["gate_bias"]), free_dimension=int(settings["free_dimension"]),
                           semantic_dimension=width if config["train"]["semantic_weight"] else 0), None
    schedule = frame_variant(FrameSchedule(ontology["offsets"], ontology["relations"], ontology["fillers"]),
                             settings["frames"], int(ontology["relation_count"]), seed=int(config["seed"]))
    atomic_count = int(ontology["relation_count"]) if settings["frames"] == "relation_only" else int(ontology["atomic_count"])
    context_window = int(settings["context_window"])
    composer = FrameComposer(
        schedule, atomic_count, int(ontology["relation_count"]), int(settings["dimension"]),
        operator=settings["operator"], mode=settings["composition"], concept_factor=settings["concept_factor"],
        key_dimension=int(settings["key_dimension"]),
        context_dimension=int(settings["key_dimension"]) if context_window else 0,
    )
    channel = SpanChannel(composer, width, entry_count=entries, gate_bias=float(settings["gate_bias"]),
                          semantic_dimension=width if config["train"]["semantic_weight"] else 0)
    context = CausalLocalContext(width, int(settings["key_dimension"]), window=context_window) if context_window else None
    return channel, context


def frame_variant(schedule: FrameSchedule, variant: str, relation_count: int, *, seed: int) -> FrameSchedule:
    """`ontology` (as is); `relation_only` (C3t: each edge's filler is its relation type, so the
    concept is a bag of relation types — no fillers, and with `untyped`, no binding); `shuffled`
    (C1s: entries receive other entries' frames — matched parameters, wrong structure)."""
    if variant == "ontology":
        return schedule
    if variant == "relation_only":
        return FrameSchedule(schedule.offsets, schedule.relations, schedule.relations.clone())
    if variant == "shuffled":
        from ..real_relations import derange_labels
        count = schedule.concept_count
        order = derange_labels(torch.arange(count), torch.Generator().manual_seed(seed + 4242))
        frames = [list(zip(schedule.relations[schedule.offsets[j]:schedule.offsets[j + 1]].tolist(),
                           schedule.fillers[schedule.offsets[j]:schedule.offsets[j + 1]].tolist())) for j in order.tolist()]
        return FrameSchedule.from_frames(frames)
    raise ValueError("frames must be ontology, relation_only or shuffled")


def _lr(step: int, total_steps: int, warmup_steps: int, peak: float, floor_ratio: float) -> float:
    if step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return peak * (floor_ratio + (1 - floor_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def stratum_masks(ids: torch.Tensor, spans: dict[str, torch.Tensor], frequency: np.ndarray | None,
                  heldout: set[int]) -> dict[str, torch.Tensor]:
    """Boolean masks over target positions `j ∈ [1, T)` (shape batch × (T − 1))."""
    batch, length = ids.shape
    shape = (batch, length - 1)
    inside = torch.zeros(shape, dtype=torch.bool)
    after: dict[str, torch.Tensor] = {name: torch.zeros(shape, dtype=torch.bool) for name in (
        "after", "after_heldout", "after_rare", "after_mid", "after_frequent", "after_len1", "after_len2", "after_len3plus",
        "after_unseen", "after_rare_seen")}
    for b, s, e, entry, n in zip(spans["batch"].tolist(), spans["start"].tolist(), spans["end"].tolist(),
                                 spans["entry"].tolist(), spans["length"].tolist()):
        if e > s:
            inside[b, s:e] = True                       # targets s+1..e sit at positions s..e−1
        lo, hi = e, min(length - 1, e + AFTER_WINDOW)   # targets e+1..e+8 at positions e..e+7
        if lo >= hi:
            continue
        names = ["after", "after_len1" if n == 1 else "after_len2" if n == 2 else "after_len3plus"]
        if entry in heldout:
            names.append("after_heldout")
        elif frequency is not None:
            count = int(frequency[entry])
            names.append("after_rare" if count < 10 else "after_mid" if count < 100 else "after_frequent")
            if count < 10:   # `after_rare` keeps entries never linked in training; these two split it
                names.append("after_unseen" if count == 0 else "after_rare_seen")
        for name in names:
            after[name][b, lo:hi] = True
    masks = {"all": torch.ones(shape, dtype=torch.bool), "inside": inside, **after}
    masks["unlinked"] = ~(inside | after["after"])
    return masks


def save_reference_strata(path: Path, starts: list[int] | np.ndarray, masks: dict[str, np.ndarray], length: int) -> None:
    """Write fixed target masks (each windows × (length − 1), bool) for `eval.reference_strata`."""
    starts = np.asarray(starts, dtype=np.int64)
    arrays: dict[str, np.ndarray] = {"starts": starts, "names": np.asarray(sorted(masks)), "length": np.asarray(length)}
    for name in sorted(masks):
        mask = np.asarray(masks[name], dtype=bool)
        if mask.shape != (starts.size, length - 1):
            raise ValueError(f"mask {name!r} has shape {mask.shape}, expected {(starts.size, length - 1)}")
        arrays[f"mask_{name}"] = np.packbits(mask, axis=None)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_reference_strata(path: Path, starts: list[int], length: int) -> dict[str, np.ndarray]:
    """Masks of `save_reference_strata`, checked against the run's evaluation windows; keys `ref_<name>`."""
    with np.load(path) as data:
        if not np.array_equal(data["starts"], np.asarray(starts, dtype=np.int64)) or int(data["length"]) != length:
            raise ValueError(f"{path} was written for other evaluation windows")
        shape = (len(starts), length - 1)
        return {f"ref_{name}": np.unpackbits(data[f"mask_{name}"], count=shape[0] * shape[1]).astype(bool).reshape(shape)
                for name in data["names"].tolist()}


@torch.no_grad()
def evaluate(model: ChannelLM, corpus: TokenCorpus, starts: list[int], config: dict[str, Any],
             frequency: np.ndarray | None, heldout: set[int], device: torch.device, *,
             window_sink: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] | None = None,
             reference: dict[str, np.ndarray] | None = None) -> dict[str, dict[str, float]]:
    """Stratified loss over the evaluation windows; `window_sink` (if given) collects per-window
    loss sums and target counts per stratum; `reference` adds fixed per-window target masks."""
    model.eval()
    length, batch = config["model"]["seq_len"], config["eval"]["batch"]
    sums: dict[str, float] = {}; counts: dict[str, int] = {}
    for i in range(0, len(starts), batch):
        windows = [corpus.window(s, length, min_subtokens=config["data"]["min_subtokens"]) for s in starts[i:i + batch]]
        ids, spans = collate_windows(windows)
        ids_d = ids.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            per_token = model(ids_d, spans={k: v.to(device) for k, v in spans.items()} if model.channel else None,
                              labels=ids_d, reduction="none")["loss"].float().cpu()
        masks = stratum_masks(ids, spans, frequency, heldout)
        if reference:
            masks.update({name: torch.from_numpy(values[i:i + batch]) for name, values in reference.items()})
        for name, mask in masks.items():
            sums[name] = sums.get(name, 0.0) + float(per_token[mask].sum())
            counts[name] = counts.get(name, 0) + int(mask.sum())
            if window_sink is not None:
                window_sums, window_counts = window_sink.setdefault(name, ([], []))
                window_sums.append(per_token.double().masked_fill(~mask, 0.0).sum(1).numpy())
                window_counts.append(mask.sum(1).numpy().astype(np.int32))
    model.train()
    return {name: {"loss": sums[name] / counts[name] if counts[name] else float("nan"), "tokens": counts[name]}
            for name in sums}


def check_host_corpora(config: dict[str, Any], base: torch.nn.Module, corpora: tuple[TokenCorpus, ...]) -> None:
    """Refuse corpora tokenized for another host: ids must fit the host's embedding table, and a
    recorded tokenizer fingerprint (host corpora record one) must equal the host tokenizer's."""
    rows = base.get_input_embeddings().weight.shape[0]
    fingerprint = None
    for corpus in corpora:
        recorded = corpus.manifest.get("tokenizer_sha256")
        if recorded is not None:
            if fingerprint is None:
                from transformers import AutoTokenizer
                fingerprint = tokenizer_fingerprint(AutoTokenizer.from_pretrained(config["model"]["pretrained"], local_files_only=True))
            if recorded != fingerprint:
                raise ValueError(f"corpus tokenized with {corpus.manifest.get('tokenizer')}, not the tokenizer of "
                                 f"{config['model']['pretrained']}")
        largest = int(corpus.tokens.max()) if len(corpus) else -1
        if largest >= rows:
            raise ValueError(f"corpus token id {largest} ≥ {rows} embedding rows of {config['model']['pretrained']}")


def _host_lr_groups(trainable: list[tuple[str, torch.nn.Parameter]], embedding_names: tuple[str, ...],
                    train_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Parameter groups with host parameters (names under `model.`) at `host_lr` via `lr_scale`."""
    groups = []
    for host in (False, True):
        for decayed in (True, False):
            params = [p for n, p in trainable if n.startswith("model.") == host
                      and (p.ndim >= 2 and not any(e in n for e in embedding_names)) == decayed]
            if params:
                group = {"params": params, "weight_decay": train_cfg["weight_decay"] if decayed else 0.0}
                if host:
                    group["lr_scale"] = float(train_cfg["host_lr"]) / float(train_cfg["lr"])
                groups.append(group)
    return groups


def _evaluate_only(model: ChannelLM, config: dict[str, Any], output_dir: Path, eval_corpus: TokenCorpus,
                   frequency: np.ndarray | None, heldout: set[int], device: torch.device,
                   git_at_start: dict[str, Any] | None) -> dict[str, Any]:
    """C0' on a frozen host: one evaluation, logged at every point of the run's evaluation schedule
    (marked `eval_only`), so its curve lines up with the trained conditions'."""
    seq_len, train_cfg = config["model"]["seq_len"], config["train"]
    tokens_per_step = seq_len * train_cfg["micro_batch"] * train_cfg["grad_accum"]
    total_steps = max(1, train_cfg["total_tokens"] // tokens_per_step)
    schedule = eval_token_schedule(config["eval"]["first_tokens"], total_steps * tokens_per_step)
    starts = eval_windows(eval_corpus, count=config["eval"]["windows"], length=seq_len)
    sink: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] | None = {} if config["eval"].get("save_window_losses") else None
    reference = (load_reference_strata(Path(config["eval"]["reference_strata"]), starts, seq_len)
                 if config["eval"].get("reference_strata") else None)
    results = evaluate(model, eval_corpus, starts, config, frequency, heldout, device, window_sink=sink, reference=reference)
    # Same row format as a trained run's evaluations, at the step where that run evaluates each point.
    rows = [{"type": "eval", "step": -(-tokens // tokens_per_step), "tokens": tokens, "stratum": stratum,
             "loss": value["loss"], "stratum_tokens": value["tokens"], "eval_only": True}
            for tokens in [0, *schedule] for stratum, value in results.items()]
    (output_dir / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    if sink is not None:
        for tokens in [0, *schedule]:
            save_window_losses(output_dir / "eval_windows.npz", tokens, starts, sink)
    torch.save({"model": model_state(model, True), "config": config, "composer_schedule": _schedule_state(model.channel),
                "trainable_only": True, "eval_only": True}, output_dir / "final.pt")
    if not (output_dir / "manifest.json").exists():
        write_run_metadata(output_dir, config, git_at_start=git_at_start, device=device,
                           parameters=sum(p.numel() for p in model.parameters()),
                           channel_parameters=sum(p.numel() for p in model.channel.parameters()) if model.channel else 0,
                           steps=0, tokens_per_step=tokens_per_step, eval_only=True)
    return {"steps": 0, "tokens": 0, "eval_only": True}


def train(config: dict[str, Any], output_dir: Path, *, resume: bool = False) -> dict[str, Any]:
    config = resolve_config(config)
    checkpoint_path = output_dir / "checkpoint.pt"
    eval_only = bool(config["train"].get("eval_only", False))
    if eval_only and resume and (output_dir / "final.pt").exists():
        return {"steps": 0, "tokens": 0, "eval_only": True}
    git_at_start = None
    if resume and not checkpoint_path.exists() and not eval_only and output_dir.exists() and any(output_dir.iterdir()):
        # Died before its first checkpoint: start over, keeping the partial files aside.
        git_at_start = set_aside_partial_run(output_dir)
        (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    # An interrupted evaluation-only run has no checkpoint; it restarts in place.
    elif not (resume and (checkpoint_path.exists() or (eval_only and output_dir.exists()))):
        git_at_start = prepare_output_dir(output_dir)
        (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["seed"])
    corpus = TokenCorpus.open(Path(config["data"]["train"]))
    eval_corpus = TokenCorpus.open(Path(config["data"]["eval"]))
    ontology = torch.load(config["data"]["ontology"], weights_only=False) if config["data"].get("ontology") else None
    heldout = set(ontology["heldout_entries"]) if ontology else set()
    frequency = np.asarray(ontology["train_frequency"]) if ontology else None
    base = build_model(config)
    if config["model"]["pretrained"]:
        check_host_corpora(config, base, (corpus, eval_corpus))
    width = base.get_input_embeddings().weight.shape[1]
    channel, context = build_channel(config, ontology, width)
    if channel is not None and ontology is not None:
        channel.set_unseen(ontology["heldout_entries"])
    model = ChannelLM(base, channel, context=context, host_mode=config["model"]["host_mode"] if config["model"]["pretrained"] else "train",
                      lora_rank=int(config["model"]["lora_rank"])).to(device)
    train_cfg = config["train"]
    trainable_only = bool(train_cfg.get("save_trainable_only", False))
    if train_cfg.get("init_from") and not (resume and checkpoint_path.exists()):
        initial = torch.load(train_cfg["init_from"], weights_only=False, map_location="cpu")
        load_model_state(model, initial["model"], trainable_only=bool(initial.get("trainable_only", False)))
    if eval_only:
        return _evaluate_only(model, config, output_dir, eval_corpus, frequency, heldout, device, git_at_start)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    embedding_names = ("wte", "wpe", "embed_tokens")
    decay = [p for n, p in trainable if p.ndim >= 2 and not any(e in n for e in embedding_names)]
    no_decay = [p for n, p in trainable if not (p.ndim >= 2 and not any(e in n for e in embedding_names))]
    if not decay and not no_decay:
        raise ValueError("no trainable parameters (frozen host without a channel? use train.eval_only)")
    groups = [{"params": decay, "weight_decay": train_cfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}]
    if train_cfg.get("host_lr") is not None:
        groups = _host_lr_groups(trainable, embedding_names, train_cfg)
    optimizer = torch.optim.AdamW(groups, lr=train_cfg["lr"],
                                  betas=(train_cfg["beta1"], train_cfg["beta2"]), fused=device.type == "cuda")
    tracker = None
    if channel is not None and channel.composer is not None and config["channel"]["developmental"]:
        tracker = DevelopmentalDictionary(channel.composer, optimizer, DevelopmentalConfig(**config["channel"]["developmental"]))
    seq_len, micro, accum = config["model"]["seq_len"], train_cfg["micro_batch"], train_cfg["grad_accum"]
    tokens_per_step = seq_len * micro * accum
    total_steps = max(1, train_cfg["total_tokens"] // tokens_per_step)
    warmup_steps = max(1, train_cfg["warmup_tokens"] // tokens_per_step)
    schedule = [t for t in eval_token_schedule(config["eval"]["first_tokens"], total_steps * tokens_per_step)]
    eval_starts = eval_windows(eval_corpus, count=config["eval"]["windows"], length=seq_len)
    entry_mask = None
    if ontology is not None:
        entry_mask = np.ones(int(ontology["entry_count"]), dtype=bool)
        entry_mask[list(heldout)] = False           # held-out concepts are never linked in training
    step, evaluated = 0, set()
    if resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        if tracker is not None and state.get("composer_schedule") is not None:
            _restore_growth(channel, tracker, state, optimizer)
        load_model_state(model, state["model"], trainable_only=bool(state.get("trainable_only", False)))
        optimizer.load_state_dict(state["optimizer"])
        step, evaluated = state["step"], set(state["evaluated"])
        torch.set_rng_state(state["rng_cpu"])
        if device.type == "cuda" and state.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(state["rng_cuda"])
    metrics_path = output_dir / "metrics.jsonl"
    last_checkpoint = time.monotonic()

    def log(row: dict[str, Any]) -> None:
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    save_windows = bool(config["eval"].get("save_window_losses", False))   # opt-in; absent from older configs
    reference = (load_reference_strata(Path(config["eval"]["reference_strata"]), eval_starts, seq_len)
                 if config["eval"].get("reference_strata") else None)

    def run_eval(tokens: int) -> None:
        sink: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] | None = {} if save_windows else None
        results = evaluate(model, eval_corpus, eval_starts, config, frequency, heldout, device, window_sink=sink,
                           reference=reference)
        for stratum, value in results.items():
            # `tokens` = training tokens at this evaluation; `stratum_tokens` = target tokens in the
            # stratum (rows written before this fix carry only the latter, under `tokens`).
            log({"type": "eval", "step": step, "tokens": tokens, "stratum": stratum, "loss": value["loss"],
                 "stratum_tokens": value["tokens"]})
        if sink is not None:
            save_window_losses(output_dir / "eval_windows.npz", tokens, eval_starts, sink)
        evaluated.add(tokens)

    if 0 not in evaluated:
        run_eval(0)
    model.train()
    started = time.monotonic()
    while step < total_steps:
        lr = _lr(step, total_steps, warmup_steps, train_cfg["lr"], train_cfg["min_lr_ratio"])
        for group in optimizer.param_groups:
            group["lr"] = lr * group.get("lr_scale", 1.0)
        if tracker is not None:
            tracker.begin()
        total_loss = 0.0
        for micro_step in range(accum):
            ids, spans = sample_batch(corpus, seed=config["data"]["seed"], step=step, micro_step=micro_step, batch=micro,
                                      length=seq_len, min_subtokens=config["data"]["min_subtokens"], entry_mask=entry_mask)
            ids = ids.to(device)
            spans_d = {k: v.to(device) for k, v in spans.items()} if channel is not None else None
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(ids, spans=spans_d, labels=ids)
                loss = out["loss"]
                if channel is not None and train_cfg["semantic_weight"]:
                    loss = loss + train_cfg["semantic_weight"] * channel.semantic_loss(out["hidden"], spans_d, input_ids=ids)
                if channel is not None and channel.composer is not None:
                    loss = loss + train_cfg["delta_weight"] * channel.composer.delta_penalty()
            if loss.requires_grad:      # a frozen host with no linked span in this micro-batch has nothing to train
                (loss / accum).backward()
            total_loss += float(loss.detach()) / accum
        if tracker is not None:
            tracker.observe()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], train_cfg["grad_clip"])
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        if tracker is not None:
            for card in tracker.grow():
                log({"type": "card", "step": step, **{k: v for k, v in card.items() if k != "direction"}})
        step += 1
        tokens = step * tokens_per_step
        if step % train_cfg["log_every"] == 0 or step == total_steps:
            elapsed = time.monotonic() - started
            log({"type": "train", "step": step, "tokens": tokens, "loss": total_loss, "lr": lr,
                 "tokens_per_s": train_cfg["log_every"] * tokens_per_step / max(elapsed, 1e-9)})
            started = time.monotonic()
        due = [t for t in schedule if t <= tokens and t not in evaluated]
        if due:
            run_eval(max(due)); evaluated.update(due)
        if train_cfg.get("stop_after_steps") and step == int(train_cfg["stop_after_steps"]) and step < total_steps:
            # Testing hook: simulate an interruption right after a checkpoint.
            _save_checkpoint(checkpoint_path, model, optimizer, step, evaluated, channel, tracker, device,
                             trainable_only=trainable_only)
            return {"steps": step, "tokens": tokens, "interrupted": True}
        if time.monotonic() - last_checkpoint > 60 * train_cfg["checkpoint_minutes"]:
            _save_checkpoint(checkpoint_path, model, optimizer, step, evaluated, channel, tracker, device,
                             trainable_only=trainable_only)
            last_checkpoint = time.monotonic()
    _save_checkpoint(checkpoint_path, model, optimizer, step, evaluated, channel, tracker, device,
                     trainable_only=trainable_only)
    torch.save({"model": model_state(model, trainable_only), "config": config,
                "composer_schedule": _schedule_state(channel), **({"trainable_only": True} if trainable_only else {})},
               output_dir / "final.pt")
    if tracker is not None:
        (output_dir / "cards.json").write_text(json.dumps(tracker.cards, indent=2, default=str) + "\n")
    if not (output_dir / "manifest.json").exists():
        write_run_metadata(output_dir, config, git_at_start=git_at_start, device=device,
                           parameters=sum(p.numel() for p in model.parameters()),
                           channel_parameters=sum(p.numel() for p in channel.parameters()) if channel else 0,
                           steps=total_steps, tokens_per_step=tokens_per_step)
    return {"steps": step, "tokens": step * tokens_per_step}


def set_aside_partial_run(output_dir: Path) -> dict[str, Any]:
    """Prepare a run folder whose run died before its first checkpoint for a fresh start: each file
    is kept as `<stem>.aborted-<n><suffix>` (so `metrics.jsonl` is free again); a folder holding a
    finished run (`manifest.json` or `final.pt`) is refused. Returns the git state, like
    `prepare_output_dir`."""
    if (output_dir / "manifest.json").exists() or (output_dir / "final.pt").exists():
        raise FileExistsError(f"{output_dir} holds a finished run but no checkpoint; refusing to restart it")
    previous = [int(m[1]) for p in output_dir.iterdir() if (m := re.search(r"\.aborted-(\d+)", p.name))]
    attempt = 1 + max(previous, default=0)
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and ".aborted-" not in path.name:
            path.rename(path.with_name(f"{path.stem}.aborted-{attempt}{path.suffix}"))
    return git_state()


def save_window_losses(path: Path, tokens: int, starts: list[int],
                       sink: dict[str, tuple[list[np.ndarray], list[np.ndarray]]]) -> None:
    """Add one evaluation's per-window sums/counts (strata × windows) to `path`, rewritten atomically."""
    arrays: dict[str, np.ndarray] = {}
    if path.exists():
        with np.load(path) as existing:
            arrays = {key: existing[key] for key in existing.files}
    strata = list(sink)
    if "strata" in arrays and arrays["strata"].tolist() != strata:
        raise ValueError(f"{path} holds strata {arrays['strata'].tolist()}, not {strata}")
    arrays["strata"] = np.asarray(strata)
    arrays["starts"] = np.asarray(starts, dtype=np.int64)
    arrays[f"sum_{tokens}"] = np.stack([np.concatenate(sink[name][0]) for name in strata])
    arrays[f"count_{tokens}"] = np.stack([np.concatenate(sink[name][1]) for name in strata])
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_window_losses(path: Path) -> dict[str, Any]:
    """`{"strata": [...], "starts": array, "evals": {tokens: (sums, counts)}}` from `eval_windows.npz`."""
    with np.load(path) as data:
        evals = {int(key[4:]): (data[key], data[f"count_{key[4:]}"]) for key in data.files if key.startswith("sum_")}
        return {"strata": data["strata"].tolist(), "starts": data["starts"], "evals": dict(sorted(evals.items()))}


def model_state(model: ChannelLM, trainable_only: bool = False) -> dict[str, torch.Tensor]:
    """`model.state_dict()`, or (trainable_only) without the frozen host weights: the host's trainable
    parameters (LoRA adapters) and everything outside the host (channel, context, their buffers)."""
    state = model.state_dict()
    if not trainable_only:
        return state
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    return {k: v for k, v in state.items() if not k.startswith("model.") or k in trainable}


def load_model_state(model: ChannelLM, state: dict[str, torch.Tensor], *, trainable_only: bool = False) -> None:
    """Inverse of `model_state`: a trainable-only state may omit exactly the frozen host weights."""
    if not trainable_only:
        model.load_state_dict(state)
        return
    missing, unexpected = model.load_state_dict(state, strict=False)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    bad = [k for k in missing if not k.startswith("model.") or k in trainable]
    if bad or unexpected:
        raise RuntimeError(f"trainable-only state does not fit the model: missing {bad[:5]}, unexpected {unexpected[:5]}")


def load_final(path: Path, device: torch.device | str = "cpu") -> ChannelLM:
    """Rebuild a trained `ChannelLM` from a run's `final.pt` (full or trainable-only state).

    The host is rebuilt from the config (a pretrained host is reloaded from its hub id), the channel
    from `data.ontology`, and a grown composer is re-grown to the saved dictionary size."""
    final = torch.load(Path(path), weights_only=False, map_location="cpu")
    config = final["config"]
    ontology = torch.load(config["data"]["ontology"], weights_only=False) if config["data"].get("ontology") else None
    base = build_model(config)
    channel, context = build_channel(config, ontology, base.get_input_embeddings().weight.shape[1])
    if channel is not None and ontology is not None:
        channel.set_unseen(ontology["heldout_entries"])
    saved = final.get("composer_schedule")
    if channel is not None and channel.composer is not None and saved is not None:
        composer = channel.composer
        if int(saved["atomics"]) > composer.atomics.shape[0]:
            composer.add_atomics(torch.zeros(int(saved["atomics"]) - composer.atomics.shape[0], composer.atomics.shape[1]))
        if int(saved["relations_count"]) > composer.relation_count:
            composer.add_relation_copies(torch.zeros(int(saved["relations_count"]) - composer.relation_count, dtype=torch.long))
        composer.set_schedule(FrameSchedule(saved["offsets"], saved["relations"], saved["fillers"]))
    model = ChannelLM(base, channel, context=context, host_mode=config["model"]["host_mode"] if config["model"]["pretrained"] else "train",
                      lora_rank=int(config["model"]["lora_rank"]))
    load_model_state(model, final["model"], trainable_only=bool(final.get("trainable_only", False)))
    return model.to(device).eval()


def _schedule_state(channel: SpanChannel | None) -> dict[str, torch.Tensor] | None:
    if channel is None or channel.composer is None:
        return None
    s = channel.composer.schedule
    return {"offsets": s.offsets.cpu(), "relations": s.relations.cpu(), "fillers": s.fillers.cpu(),
            "atomics": channel.composer.atomics.shape[0], "relations_count": channel.composer.relation_count}


def _save_checkpoint(path: Path, model: ChannelLM, optimizer: torch.optim.Optimizer, step: int, evaluated: set[int],
                     channel: SpanChannel | None, tracker: DevelopmentalDictionary | None, device: torch.device, *,
                     trainable_only: bool = False) -> None:
    state = {"model": model_state(model, trainable_only), "optimizer": optimizer.state_dict(), "step": step,
             "evaluated": sorted(evaluated), "rng_cpu": torch.get_rng_state(),
             "rng_cuda": torch.cuda.get_rng_state() if device.type == "cuda" else None,
             "composer_schedule": _schedule_state(channel), **({"trainable_only": True} if trainable_only else {}),
             "tracker": None if tracker is None else {k: getattr(tracker, k) for k in (
                 "momentum", "absolute", "initial_count", "step", "splits", "candidates", "records",
                 "cooldown_until", "frozen", "low_activity_steps", "provisional", "siblings", "cards")}}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _restore_growth(channel: SpanChannel, tracker: DevelopmentalDictionary, state: dict[str, Any],
                    optimizer: torch.optim.Optimizer) -> None:
    """Re-grow the composer to the checkpoint's dictionary size before loading weights."""
    composer, saved = channel.composer, state["composer_schedule"]
    extra = int(saved["atomics"]) - composer.atomics.shape[0]
    if extra > 0:
        old = composer.atomics
        composer.add_atomics(torch.zeros(extra, old.shape[1]))
        for group in optimizer.param_groups:
            group["params"] = [composer.atomics if p is old else p for p in group["params"]]
    extra_rel = int(saved["relations_count"]) - composer.relation_count
    if extra_rel > 0:
        old = composer.relation_vectors()
        composer.add_relation_copies(torch.zeros(extra_rel, dtype=torch.long))
        for group in optimizer.param_groups:
            group["params"] = [composer.relation_vectors() if p is old else p for p in group["params"]]
    composer.set_schedule(FrameSchedule(saved["offsets"], saved["relations"], saved["fillers"]))
    for key, value in (state.get("tracker") or {}).items():
        setattr(tracker, key, value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(train(yaml.safe_load(args.config.read_text()), args.output, resume=args.resume)))


if __name__ == "__main__":
    main()
