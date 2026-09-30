"""From-scratch causal-LM training with the span channel (B7), and stratified evaluation.

One run = one condition × seed. Conditions differ only in the channel; the batches are a pure
function of `(data.seed, step)`, so runs are paired. The run folder holds `resolved_config.yaml`,
`manifest.json`, `metrics.jsonl` (training loss and stratified evaluations at log-spaced token
counts), `checkpoint.pt` (latest, rewritten every `checkpoint_minutes`) and `final.pt`.

Evaluation strata (per target token `j`, predicted from position `j − 1`):
`all`; `unlinked` (not inside or within 8 tokens after a linked span); `inside` (subtokens 2..ℓ of
a span); `after` (the 8 tokens after a span) and its splits by entry status — `after_heldout`
(held-out concepts, never linked in training), `after_rare` / `after_mid` / `after_frequent`
(training frequency 1–9 / 10–99 / ≥ 100) — and by span length (`after_len1`, `after_len2`,
`after_len3plus`).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import GPT2Config, GPT2LMHeadModel

from ..compose import FrameComposer, FrameSchedule
from ..context import CausalLocalContext
from ..data.corpus import TokenCorpus, collate_windows, eval_windows, sample_batch
from ..developmental import DevelopmentalConfig, DevelopmentalDictionary
from ..integrations.transformers import ChannelLM
from ..provenance import prepare_output_dir, write_run_metadata
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
                       "concept_factor": "induced"}.items():
        channel.setdefault(key, value)
    c.setdefault("device", "cuda")
    return c


def eval_token_schedule(first: int, total: int) -> list[int]:
    """Log-spaced evaluation points: first, 2·first, 4·first, … and the end."""
    points, t = [], first
    while t < total:
        points.append(t); t *= 2
    return points + [total]


def build_model(config: dict[str, Any]) -> GPT2LMHeadModel:
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
                           gate_bias=float(settings["gate_bias"]),
                           semantic_dimension=width if config["train"]["semantic_weight"] else 0), None
    schedule = FrameSchedule(ontology["offsets"], ontology["relations"], ontology["fillers"])
    context_window = int(settings["context_window"])
    composer = FrameComposer(
        schedule, int(ontology["atomic_count"]), int(ontology["relation_count"]), int(settings["dimension"]),
        operator=settings["operator"], mode=settings["composition"], concept_factor=settings["concept_factor"],
        key_dimension=int(settings["key_dimension"]),
        context_dimension=int(settings["key_dimension"]) if context_window else 0,
    )
    channel = SpanChannel(composer, width, entry_count=entries, gate_bias=float(settings["gate_bias"]),
                          semantic_dimension=width if config["train"]["semantic_weight"] else 0)
    context = CausalLocalContext(width, int(settings["key_dimension"]), window=context_window) if context_window else None
    return channel, context


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
        "after", "after_heldout", "after_rare", "after_mid", "after_frequent", "after_len1", "after_len2", "after_len3plus")}
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
        for name in names:
            after[name][b, lo:hi] = True
    masks = {"all": torch.ones(shape, dtype=torch.bool), "inside": inside, **after}
    masks["unlinked"] = ~(inside | after["after"])
    return masks


@torch.no_grad()
def evaluate(model: ChannelLM, corpus: TokenCorpus, starts: list[int], config: dict[str, Any],
             frequency: np.ndarray | None, heldout: set[int], device: torch.device) -> dict[str, dict[str, float]]:
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
        for name, mask in stratum_masks(ids, spans, frequency, heldout).items():
            sums[name] = sums.get(name, 0.0) + float(per_token[mask].sum())
            counts[name] = counts.get(name, 0) + int(mask.sum())
    model.train()
    return {name: {"loss": sums[name] / counts[name] if counts[name] else float("nan"), "tokens": counts[name]}
            for name in sums}


def train(config: dict[str, Any], output_dir: Path, *, resume: bool = False) -> dict[str, Any]:
    config = resolve_config(config)
    checkpoint_path = output_dir / "checkpoint.pt"
    git_at_start = None
    if not (resume and checkpoint_path.exists()):
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
    channel, context = build_channel(config, ontology, base.config.n_embd)
    if channel is not None and ontology is not None:
        channel.set_unseen(ontology["heldout_entries"])
    model = ChannelLM(base, channel, context=context).to(device)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "wte" not in n and "wpe" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "wte" not in n and "wpe" not in n)]
    train_cfg = config["train"]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": train_cfg["weight_decay"]},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=train_cfg["lr"],
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
        model.load_state_dict(state["model"])
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

    def run_eval(tokens: int) -> None:
        results = evaluate(model, eval_corpus, eval_starts, config, frequency, heldout, device)
        for stratum, value in results.items():
            log({"type": "eval", "step": step, "tokens": tokens, "stratum": stratum, **value})
        evaluated.add(tokens)

    if 0 not in evaluated:
        run_eval(0)
    model.train()
    started = time.monotonic()
    while step < total_steps:
        lr = _lr(step, total_steps, warmup_steps, train_cfg["lr"], train_cfg["min_lr_ratio"])
        for group in optimizer.param_groups:
            group["lr"] = lr
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
            (loss / accum).backward()
            total_loss += float(loss.detach()) / accum
        if tracker is not None:
            tracker.observe()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
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
            _save_checkpoint(checkpoint_path, model, optimizer, step, evaluated, channel, tracker, device)
            return {"steps": step, "tokens": tokens, "interrupted": True}
        if time.monotonic() - last_checkpoint > 60 * train_cfg["checkpoint_minutes"]:
            _save_checkpoint(checkpoint_path, model, optimizer, step, evaluated, channel, tracker, device)
            last_checkpoint = time.monotonic()
    _save_checkpoint(checkpoint_path, model, optimizer, step, evaluated, channel, tracker, device)
    torch.save({"model": model.state_dict(), "config": config,
                "composer_schedule": _schedule_state(channel)}, output_dir / "final.pt")
    if tracker is not None:
        (output_dir / "cards.json").write_text(json.dumps(tracker.cards, indent=2, default=str) + "\n")
    if not (output_dir / "manifest.json").exists():
        write_run_metadata(output_dir, config, git_at_start=git_at_start, device=device,
                           parameters=sum(p.numel() for p in model.parameters()),
                           channel_parameters=sum(p.numel() for p in channel.parameters()) if channel else 0,
                           steps=total_steps, tokens_per_step=tokens_per_step)
    return {"steps": step, "tokens": step * tokens_per_step}


def _schedule_state(channel: SpanChannel | None) -> dict[str, torch.Tensor] | None:
    if channel is None or channel.composer is None:
        return None
    s = channel.composer.schedule
    return {"offsets": s.offsets.cpu(), "relations": s.relations.cpu(), "fillers": s.fillers.cpu(),
            "atomics": channel.composer.atomics.shape[0], "relations_count": channel.composer.relation_count}


def _save_checkpoint(path: Path, model: ChannelLM, optimizer: torch.optim.Optimizer, step: int, evaluated: set[int],
                     channel: SpanChannel | None, tracker: DevelopmentalDictionary | None, device: torch.device) -> None:
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
             "evaluated": sorted(evaluated), "rng_cpu": torch.get_rng_state(),
             "rng_cuda": torch.cuda.get_rng_state() if device.type == "cuda" else None,
             "composer_schedule": _schedule_state(channel),
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
