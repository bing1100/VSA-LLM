"""Evaluation-only re-scoring of finished E9 runs (WP-PQ1): the filler / non-filler loss split (claim A's copy
concern) and the claim-B quantization controls, on the run's own evaluation windows, from its `final.pt`.

**Filler strata** (novelty check §2.2 / A-B5). T5 text is generated from the ontology whose frames the channel reads,
so a lower loss after a term could be the model copying filler names that the template writes right after the term
("Plurb Standard: a customer support policy … owned by the Zash Team"). Every after-span stratum `X` (`after`,
`after_heldout`, `after_rare_seen`, `after_unseen`, `after_len3plus`, …) is split into `X_filler` — targets in the 8
tokens after a span that belong to an occurrence, starting after the span, of a token sequence verbalizing a filler of
*that span's* ontology frame — and `X_nonfiller` (the rest of `X`). A filler's sequences are its aliases (a filler that
names a concept: every alias of that concept), the track lexicon's wording and its value, without a leading article,
each as written, lower-cased and capitalized, with and without a leading space, tokenized by the track's tokenizer
(`fillers`; one table per track and tokenizer family). The same masks are the trainer's opt-in `eval.filler_strata`
(`training.lm.stratum_masks`); the old strata are recomputed exactly (`ref_check` against the run's
`eval_windows.npz`).

**Variants** (`--variants`; a name is a base, then optional flags):

- bases: `ref` (the run as trained, bf16 autocast — the trainer's own evaluation), `int8-A`, `int4-A` (torchao RTN,
  tile-packed, as `e4_quant`; CUDA), `int4-B` (+ channel quantized), and the simulated weight-only schemes of
  `evaluation.quantization` (CPU or CUDA): `int4-rtn`, `int4-hqq`, `int4-nf4`, `int4-gptq`, `int4-awq` (GPTQ and AWQ
  calibrate on `--calibration-windows` windows of the run's training corpus — the in-domain mix with the rare terms —
  drawn with their own seed; `--calibration-corpus general` uses the track's `train-general`);
- `-emb`: the input-embedding table quantized too (4-bit `rtn` grid; a tied output head keeps its 16-bit copy);
  `-embhead`: the tied table, i.e. lookup and head (GGUF-style);
- `-off`: the channel switched off (gate 0: no row is injected) — claim-B control B-B2, "how much of the INT4 gap is the
  channel and how much the C5-trained host weights". A run without a channel answers `-off` with its plain variant.

Profiles (`--profile`): `controls` = `ref ref-off int4-A int4-A-off int4-A-emb int4-hqq int4-nf4 int4-gptq int4-awq`
(P0, C0′, C2, C5 — the claim-B table), `light` = `ref int4-A` (the ablation / row-source arms); `auto` picks by the
model in the run name. Each variant group (base + embedding flag) is quantized once and evaluated with the channel on
and, if asked, off.

Outputs (`RUN/rescore/`, run-folder contract): `windows.npz` (`strata`, `starts`, `count`, `sum_<variant>`: per-window
loss sums, strata × windows, paired with every other variant and with the run's `eval_windows.npz`), `rescore.json`
(per variant: stratified losses, quantization record, seconds; `ref_check`), `resolved_config.yaml`, `manifest.json`.
`--resume` keeps the variants already in `windows.npz`.

    python -m vsa_embed.experiments.e9_rescore fillers --track t5 [--family smollm2] [--output PATH]
    python -m vsa_embed.experiments.e9_rescore score --run RUN [--variants ref int4-A …|--profile auto] [--resume]
    python -m vsa_embed.experiments.e9_rescore queue --stage t5 [--priority 51] [--profile auto] [--models C5 …]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import yaml

from ..data.corpus import TokenCorpus, eval_windows, sample_batch
from ..provenance import git_state, write_run_metadata
from ..row_sources import frames_digest, load_filler_index, save_filler_table
from ..span_channel import AliasTable, normalize_alias
from ..training.lm import evaluate, load_final, resolve_config

ROOT = Path("experiments/e9-retrofit")
FILLER_ROOT = Path("~/data/vsa-llm/e9/filler-tables").expanduser()
PROFILES = {
    "controls": ("ref", "ref-off", "int4-A", "int4-A-off", "int4-A-emb", "int4-hqq", "int4-nf4", "int4-gptq", "int4-awq"),
    "light": ("ref", "int4-A"),
}
CONTROL_MODELS = frozenset({"P0", "C0p", "C0'", "C2", "C5"})
TORCHAO_BASES = {"int8-A": (8, False), "int4-A": (4, False), "int4-B": (4, True)}
SIMULATED_BASES = {"int4-rtn": "rtn", "int4-hqq": "hqq", "int4-nf4": "nf4", "int4-gptq": "gptq", "int4-awq": "awq"}
BASES = ("ref", *TORCHAO_BASES, *SIMULATED_BASES)
ARTICLES = ("the ", "a ", "an ")
CALIBRATION_SEED = 777


# ---------------------------------------------------------------- filler tables


def filler_table_path(track: str, family: str = "smollm2", *, root: Path = FILLER_ROOT) -> Path:
    return Path(root) / f"{track}-{family}.pt"


def _clean(surface: str) -> str | None:
    text = " ".join(str(surface).split())
    lowered = text.lower()
    for article in ARTICLES:
        if lowered.startswith(article):
            text, lowered = text[len(article):], lowered[len(article):]
    return text if len(text) >= 2 and any(ch.isalnum() for ch in text) else None


def atomic_surfaces(ontology: dict[str, Any], table: AliasTable | None, lexicon: Any = None) -> list[list[str]]:
    """Per atomic the surface strings that verbalize it (see the module docstring)."""
    names = [str(n) for n in (ontology.get("concept_names") or [])]
    concept_of = {name: c for c, name in enumerate(names)}
    aliases: dict[int, set[str]] = {}
    display = {normalize_alias(n): n for n in names}
    if table is not None:
        for alias, entry in table.alias_to_entry.items():
            for concept in table.entry_concepts[entry]:
                aliases.setdefault(int(concept), set()).add(display.get(alias, alias))
    out = []
    for atom in ontology["atomic_names"]:
        value = atom.split(":", 1)[1] if ":" in atom else atom
        found: set[str] = set()
        text = lexicon.text(atom) if lexicon is not None and hasattr(lexicon, "text") else None
        if text:
            found.add(str(text))
        concept = concept_of.get(value)
        if concept is not None:
            found |= aliases.get(concept, set())
        elif not text and not any(ch.isdigit() for ch in value):
            found.add(value.replace("_", " "))
        out.append(sorted({c for c in map(_clean, found) if c}))
    return out


def surface_sequences(surfaces: Sequence[str], tokenizer: Any) -> list[list[int]]:
    """Token sequences of each surface as written, lower-cased and capitalized, with and without a leading space."""
    seen: set[tuple[int, ...]] = set()
    for surface in surfaces:
        for form in {surface, surface.lower(), surface[:1].upper() + surface[1:]}:
            for text in (" " + form, form):
                ids = tuple(tokenizer.encode(text, add_special_tokens=False))
                if ids:
                    seen.add(ids)
    return [list(s) for s in sorted(seen)]


def build_filler_table(ontology: dict[str, Any], table: AliasTable | None, tokenizer: Any, output: Path, *, lexicon: Any = None,
                       meta: dict[str, Any] | None = None) -> dict[str, Any]:
    surfaces = atomic_surfaces(ontology, table, lexicon)
    sequences = [surface_sequences(s, tokenizer) for s in surfaces]
    record = {**(meta or {}), "atomics_with_surfaces": sum(bool(s) for s in surfaces),
              "sequences": sum(map(len, sequences)), "rule": "aliases of the concept a filler names + lexicon text + "
              "non-numeric value; leading article stripped; as written / lower / capitalized; with and without a leading space"}
    return save_filler_table(output, ontology=ontology, surfaces=surfaces, sequences=sequences, meta=record)


def ensure_filler_table(track: str, family: str = "smollm2", output: Path | None = None, *, overwrite: bool = False) -> Path:
    """Build a track's filler table once (CPU: the track lexicon, the alias table and the family's tokenizer)."""
    from transformers import AutoTokenizer

    from ..evaluation.channel_probes import load_alias_table, resolve_alias_table
    from .e9_tracks import FAMILY_TOKENIZERS, ensure_alias_table, lexicon_for, track_spec
    path = Path(output) if output else filler_table_path(track, family)
    if path.exists() and not overwrite:
        return path
    spec = track_spec(track, family)
    ontology = torch.load(spec.ontology, weights_only=False)
    alias_path = ensure_alias_table(spec)
    table = load_alias_table(alias_path) if alias_path else resolve_alias_table(ontology, spec.ontology)[0]
    tokenizer = AutoTokenizer.from_pretrained(FAMILY_TOKENIZERS[family], local_files_only=True)
    build_filler_table(ontology, table, tokenizer, path, lexicon=lexicon_for(spec, ontology),
                       meta={"track": track, "family": family, "tokenizer": FAMILY_TOKENIZERS[family], "ontology": str(spec.ontology)})
    return path


# ---------------------------------------------------------------- variants


def parse_variant(name: str) -> dict[str, Any]:
    """`<base>[-emb|-embhead][-off]` → {"base", "emb", "off"}."""
    rest, off, emb = name, False, None
    if rest.endswith("-off"):
        rest, off = rest[:-4], True
    for flag in ("embhead", "emb"):
        if rest.endswith(f"-{flag}"):
            rest, emb = rest[:-(len(flag) + 1)], flag
            break
    if rest not in BASES:
        raise ValueError(f"unknown variant {name!r}: base must be one of {BASES}, flags -emb/-embhead then -off")
    return {"base": rest, "emb": emb, "off": off, "name": name}


def plain_variant(name: str) -> str:
    """The variant a run without a channel evaluates for `name` (`-off` dropped)."""
    return name[:-4] if name.endswith("-off") else name


def profile_variants(profile: str, model: str | None) -> list[str]:
    if profile == "auto":
        profile = "controls" if (model or "") in CONTROL_MODELS else "light"
    if profile not in PROFILES:
        raise ValueError(f"profile must be auto or one of {sorted(PROFILES)}")
    return list(PROFILES[profile])


@contextlib.contextmanager
def channel_off(model: Any) -> Iterator[None]:
    """Within the block the model runs without its channel (gate 0: no row injected); the P1 context needs the channel."""
    channel = model.channel
    model.channel = None
    try:
        yield
    finally:
        model.channel = channel


def calibration_forward(model: Any, config: dict[str, Any], *, windows: int, corpus: str = "train", seed: int = CALIBRATION_SEED,
                        device: torch.device) -> Any:
    """`forward(step)` for `simulate_weight_only_`: `windows` sequences of the run's length from the training corpus
    (`general`: the track's `train-general` beside it), sampled with their own seed, run with the channel live."""
    path = Path(config["data"]["train"])
    if corpus == "general":
        path = path.with_name("train-general")
    elif corpus != "train":
        path = Path(corpus)
    data = TokenCorpus.open(path)
    length, batch = int(config["model"]["seq_len"]), max(1, int(config["eval"]["batch"]))
    batches = [sample_batch(data, seed=seed, step=i, micro_step=0, batch=min(batch, windows - i * batch), length=length,
                            min_subtokens=int(config["data"]["min_subtokens"])) for i in range(-(-windows // batch))]

    def forward(step: Any) -> None:
        for ids, spans in batches:
            ids_d = ids.to(device)
            spans_d = {k: v.to(device) for k, v in spans.items()} if model.channel is not None else None

            def run(ids_d=ids_d, spans_d=spans_d) -> None:
                with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    model(ids_d, spans=spans_d)

            step(run)
    forward.record = {"windows": windows, "corpus": str(path), "seed": seed, "length": length}
    return forward


def prepare_variant(run_dir: Path, base: str, emb: str | None, *, device: torch.device, group_size: str | int,
                    calibration: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Load the run and apply one variant group's weights (base scheme + embedding flag)."""
    from ..evaluation.quantization import (auto_group_size, merge_lora, quantize_input_embedding_, simulate_weight_only_)
    from .e4_quant import quantize_channel, quantize_host
    model = load_final(run_dir / "final.pt", device)
    info: dict[str, Any] = {}
    size = auto_group_size(model.model) if group_size == "auto" else int(group_size)
    if base in TORCHAO_BASES:
        bits, channel = TORCHAO_BASES[base]
        if bits == 4 and device.type != "cuda":
            raise RuntimeError("int4-A / int4-B (torchao tile-packed) need CUDA; the simulated int4-* schemes run on the CPU")
        info = {"scheme": "torchao", "bits": bits, "group_size": size, **quantize_host(model, bits, size)}
        if channel:
            info["channel_tensors"] = sorted(quantize_channel(model, bits, size))
    elif base in SIMULATED_BASES:
        scheme = SIMULATED_BASES[base]
        info["lora_merged"] = merge_lora(model.model)
        forward = None
        if scheme in {"gptq", "awq"}:
            forward = calibration_forward(model, calibration["config"], windows=calibration["windows"], corpus=calibration["corpus"],
                                          device=device)
            info["calibration"] = forward.record
        info.update(simulate_weight_only_(model.model, scheme, group_size=size, forward=forward))
    if emb:
        info.update(quantize_input_embedding_(model.model, size, include_head=emb == "embhead"))
    return model, info


# ---------------------------------------------------------------- scoring a run


def _sink_arrays(sink: dict[str, tuple[list[np.ndarray], list[np.ndarray]]]) -> tuple[list[str], np.ndarray, np.ndarray]:
    strata = list(sink)
    return (strata, np.stack([np.concatenate(sink[s][0]) for s in strata]),
            np.stack([np.concatenate(sink[s][1]) for s in strata]).astype(np.int32))


def _write_windows(path: Path, strata: list[str], starts: list[int], counts: np.ndarray, sums: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, strata=np.asarray(strata), starts=np.asarray(starts, dtype=np.int64), count=counts,
                            **{f"sum_{k}": v for k, v in sums.items()})
    temporary.replace(path)


def load_rescore(folder: Path) -> dict[str, Any] | None:
    """`{"strata", "starts", "count", "sums": {variant: strata × windows}, "record"}` of a rescore folder, or None."""
    path = Path(folder) / "windows.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        out = {"strata": data["strata"].tolist(), "starts": data["starts"], "count": data["count"],
               "sums": {key[4:]: data[key] for key in data.files if key.startswith("sum_")}}
    record_path = Path(folder) / "rescore.json"
    out["record"] = json.loads(record_path.read_text()) if record_path.exists() else {}
    return out


def score_run(run_dir: Path, output: Path | None = None, *, variants: Sequence[str] = ("ref",), fillers: Path | None | str = "auto",
              device: str = "cuda", group_size: str | int = "auto", calibration_windows: int = 64, calibration_corpus: str = "train",
              eval_batch: int | None = None, resume: bool = False, overwrite: bool = False) -> dict[str, Any]:
    """Evaluate `run_dir` under each variant on its evaluation windows with the old and the filler strata."""
    from .e4_quant import reference_check
    run_dir = Path(run_dir)
    output = Path(output) if output else run_dir / "rescore"
    final = torch.load(run_dir / "final.pt", weights_only=False, map_location="cpu")
    config = resolve_config(final["config"])
    del final
    if eval_batch:
        config["eval"]["batch"] = int(eval_batch)
    target = torch.device(device if torch.cuda.is_available() else "cpu")
    ontology = torch.load(config["data"]["ontology"], weights_only=False) if config["data"].get("ontology") else None
    if ontology is None:
        raise ValueError("rescoring needs the run's ontology")
    if fillers == "auto":
        track, family = config.get("e9_track"), config.get("e9_family", "smollm2")
        if not track:
            raise ValueError("the run records no e9_track; pass --fillers PATH (or --no-fillers)")
        fillers = ensure_filler_table(track, family)
    index = load_filler_index(Path(fillers), ontology) if fillers else None
    heldout = set(int(e) for e in ontology["heldout_entries"])
    frequency = np.asarray(ontology["train_frequency"]) if ontology.get("train_frequency") is not None else None
    eval_corpus = TokenCorpus.open(Path(config["data"]["eval"]))
    starts = eval_windows(eval_corpus, count=config["eval"]["windows"], length=config["model"]["seq_len"])
    if overwrite and output.exists():
        for name in ("windows.npz", "rescore.json", "resolved_config.yaml", "manifest.json"):
            (output / name).unlink(missing_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    git_at_start = git_state()
    existing = load_rescore(output) if resume else None
    if not resume and (output / "windows.npz").exists():
        raise FileExistsError(f"{output} holds a rescoring; pass --resume or --overwrite")
    strata, counts = (existing["strata"], existing["count"]) if existing else (None, None)
    sums: dict[str, np.ndarray] = dict(existing["sums"]) if existing else {}
    record: dict[str, Any] = existing["record"] if existing and existing["record"] else {
        "run": str(run_dir), "id": f"{run_dir.parent.name}__{run_dir.name}", "variants": {}}
    has_channel = config["channel"]["mode"] != "none"
    record.update(channel=config["channel"]["mode"], fillers=str(fillers) if fillers else None,
                  frames_sha256=frames_digest(ontology), windows=len(starts))
    requested = [parse_variant(v) for v in variants]
    todo = [v for v in requested if v["name"] not in sums and (has_channel or not v["off"])]
    groups: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
    for v in todo:
        groups.setdefault((v["base"], v["emb"]), []).append(v)
    calibration = {"config": config, "windows": int(calibration_windows), "corpus": calibration_corpus}
    for (base, emb), members in groups.items():
        started = time.monotonic()
        model, info = prepare_variant(run_dir, base, emb, device=target, group_size=group_size, calibration=calibration)
        prepared = time.monotonic() - started
        ordered = sorted(members, key=lambda m: m["off"])
        for v in ordered:
            began = time.monotonic()
            sink: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
            with channel_off(model) if v["off"] else contextlib.nullcontext():
                results = evaluate(model, eval_corpus, starts, config, frequency, heldout, target, window_sink=sink, fillers=index)
            names, these, these_counts = _sink_arrays(sink)
            if strata is None:
                strata, counts = names, these_counts
            elif names != strata or not np.array_equal(these_counts, counts):
                raise RuntimeError(f"{run_dir}: variant {v['name']} produced other strata or target counts")
            sums[v["name"]] = these
            record["variants"][v["name"]] = {"strata": results, "quantization": info, "channel_off": v["off"],
                                             "seconds": round(time.monotonic() - began + (prepared if v is ordered[0] else 0), 2)}
            _write_windows(output / "windows.npz", strata, starts, counts, sums)
            print(json.dumps({"run": record["id"], "variant": v["name"], "all": results["all"]["loss"],
                              "seconds": record["variants"][v["name"]]["seconds"]}), flush=True)
        del model
        if target.type == "cuda":
            torch.cuda.empty_cache()
    if "ref" in sums and strata is not None:
        record["ref_check"] = reference_check(run_dir, strata, starts, sums["ref"], counts)
    if not has_channel:
        record["off_variants"] = "a run without a channel answers -off with its plain variant"
    (output / "rescore.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    settings = {"run": str(run_dir), "variants": list(variants), "fillers": str(fillers) if fillers else None,
                "group_size": group_size, "calibration_windows": calibration_windows, "calibration_corpus": calibration_corpus,
                "eval_batch": eval_batch, "device": device}
    write_run_metadata(output, settings, git_at_start=git_at_start, device=target, variants=sorted(sums))
    return record


# ---------------------------------------------------------------- queueing


def model_of(stem: str) -> str | None:
    match = re.search(r"-(?P<model>[^-]+)-s\d+$", stem)
    return match["model"] if match else None


def rescore_command(run_dir: Path, variants: Sequence[str], *, python: str = sys.executable, fillers: Path | None = None,
                    calibration_windows: int | None = None, batch_size: int | None = None) -> list[str]:
    return [python, "-m", "vsa_embed.experiments.e9_rescore", "score", "--run", str(run_dir), "--variants", *variants,
            "--output", str(Path(run_dir) / "rescore"), "--resume",
            *(["--fillers", str(fillers)] if fillers else []),
            *(["--calibration-windows", str(calibration_windows)] if calibration_windows else []),
            *(["--eval-batch", str(batch_size)] if batch_size else [])]


def queue_stage(stage: str, *, priority: int = 51, profile: str = "auto", models: Sequence[str] | None = None,
                root: Path = ROOT, queue_dir: Path | None = None, python: str | None = None) -> list[str]:
    """One rescoring job per config of `stage` (run folders need not exist yet: the job runs after the stage's
    training at a lower priority number); named `<stage>-<stem>-rescore`, as `e9_plan`'s chained jobs (idempotent)."""
    from vsa_embed.jobqueue import DEFAULT_DIR, add

    from .cpt_plan import pinned_python
    from .e9_plan import EVAL_JOB_BATCH, _config_host, stage_python
    queued = []
    configs = sorted((Path(root) / "configs" / stage).glob("*.yaml"))
    hosts = [h for h in (_config_host(yaml.safe_load(p.read_text())) for p in configs) if h]
    python = python or (stage_python(hosts) if hosts else pinned_python())
    for path in configs:
        model = model_of(path.stem)
        if models and model not in models:
            continue
        config = yaml.safe_load(path.read_text())
        run_dir = Path(root) / "runs" / stage / path.stem
        command = rescore_command(run_dir, profile_variants(profile, model), python=python,
                                  batch_size=EVAL_JOB_BATCH.get(_config_host(config) or ""))
        name = f"{stage}-{path.stem}-rescore"
        try:
            add(queue_dir or DEFAULT_DIR, command, name=name, priority=priority, min_free_gb=5, env={"PYTHONPATH": "src"}, resume_args=[])
            queued.append(name)
        except FileExistsError:
            pass
    return queued


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    fill = sub.add_parser("fillers", help="build a track's filler table")
    fill.add_argument("--track", required=True); fill.add_argument("--family", default="smollm2")
    fill.add_argument("--output", type=Path, default=None); fill.add_argument("--overwrite", action="store_true")
    score = sub.add_parser("score", help="rescore one finished run")
    score.add_argument("--run", type=Path, required=True); score.add_argument("--output", type=Path, default=None)
    score.add_argument("--variants", nargs="+", default=None, help="variant names (see the module docstring)")
    score.add_argument("--profile", default=None, help="controls | light | auto (instead of --variants)")
    score.add_argument("--fillers", type=Path, default=None, help="filler table (default: the run's track's, built if missing)")
    score.add_argument("--no-fillers", action="store_true", help="old strata only")
    score.add_argument("--device", default="cuda"); score.add_argument("--group-size", default="auto")
    score.add_argument("--calibration-windows", type=int, default=64)
    score.add_argument("--calibration-corpus", default="train", help="train | general | a corpus folder")
    score.add_argument("--eval-batch", type=int, default=None)
    score.add_argument("--resume", action="store_true"); score.add_argument("--overwrite", action="store_true")
    queue = sub.add_parser("queue", help="queue one rescoring job per config of a stage")
    queue.add_argument("--stage", required=True); queue.add_argument("--priority", type=int, default=51)
    queue.add_argument("--profile", default="auto"); queue.add_argument("--models", nargs="*", default=None)
    queue.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if args.command == "fillers":
        print(json.dumps({"path": str(ensure_filler_table(args.track, args.family, args.output, overwrite=args.overwrite))}))
    elif args.command == "score":
        variants = args.variants or profile_variants(args.profile or "auto", model_of(args.run.name))
        fillers: Path | None | str = None if args.no_fillers else (args.fillers or "auto")
        group_size: str | int = args.group_size if args.group_size == "auto" else int(args.group_size)
        record = score_run(args.run, args.output, variants=variants, fillers=fillers, device=args.device, group_size=group_size,
                           calibration_windows=args.calibration_windows, calibration_corpus=args.calibration_corpus,
                           eval_batch=args.eval_batch, resume=args.resume, overwrite=args.overwrite)
        print(json.dumps({"run": record["id"], "variants": sorted(record["variants"]), "ref_check": record.get("ref_check")}, default=str))
    else:
        print(json.dumps({"queued": queue_stage(args.stage, priority=args.priority, profile=args.profile, models=args.models,
                                                root=args.root)}, indent=2))


if __name__ == "__main__":
    main()
