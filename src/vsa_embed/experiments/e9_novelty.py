"""E9 novelty screen (author decision 55): rank candidate tracks by how new their vocabulary is to the host
before one is built and queued.

R9 ("Across tracks") measured the **novelty index** of a track as the relative loss change of plain continued
training after its terms, C0′ − P0 (large when the host does not know the terms), and found that the channel's gain
(C5 − C0′) follows it: T5 > T4 > T1 = WordNet = 0. Both need trained runs. This module calibrates **P0-only
proxies** (one evaluation of the original host, or no GPU at all) on the four finished tracks, and measures them on
candidate corpora:

- `inside_loss` — P0 loss on subtokens 2..ℓ of linked spans (stratum `inside`);
- `after_unlinked` — P0 loss after linked spans relative to unlinked text (`after` / `unlinked`);
- `inside_nats_per_span` — P0 nats spent spelling a linked term after its first subtoken: `inside` loss × `inside`
  targets / linked spans (ℓ ≥ 2) in the evaluation windows (the term's surprisal to the host);
- `subtokens_per_span` — mean subtokens of the linked spans (ℓ ≥ 2) in the evaluation windows;
- `general_to_domain` — linked spans (ℓ ≥ 2) per 1,000 tokens of general text (FineWeb-Edu, the hosts' kind of
  pretraining text; the track's `eval-general` split, for WordNet the first tokens of its general training split)
  over the same rate in the domain evaluation windows (no GPU: corpus frequency of the vocabulary in general text);
- `general_absent_share` — share of the domain span occurrences whose entry never occurs in that general sample.

Agreement with the novelty ordering is reported as Kendall's τ and Spearman's ρ over tracks, with the exact
permutation p-value (with 4 tracks a perfect ordering has p = 1/24).

    python -m vsa_embed.experiments.e9_novelty calibrate --output experiments/e9-retrofit/novelty/calibration
    python -m vsa_embed.experiments.e9_novelty p0-config --data-root ROOT [--eval-split eval] [--windows 512]
        [--host SmolLM2-360M] --name NAME --output CONFIG.yaml
    python -m vsa_embed.experiments.e9_novelty proxies --run RUN [--general CORPUS] [--general-tokens 2300000]
    python -m vsa_embed.experiments.e9_novelty summarize --calibration DIR/calibration.json --runs RUNS
        --candidates experiments/e9-retrofit/novelty/candidates.yaml --output experiments/e9-retrofit/novelty

The decision-55 screen (`experiments/e9-retrofit/novelty/`): `calibration/` (the four finished tracks), `runs/` (P0 slices of
the candidates: the built D8 tracks T2 / T3 / T6 and the MeSH vocabularies of `ontologies/mesh_novel.py`), `candidates.md`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

RUNS = Path("experiments/e9-retrofit/runs")
CALIBRATION_TRACKS = ("t5", "t4", "t1", "wordnet")
CALIBRATION_HOSTS = ("SmolLM2-360M", "SmolLM2-135M")
NOVELTY_STRATA = ("after_heldout", "after_unseen", "after_rare_seen", "after_len3plus")
PROXIES = ("inside_loss", "after_unlinked", "inside_nats_per_span", "subtokens_per_span", "general_to_domain",
           "general_absent_share")
# Direction in which a proxy grows with novelty (+1) or shrinks (−1).
PROXY_SIGN = {"inside_loss": 1, "after_unlinked": 1, "inside_nats_per_span": 1, "subtokens_per_span": 1,
              "general_to_domain": -1, "general_absent_share": 1}
GENERAL_TOKENS = 2_300_000          # general-text sample (the tracks' `eval-general` holds 2.2M SmolLM2 tokens)
NULL_GAIN = 0.001                   # |gain| below 0.1%: tied at 0 (R9: every T1 / WordNet term stratum within ±0.06%, n.s.)
WINDOW = 1024


# -- run folders ---------------------------------------------------------------------------------------------------

def final_losses(run_dir: Path) -> dict[str, tuple[float, int]]:
    """{stratum: (loss, targets)} of a run's last evaluation (`metrics.jsonl`)."""
    rows = [json.loads(line) for line in (Path(run_dir) / "metrics.jsonl").read_text().splitlines() if line.strip()]
    evals = [r for r in rows if r.get("type") == "eval"]
    if not evals:
        raise ValueError(f"{run_dir} has no evaluation rows")
    last = max(r["tokens"] for r in evals)
    return {r["stratum"]: (float(r["loss"]), int(r.get("stratum_tokens", 0))) for r in evals if r["tokens"] == last}


def relative(reference: float, other: float) -> float:
    return (other - reference) / reference


def novelty_and_gain(runs: Path, track: str, host: str, *, mode: str = "full", seed: int = 1,
                     strata: Sequence[str] = NOVELTY_STRATA) -> dict[str, Any]:
    """Per stratum: novelty = (C0′ − P0) / P0 and gain = (C5 − C0′) / C0′ (seed `seed`), and their means."""
    p0 = final_losses(runs / track / f"{host}-frozen-P0-s1")
    c0 = final_losses(runs / track / f"{host}-{mode}-C0p-s{seed}")
    c5 = final_losses(runs / track / f"{host}-{mode}-C5-s{seed}")
    out: dict[str, Any] = {"novelty": {}, "gain": {}}
    for stratum in (*strata, "inside", "unlinked"):
        out["novelty"][stratum] = relative(p0[stratum][0], c0[stratum][0])
        out["gain"][stratum] = relative(c0[stratum][0], c5[stratum][0])
    out["novelty_mean"] = float(np.mean([out["novelty"][s] for s in strata]))
    out["gain_mean"] = float(np.mean([out["gain"][s] for s in strata]))
    out["p0"] = p0
    return out


# -- span proxies (CPU) --------------------------------------------------------------------------------------------

def window_mask(spans: dict[str, np.ndarray], starts: Sequence[int], length: int = WINDOW) -> np.ndarray:
    """Spans lying fully inside one of the evaluation windows (as `t1_open_corpus.window_mask`)."""
    begins = np.sort(np.asarray(starts, dtype=np.int64))
    if begins.size == 0:
        return np.zeros(len(spans["start"]), dtype=bool)
    index = np.searchsorted(begins, spans["inject"], side="right") - 1
    valid = index >= 0
    begin = begins[np.clip(index, 0, None)]
    return valid & (spans["inject"] < begin + length) & (spans["start"] >= begin)


def span_proxies(eval_spans: dict[str, np.ndarray], starts: Sequence[int], general_spans: dict[str, np.ndarray] | None,
                 general_tokens: int, *, min_subtokens: int = 2, length: int = WINDOW) -> dict[str, Any]:
    """Subtokens per linked span in the evaluation windows and the general-text frequency of the linked entries.

    `general_spans` are the spans of a general-text corpus linked with the same alias table (None: not measured);
    only spans starting in its first `general_tokens` tokens count."""
    keep = window_mask(eval_spans, starts, length) & (eval_spans["length"] >= min_subtokens)
    lengths = eval_spans["length"][keep].astype(np.float64)
    entries = eval_spans["entry"][keep].astype(np.int64)
    domain_tokens = len(starts) * length
    out: dict[str, Any] = {"linked_spans": int(keep.sum()), "linked_entries": int(np.unique(entries).size),
                           "subtokens_per_span": float(lengths.mean()) if lengths.size else math.nan,
                           "domain_spans_per_1k": 1e3 * float(keep.sum()) / max(1, domain_tokens)}
    if general_spans is None:
        out.update(general_spans_per_1k=None, general_to_domain=None, general_absent_share=None, general_tokens=0)
        return out
    gkeep = (general_spans["length"] >= min_subtokens) & (general_spans["start"] < general_tokens)
    gentries = general_spans["entry"][gkeep].astype(np.int64)
    top = int(max(entries.max(initial=0), gentries.max(initial=0))) + 1
    counts = np.bincount(gentries, minlength=top)
    rate = 1e3 * float(gkeep.sum()) / max(1, general_tokens)
    out.update(general_spans_per_1k=rate, general_tokens=int(general_tokens),
               general_to_domain=rate / out["domain_spans_per_1k"] if out["domain_spans_per_1k"] else None,
               general_absent_share=float((counts[entries] == 0).mean()) if entries.size else None)
    return out


def loss_proxies(p0: dict[str, tuple[float, int]], linked_spans: int) -> dict[str, float]:
    """P0-loss proxies from a P0 run's strata and the number of linked spans (ℓ ≥ 2) in its windows."""
    inside, inside_targets = p0["inside"]
    return {"inside_loss": inside, "after_unlinked": p0["after"][0] / p0["unlinked"][0],
            "inside_nats_per_span": inside * inside_targets / linked_spans if linked_spans else math.nan,
            "after_loss": p0["after"][0], "unlinked_loss": p0["unlinked"][0]}


def run_proxies(run_dir: Path, *, general: Path | None = None, general_tokens: int = GENERAL_TOKENS,
                min_subtokens: int = 2) -> dict[str, Any]:
    """Every proxy of a P0 (eval-only) run: its final strata, its evaluation windows (`eval_windows.npz`), its
    evaluation corpus's spans and, if given, a general-text corpus linked with the same alias table."""
    from vsa_embed.data.corpus import TokenCorpus
    run_dir = Path(run_dir)
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    starts = np.load(run_dir / "eval_windows.npz")["starts"]
    corpus = TokenCorpus.open(Path(config["data"]["eval"]))
    general_spans, tokens = None, 0
    if general is not None:
        general_corpus = TokenCorpus.open(Path(general))
        general_spans, tokens = general_corpus.spans, min(len(general_corpus), int(general_tokens))
    spans = span_proxies(corpus.spans, starts, general_spans, tokens, min_subtokens=min_subtokens,
                         length=int(config["model"]["seq_len"]))
    p0 = final_losses(run_dir)
    return {"run": str(run_dir), "eval": config["data"]["eval"], "windows": int(len(starts)), **spans,
            **loss_proxies(p0, spans["linked_spans"]), "strata": {k: list(v) for k, v in p0.items()}}


# -- rank agreement ------------------------------------------------------------------------------------------------

def _ranks(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = values.argsort(kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    for value in np.unique(values):             # average ranks of ties
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall's τ-b over the pairs (corrected for ties in either variable; NaN if one variable is constant)."""
    pairs = list(itertools.combinations(range(len(x)), 2))
    if not pairs:
        return math.nan
    concordance = sum(float(np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])) for i, j in pairs)
    untied_x = sum(1 for i, j in pairs if x[i] != x[j])
    untied_y = sum(1 for i, j in pairs if y[i] != y[j])
    if not untied_x or not untied_y:
        return math.nan
    return concordance / math.sqrt(untied_x * untied_y)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    rx, ry = _ranks(x), _ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def permutation_p(x: Sequence[float], y: Sequence[float]) -> float:
    """Exact one-sided p of Kendall's τ: the share of the n! orderings of `y` with τ ≥ the observed (n ≤ 8)."""
    if len(x) > 8:
        raise ValueError("exact permutation p only for n ≤ 8")
    observed = kendall_tau(x, y)
    perms = list(itertools.permutations(y))
    return sum(kendall_tau(x, p) >= observed - 1e-12 for p in perms) / len(perms)


def agreement(proxy: Sequence[float], target: Sequence[float], sign: int, *, null: float = 0.0) -> dict[str, float]:
    """Agreement of a proxy (oriented by `sign` so that larger = more novel) with a target's magnitude (novelty
    |C0′ − P0| or gain |C5 − C0′|); target magnitudes below `null` are tied at 0 (a null effect has no order)."""
    oriented = [sign * v for v in proxy]
    magnitude = [abs(v) if abs(v) >= null else 0.0 for v in target]
    return {"kendall_tau": kendall_tau(oriented, magnitude), "spearman": spearman(oriented, magnitude),
            "permutation_p": permutation_p(oriented, magnitude)}


# -- calibration ---------------------------------------------------------------------------------------------------

def general_corpus_for(track: str) -> Path | None:
    """The general-text corpus linked with a calibration track's alias table (WordNet: its general training split)."""
    from .e9_tracks import track_spec
    spec = track_spec(track)
    return spec.general_corpus if spec.general_corpus is not None else spec.data_root / "train"


def calibrate(runs: Path = RUNS, *, tracks: Sequence[str] = CALIBRATION_TRACKS, hosts: Sequence[str] = CALIBRATION_HOSTS,
              general_tokens: int = GENERAL_TOKENS, null_gain: float = NULL_GAIN) -> dict[str, Any]:
    """Novelty, gain and every P0 proxy per (track, host), and each proxy's rank agreement with novelty per host."""
    rows = []
    for host in hosts:
        for track in tracks:
            measured = novelty_and_gain(runs, track, host)
            proxies = run_proxies(runs / track / f"{host}-frozen-P0-s1", general=general_corpus_for(track),
                                  general_tokens=general_tokens)
            rows.append({"track": track, "host": host, "novelty": measured["novelty"], "gain": measured["gain"],
                         "novelty_mean": measured["novelty_mean"], "gain_mean": measured["gain_mean"],
                         **{k: proxies[k] for k in (*PROXIES, "linked_spans", "linked_entries", "windows", "domain_spans_per_1k",
                                                    "general_spans_per_1k")}})
    summary = {}
    for host in hosts:
        block = [r for r in rows if r["host"] == host]
        novelty = [r["novelty_mean"] for r in block]
        gain = [r["gain_mean"] for r in block]
        summary[host] = {proxy: {"novelty": agreement([r[proxy] for r in block], novelty, PROXY_SIGN[proxy]),
                                 "gain": agreement([r[proxy] for r in block], gain, PROXY_SIGN[proxy], null=null_gain)}
                         for proxy in PROXIES}
    return {"tracks": list(tracks), "hosts": list(hosts), "strata": list(NOVELTY_STRATA), "null_gain": null_gain, "rows": rows,
            "agreement": summary}


def render_calibration(result: dict[str, Any]) -> str:
    def pct(v: float) -> str:
        return f"{100 * v:+.1f}%"
    lines = ["# E9 novelty screen — proxy calibration (decision 55)", "",
             "Novelty = (C0′ − P0) / P0 and gain = (C5 − C0′) / C0′ after terms, seed 1, mean over the strata "
             f"{', '.join(result['strata'])}. Proxies come from the P0 run alone (or from no run: subtokens, general-text "
             "frequency).", "",
             "| track | host | novelty | gain | inside loss | after / unlinked | inside nats / span | subtokens / span | "
             "general / domain rate | general-absent share |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in result["rows"]:
        lines.append(f"| {r['track']} | {r['host']} | {pct(r['novelty_mean'])} | {pct(r['gain_mean'])} | {r['inside_loss']:.3f} | "
                     f"{r['after_unlinked']:.3f} | {r['inside_nats_per_span']:.2f} | {r['subtokens_per_span']:.2f} | "
                     f"{r['general_to_domain']:.3f} | {r['general_absent_share']:.2f} |")
    lines += ["", "Rank agreement with novelty magnitude and with gain magnitude over the tracks (τ = Kendall τ-b, ρ = Spearman, "
              "p = exact one-sided permutation p of τ; with 4 tracks a perfect ordering has p = 1/24 ≈ 0.042). Gains below "
              f"{100 * result['null_gain']:.1f}% in magnitude are tied at 0 (T1 and WordNet: n.s. in R9), so the best possible gain "
              "ordering has three levels (τ-b 0.91, p = 0.083).", "",
              "| host | proxy | τ novelty | ρ novelty | p | τ gain | ρ gain | p |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for host, block in result["agreement"].items():
        for proxy, a in block.items():
            n, g = a["novelty"], a["gain"]
            lines.append(f"| {host} | {proxy} | {n['kendall_tau']:+.2f} | {n['spearman']:+.2f} | {n['permutation_p']:.3f} | "
                         f"{g['kendall_tau']:+.2f} | {g['spearman']:+.2f} | {g['permutation_p']:.3f} |")
    return "\n".join(lines) + "\n"


# -- P0 slice configs ----------------------------------------------------------------------------------------------

def p0_config(data_root: Path, *, name: str, eval_split: str = "eval", windows: int = 512, host: str = "SmolLM2-360M",
              eval_batch: int = 4, train_split: str | None = None) -> dict[str, Any]:
    """An E9 P0 configuration (`e9_plan.run_config`, evaluation only) on a candidate corpus with fewer windows and a
    small evaluation batch (a slice measurement: ≤ 4 GB, a few minutes). `train_split` defaults to the evaluation
    split (the trainer opens it for the host-corpus check only)."""
    from .e9_plan import run_config
    root = Path(data_root)
    _, config = run_config(stage="novelty", host=host, mode="train", model="P0", seed=1, data_root=root, tokens=65_536,
                           lora_rank=64, host_lr=None, gate_bias=0.0, free_dimension=0, eval_split=eval_split)
    config["eval"].update(windows=int(windows), batch=int(eval_batch), first_tokens=65_536)
    config["data"]["train"] = str(root / (train_split or eval_split))
    config["experiment"] = f"e9-novelty-{name}-{host}-P0"
    return config


# -- MeSH candidate slices -----------------------------------------------------------------------------------------

DATA = Path("~/data/vsa-llm").expanduser()
SLICE_PUBMED = DATA / "pubmed/text-2026"
SLICE_GENERAL = [DATA / "fineweb-edu/sample/10BT/000_00000.parquet"]      # T4's general evaluation sample (first 2,000 docs)


def mesh_slice(output: Path, policy: Any, *, domain_counts: dict[str, int], general_counts: dict[str, int],
               tokenizer_name: str = "HuggingFaceTB/SmolLM2-360M", eval_tokens: int = 640_000, general_docs: int = 2000,
               workers: int = 4, pubmed: Path = SLICE_PUBMED, general: Sequence[Path] = SLICE_GENERAL) -> dict[str, Any]:
    """A P0 slice of a MeSH candidate vocabulary (`mesh_novel.NovelVocabularyPolicy`): `eval` = the first PubMed
    2025-26 abstracts (newest file first) that mention a selected name, `eval-general` = the first `general_docs`
    FineWeb-Edu documents, both linked with the candidate's alias table, and a minimal `ontology.pt` (no holdout,
    zero training frequencies) for an evaluation-only P0 run. Counts are keyed by alias (any case and spacing)."""
    import torch
    from transformers import AutoTokenizer
    from vsa_embed.data.corpus import build_corpus
    from vsa_embed.data.pubmed import iter_pubmed
    from vsa_embed.experiments.c3_corpus import iter_texts
    from vsa_embed.ontologies.mesh_novel import (MentionCounter, candidate_aliases, load_mesh_records, mention_key,
                                                 select_aliases)
    from vsa_embed.span_channel import AliasTable
    from collections import Counter
    descriptors, scrs = load_mesh_records(DATA / "mesh/desc2026.gz", DATA / "mesh/supp-2026/supp2026.gz")
    pairs, candidate_stats = candidate_aliases(descriptors, scrs, policy)
    domain, general_counter = Counter(), Counter()
    for alias, n in domain_counts.items():
        domain[mention_key(alias)] += int(n)
    for alias, n in general_counts.items():
        general_counter[mention_key(alias)] += int(n)
    rows, selection_stats = select_aliases(pairs, domain, general_counter, policy)
    uis = sorted({r["ui"] for r in rows})
    index = {ui: i for i, ui in enumerate(uis)}
    table = AliasTable.from_pairs([(r["alias"], index[r["ui"]]) for r in rows])
    counter = MentionCounter({r["key"] for r in rows})
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    texts = (r["text"] for r in iter_pubmed(sorted(Path(pubmed).glob("pubmed26n*.parquet"), reverse=True))
             if counter.mentions(r["text"]))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    build_corpus(texts, output / "eval", tokenizer_name=tokenizer_name, table=table, eos_id=tokenizer.eos_token_id,
                 max_tokens=int(eval_tokens), workers=workers, vocab_size=len(tokenizer))
    build_corpus(iter_texts([str(p) for p in general], limit=int(general_docs)), output / "eval-general",
                 tokenizer_name=tokenizer_name, table=table, eos_id=tokenizer.eos_token_id, max_tokens=10**9, workers=workers,
                 vocab_size=len(tokenizer))
    entries = len(table.entry_concepts)
    torch.save({"entry_count": entries, "heldout_entries": [], "train_frequency": [0] * entries,
                "alias_table_sha256": table.digest()}, output / "ontology.pt")
    summary = {"policy": repr(policy), "candidates": candidate_stats, "selection": selection_stats, "entries": entries,
               "aliases": len(table.alias_to_entry), "alias_table_sha256": table.digest()}
    (output / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# -- candidate table -----------------------------------------------------------------------------------------------

def candidate_rows(calibration: dict[str, Any], runs: Path, candidates: dict[str, Any], *,
                   host: str = "SmolLM2-360M") -> list[dict[str, Any]]:
    """The calibration tracks on `host` (with their measured novelty and gain) followed by every candidate of
    `candidates` (name → metadata; `run` = its P0 slice folder under `runs`, whose `proxies.json` holds the proxies)."""
    rows = [{"name": r["track"], "kind": "calibration", "novelty": r["novelty_mean"], "gain": r["gain_mean"],
             **{k: r[k] for k in PROXIES}} for r in calibration["rows"] if r["host"] == host]
    for name, meta in candidates.items():
        path = runs / meta["run"] / "proxies.json"
        proxies = json.loads(path.read_text()) if path.exists() else {}
        rows.append({"name": name, "kind": "candidate", **{k: v for k, v in meta.items() if k != "run"},
                     **{k: proxies.get(k) for k in (*PROXIES, "linked_spans", "linked_entries", "windows")}})
    return rows


def render_candidates(rows: Sequence[dict[str, Any]], reference: str = "t4") -> str:
    def num(value: Any, digits: int = 2) -> str:
        return "—" if value is None else f"{value:.{digits}f}"
    anchor = next((r for r in rows if r["name"] == reference), None)
    lines = ["# E9 novelty screen — candidate tracks (decision 55)", "",
             "P0 = SmolLM2-360M, evaluation only, 512 windows of each candidate's slice (calibration tracks: their full E9 P0 runs). "
             "Primary proxy: inside nats per linked span (orders T5 > T4 > T1 > WordNet at both host sizes; "
             f"`calibration/`); `vs {reference}` = its ratio to {reference.upper()}'s value. The general / domain rate is biased "
             "towards 0 for candidates whose domain text keeps only abstracts that mention a name (a denser domain), so it "
             "does not rank them against the calibration tracks; inside nats per span does not depend on density.", "",
             "| track / candidate | source | licence | entries | occurrences | inside nats / span | vs " + reference + " | inside loss | "
             "subtokens / span | general / domain | verdict |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        ratio = (r["inside_nats_per_span"] / anchor["inside_nats_per_span"]) if anchor and r.get("inside_nats_per_span") else None
        if r["kind"] == "calibration":
            source, licence, entries, occ = "finished E9 track", "—", "—", "—"
            verdict = f"novelty {100 * r['novelty']:+.1f}%, gain {100 * r['gain']:+.2f}%"
        else:
            source, licence = r.get("source", ""), r.get("licence", "")
            entries = f"{r['entries']:,}" if isinstance(r.get("entries"), int) else str(r.get("entries", "—"))
            occ = f"{r['occurrences']:,}" if isinstance(r.get("occurrences"), int) else str(r.get("occurrences", "—"))
            verdict = r.get("verdict", "")
        lines.append(f"| {r['name']} | {source} | {licence} | {entries} | {occ} | {num(r.get('inside_nats_per_span'))} | "
                     f"{num(ratio)} | {num(r.get('inside_loss'))} | {num(r.get('subtokens_per_span'))} | "
                     f"{num(r.get('general_to_domain'), 4)} | {verdict} |")
    return "\n".join(lines) + "\n"


# -- CLI -----------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    cal = sub.add_parser("calibrate", help="proxies vs novelty and gain on the finished tracks")
    cal.add_argument("--runs", type=Path, default=RUNS); cal.add_argument("--output", type=Path, required=True)
    cal.add_argument("--general-tokens", type=int, default=GENERAL_TOKENS)
    cfg = sub.add_parser("p0-config", help="write a P0 slice configuration for a candidate corpus")
    cfg.add_argument("--data-root", type=Path, required=True); cfg.add_argument("--name", required=True)
    cfg.add_argument("--eval-split", default="eval"); cfg.add_argument("--train-split", default=None)
    cfg.add_argument("--windows", type=int, default=512); cfg.add_argument("--host", default="SmolLM2-360M")
    cfg.add_argument("--eval-batch", type=int, default=4); cfg.add_argument("--output", type=Path, required=True)
    prox = sub.add_parser("proxies", help="the proxies of a finished P0 slice run")
    prox.add_argument("--run", type=Path, required=True); prox.add_argument("--general", type=Path, default=None)
    prox.add_argument("--general-tokens", type=int, default=GENERAL_TOKENS)
    prox.add_argument("--output", type=Path, default=None)
    sl = sub.add_parser("mesh-slice", help="a P0 slice of a MeSH candidate vocabulary (mesh_novel policy)")
    sl.add_argument("--policy", default="{}", help="JSON keys of mesh_novel.NovelVocabularyPolicy")
    sl.add_argument("--domain-counts", type=Path, required=True,
                    help='JSON {"mentions": {alias: count}} of the PubMed 2025-26 text (screening counts)')
    sl.add_argument("--general-counts", type=Path, required=True, help="the same for the general-text sample")
    sl.add_argument("--output", type=Path, required=True); sl.add_argument("--workers", type=int, default=4)
    summ = sub.add_parser("summarize", help="the candidate table: calibration tracks + every candidate's P0 slice proxies")
    summ.add_argument("--calibration", type=Path, required=True, help="calibration.json of `calibrate`")
    summ.add_argument("--runs", type=Path, required=True, help="folder of the candidates' P0 slice runs")
    summ.add_argument("--candidates", type=Path, required=True, help="YAML: name → {run, source, licence, entries, occurrences, verdict}")
    summ.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "mesh-slice":
        from vsa_embed.ontologies.mesh_novel import NovelVocabularyPolicy
        counts = [json.loads(Path(p).read_text()) for p in (args.domain_counts, args.general_counts)]
        result = mesh_slice(args.output, NovelVocabularyPolicy.from_config(json.loads(args.policy)),
                            domain_counts=counts[0].get("mentions", counts[0]), general_counts=counts[1].get("mentions", counts[1]),
                            workers=args.workers)
        print(json.dumps(result, indent=2))
        return
    if args.command == "summarize":
        rows = candidate_rows(json.loads(args.calibration.read_text()), args.runs, yaml.safe_load(args.candidates.read_text()))
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "candidates.json").write_text(json.dumps(rows, indent=2, default=float) + "\n")
        (args.output / "candidates.md").write_text(render_candidates(rows))
        print(render_candidates(rows))
        return
    if args.command == "calibrate":
        result = calibrate(args.runs, general_tokens=args.general_tokens)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "calibration.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
        (args.output / "calibration.md").write_text(render_calibration(result))
        print(render_calibration(result))
    elif args.command == "p0-config":
        config = p0_config(args.data_root, name=args.name, eval_split=args.eval_split, windows=args.windows, host=args.host,
                           eval_batch=args.eval_batch, train_split=args.train_split)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(yaml.safe_dump(config, sort_keys=False))
        print(args.output)
    else:
        result = run_proxies(args.run, general=args.general, general_tokens=args.general_tokens)
        text = json.dumps({k: v for k, v in result.items() if k != "strata"}, indent=2, default=float)
        if args.output:
            args.output.write_text(json.dumps(result, indent=2, default=float) + "\n")
        print(text)


if __name__ == "__main__":
    main()
