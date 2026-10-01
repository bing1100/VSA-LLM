"""Experiment E1 (D1): contextual composition against frozen contextual anchors.

Polysemous WordNet lemmas with ≥ 2 senses that each have ≥ `min_per_sense` SemCor sentences.
Target: the host's centred hidden state at the lemma's last subtoken in each sentence (layer chosen
by a target-only audit: the layer with the largest between-sense share of within-lemma variance on
training lemmas). Concept = the lemma's union frame over its WordNet senses (ontology frames).

Learners (all trained on training lemmas' training sentences):
- `m0_lemma`: static bundle of the union frame (one prediction per lemma);
- `m0_sense_oracle`: static bundle of the gold sense's own frame (upper bound, needs the sense);
- `m1_p0`: attentive composition without context;
- `m1_p1`: context query from the causal mean of the sentence's token embeddings (P1);
- `m1_p2`: context query from the host state at layer L/4 at the lemma (P2) — the pre-named primary;
- `m1_p2_<operator>`: operator ablation of the primary;
- `free_sense_p2`: attention over free per-sense vectors with the P2 query (Dasigi / KnowBERT style);
  it has no vectors for unseen lemmas and is reported on seen lemmas only.

Metrics on held-out lemmas (node-disjoint) and on seen lemmas' held-out sentences: within-lemma
variance explained `1 − ‖ŷ − y‖² / ‖y − ȳ_lemma‖²`, within-lemma context retrieval MRR (rank of the
instance's own target among the lemma's evaluation targets, mid-rank ties), and sense accuracy of
the attention mass per sense sub-frame vs the WordNet first-sense (MFS) baseline.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from nltk.corpus import wordnet as wn
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from vsa_embed.compose import FrameComposer, FrameSchedule
from vsa_embed.data.semcor import load_semcor
from vsa_embed.ontologies.wordnet import build_wordnet_ontology
from vsa_embed.provenance import prepare_output_dir, write_run_metadata
from vsa_embed.statistics import mean_confidence_interval, positive_ranks

POS_MAP = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}


def select_items(instances, *, min_per_sense: int, max_per_sense: int, max_senses: int, seed: int):
    by_lemma: dict[tuple[str, str], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for item in instances:
        if item.pos in POS_MAP:
            by_lemma[(item.lemma, item.pos)][item.synset].append(item)
    rng = random.Random(seed)
    lemmas = {}
    for key, senses in sorted(by_lemma.items()):
        frequent = {s: v for s, v in senses.items() if len(v) >= min_per_sense}
        if len(frequent) < 2:
            continue
        wn_senses = [s.name() for s in wn.synsets(key[0].replace(" ", "_"), pos=POS_MAP[key[1]])][:max_senses]
        if sum(s in frequent for s in wn_senses) < 2:
            continue
        chosen = []
        for synset in wn_senses:
            if synset in frequent:
                items = list(frequent[synset]); rng.shuffle(items)
                chosen += items[:max_per_sense]
        lemmas[key] = {"senses": wn_senses, "items": chosen}
    return lemmas


@torch.no_grad()
def host_states(model, tokenizer, items, layers: list[int], device, batch: int = 32):
    """Per item: states at the lemma's last subtoken for `layers`, P1 context mean, P2 state."""
    out = {layer: [] for layer in layers}
    p1, p2 = [], []
    embed = model.get_input_embeddings()
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        encoded = tokenizer([it.text for it in chunk], return_offsets_mapping=True, add_special_tokens=False,
                            padding=True, return_tensors="pt", truncation=True, max_length=256)
        positions = []
        for row, it in enumerate(chunk):
            offsets = encoded["offset_mapping"][row].tolist()
            index = max((i for i, (s, e) in enumerate(offsets) if s < it.end and e > it.start and e > s), default=0)
            positions.append(index)
        ids = encoded["input_ids"].to(device); mask = encoded["attention_mask"].to(device)
        hidden = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
        rows = torch.arange(len(chunk), device=device); pos = torch.tensor(positions, device=device)
        for layer in layers:
            out[layer].append(hidden[layer][rows, pos].float().cpu())
        tokens = embed(ids).float()
        window = torch.arange(ids.shape[1], device=device)[None] <= pos[:, None]
        window &= torch.arange(ids.shape[1], device=device)[None] > (pos[:, None] - 8)
        p1.append(((tokens * window[..., None]).sum(1) / window.sum(1, keepdim=True)).cpu())
        p2.append(hidden[max(1, len(hidden) // 4)][rows, pos].float().cpu())
    return {layer: torch.cat(v) for layer, v in out.items()}, torch.cat(p1), torch.cat(p2)


def sense_separability(states: torch.Tensor, lemma_ids: list[int], sense_ids: list[int]) -> float:
    """Between-sense share of within-lemma variance (target-only layer audit)."""
    lemma_ids, sense_ids = np.asarray(lemma_ids), np.asarray(sense_ids)
    within, between = 0.0, 0.0
    for lemma in np.unique(lemma_ids):
        rows = states[torch.from_numpy(lemma_ids == lemma)]
        senses = sense_ids[lemma_ids == lemma]
        centre = rows.mean(0)
        within += float(((rows - centre) ** 2).sum())
        for sense in np.unique(senses):
            part = rows[torch.from_numpy(senses == sense)]
            between += len(part) * float(((part.mean(0) - centre) ** 2).sum())
    return between / max(within, 1e-12)


class FreeSenseAttention(nn.Module):
    """Context attention over free per-(lemma, sense) vectors (no transfer to unseen lemmas)."""

    def __init__(self, sense_count: int, dimension: int, context_dimension: int, key_dimension: int = 32):
        super().__init__()
        self.values = nn.Parameter(torch.randn(sense_count, dimension) / dimension**0.5)
        self.keys = nn.Parameter(torch.randn(sense_count, key_dimension) / key_dimension**0.5)
        self.query = nn.Linear(context_dimension, key_dimension, bias=False)

    def forward(self, sense_lists: list[list[int]], context: torch.Tensor) -> torch.Tensor:
        rows = []
        q = self.query(context)
        for i, senses in enumerate(sense_lists):
            index = torch.tensor(senses, device=context.device)
            weights = torch.softmax(self.keys[index] @ q[i], 0)
            rows.append((weights[:, None] * self.values[index]).sum(0))
        return F.normalize(torch.stack(rows), dim=-1)


def evaluate(pred: torch.Tensor, target: torch.Tensor, lemma_of: list[int]) -> dict[str, float]:
    lemma_of = np.asarray(lemma_of)
    se, var, rr = 0.0, 0.0, []
    for lemma in np.unique(lemma_of):
        idx = torch.from_numpy(np.nonzero(lemma_of == lemma)[0])
        y, p = target[idx], pred[idx]
        se += float(((p - y) ** 2).sum()); var += float(((y - y.mean(0)) ** 2).sum())
        if len(idx) > 1:
            scores = F.normalize(p, dim=-1) @ F.normalize(y, dim=-1).T
            rr += (1 / positive_ranks(scores, torch.eye(len(idx), dtype=torch.bool), ties="mid")).tolist()
    return {"variance_explained": 1 - se / max(var, 1e-12), "context_mrr": float(np.mean(rr)) if rr else float("nan"),
            "cosine": float(F.cosine_similarity(pred, target).mean())}


def run_seed(config: dict[str, Any], data: dict[str, Any], seed: int, device) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    lemma_keys = sorted(data["lemmas"])
    rng.shuffle(lemma_keys)
    n_test = int(len(lemma_keys) * float(config["heldout_lemma_fraction"]))
    test_lemmas = set(lemma_keys[:n_test])
    items = data["items"]
    is_test_lemma = np.asarray([items[i]["lemma"] in test_lemmas for i in range(len(items))])
    held_sentence = np.asarray([rng.random() < float(config["heldout_sentence_fraction"]) for _ in items])
    train = np.nonzero(~is_test_lemma & ~held_sentence)[0]
    eval_sets = {"heldout_lemmas": np.nonzero(is_test_lemma)[0], "seen_lemmas_heldout_sentences": np.nonzero(~is_test_lemma & held_sentence)[0]}
    targets = data["targets"]
    centre = targets[torch.from_numpy(train)].mean(0)
    y = F.normalize(targets - centre, dim=-1)
    lemma_index = {k: i for i, k in enumerate(sorted(data["lemmas"]))}
    lemma_of = [lemma_index[it["lemma"]] for it in items]
    rows = []
    onto = data["ontology"]
    dim = int(config["dimension"]); host_dim = y.shape[1]
    learners = {"m0_lemma": ("bundle", "lemma", None, "hrr"), "m0_sense_oracle": ("bundle", "sense", None, "hrr"),
                "m1_p0": ("attentive", "lemma", None, "hrr"), "m1_p1": ("attentive", "lemma", "p1", "hrr"),
                "m1_p2": ("attentive", "lemma", "p2", "hrr")}
    for operator in config["operator_ablation"]:
        learners[f"m1_p2_{operator.replace(':', '_')}"] = ("attentive", "lemma", "p2", operator)
    for name, (mode, unit, query, operator) in learners.items():
        log(f"  learner {name}")
        schedule = data["lemma_schedule"] if unit == "lemma" else data["sense_schedule"]
        concept = torch.tensor([it["lemma_concept"] if unit == "lemma" else it["sense_concept"] for it in items])
        context = data[query] if query else None
        torch.manual_seed(seed + 100)
        composer = FrameComposer(schedule, len(onto.atomic_names), len(onto.relation_names), dim, operator=operator,
                                 mode=mode, key_dimension=32, context_dimension=context.shape[1] if context is not None else 0,
                                 output_dimension=host_dim).to(device)
        fit(composer, concept, context, y, train, config, seed, device)
        for split, index in eval_sets.items():
            with torch.no_grad():
                idx = torch.from_numpy(index)
                ctx = context[idx].to(device) if context is not None else None
                rows_, weights, edges = composer.compose(concept[idx].to(device), ctx, return_weights=True)
                pred = composer.projector(rows_).cpu()
            metrics = evaluate(pred, y[idx], [lemma_of[i] for i in index])
            if mode == "attentive" and unit == "lemma":
                metrics.update(sense_accuracy(weights.cpu(), edges.cpu(), concept[idx], index, items, data))
            rows.append({"seed": seed, "learner": name, "split": split, "n": len(index), **metrics})
    # free per-sense attention (seen lemmas only)
    torch.manual_seed(seed + 100)
    free = FreeSenseAttention(data["sense_count"], host_dim, data["p2"].shape[1]).to(device)
    sense_lists = [data["lemma_sense_ids"][it["lemma"]] for it in items]
    optimizer = torch.optim.Adam(free.parameters(), lr=float(config["learning_rate"]))
    g = torch.Generator().manual_seed(seed)
    for _ in range(int(config["steps"])):
        b = train[torch.randint(len(train), (int(config["batch"]),), generator=g).numpy()]
        pred = free([sense_lists[i] for i in b], data["p2"][torch.from_numpy(b)].to(device))
        loss = (1 - F.cosine_similarity(pred, y[torch.from_numpy(b)].to(device))).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    index = eval_sets["seen_lemmas_heldout_sentences"]
    with torch.no_grad():
        pred = free([sense_lists[i] for i in index], data["p2"][torch.from_numpy(index)].to(device)).cpu()
    rows.append({"seed": seed, "learner": "free_sense_p2", "split": "seen_lemmas_heldout_sentences", "n": len(index),
                 **evaluate(pred, y[torch.from_numpy(index)], [lemma_of[i] for i in index])})
    return rows


def fit(composer, concept, context, y, train, config, seed, device):
    optimizer = torch.optim.Adam(composer.parameters(), lr=float(config["learning_rate"]))
    g = torch.Generator().manual_seed(seed)
    for _ in range(int(config["steps"])):
        b = torch.from_numpy(train[torch.randint(len(train), (int(config["batch"]),), generator=g).numpy()])
        ctx = context[b].to(device) if context is not None else None
        pred = composer(concept[b].to(device), ctx)
        loss = F.mse_loss(pred, y[b].to(device)) + (1 - F.cosine_similarity(pred, y[b].to(device))).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()


def sense_accuracy(weights, edges, concepts, index, items, data) -> dict[str, float]:
    """Predicted sense = sub-frame with the largest attention mass; MFS = WordNet first sense."""
    owner = data["edge_owner"]                      # schedule edge → list of senses that own it
    degrees = data["lemma_schedule"].degrees[concepts]
    segments = torch.repeat_interleave(torch.arange(len(index)), degrees)
    correct = mfs = counted = 0
    for k, i in enumerate(index):
        gold = items[i]["synset"]
        senses = data["lemmas"][items[i]["lemma"]]["senses"]
        if gold not in senses:
            continue
        mass = Counter()
        for w, e in zip(weights[segments == k].tolist(), edges[segments == k].tolist()):
            for sense in owner[e]:
                mass[sense] += w / len(owner[e])
        predicted = max(senses, key=lambda s: (mass[s], -senses.index(s)))
        correct += predicted == gold; mfs += senses[0] == gold; counted += 1
    return {"sense_accuracy": correct / max(1, counted), "mfs_accuracy": mfs / max(1, counted), "sense_n": counted}


def log(message: str) -> None:
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def prepare(config: dict[str, Any], device) -> dict[str, Any]:
    log("loading SemCor")
    instances = load_semcor(Path(config["semcor_root"]).expanduser(), wn)
    lemmas = select_items(instances, min_per_sense=int(config["min_per_sense"]), max_per_sense=int(config["max_per_sense"]),
                          max_senses=int(config["max_senses"]), seed=int(config["selection_seed"]))
    log(f"selected {len(lemmas)} lemmas; building ontology")
    onto = build_wordnet_ontology(wn, max_atomics=int(config["max_atomics"]))
    concept_index = onto.concept_index
    lemma_frames, sense_frames, edge_owner, items = [], [], [], []
    sense_concept: dict[str, int] = {}
    lemma_sense_ids: dict[tuple[str, str], list[int]] = {}
    sense_counter = 0
    for key in sorted(lemmas):
        senses = lemmas[key]["senses"]
        union, owners = [], {}
        for synset in senses:
            frame = onto.frames[concept_index[synset]]
            if synset not in sense_concept:
                sense_concept[synset] = len(sense_frames); sense_frames.append(frame)
            for edge in frame:
                owners.setdefault(edge, []).append(synset)
        for edge, owner in owners.items():
            union.append(edge); edge_owner.append(owner)
        lemma_frames.append(union)
        lemma_sense_ids[key] = list(range(sense_counter, sense_counter + len(senses))); sense_counter += len(senses)
        for it in lemmas[key]["items"]:
            items.append({"lemma": key, "synset": it.synset, "lemma_concept": len(lemma_frames) - 1,
                          "sense_concept": sense_concept.get(it.synset, sense_concept[senses[0]]), "obj": it})
    tokenizer = AutoTokenizer.from_pretrained(config["host"], local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token; tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(config["host"], local_files_only=True, torch_dtype=torch.float32).to(device).eval()
    n_layers = model.config.num_hidden_layers
    layers = sorted({n_layers // 2, (3 * n_layers) // 4, n_layers})
    log(f"{len(items)} sentences; computing host states")
    states, p1, p2 = host_states(model, tokenizer, [it["obj"] for it in items], layers, device)
    log("host states done")
    # Target-only layer audit on a fixed 80% of lemmas (independent of every seed's split).
    ordered = sorted(lemmas)
    audit_lemmas = {k for i, k in enumerate(ordered) if i % 5}
    audit = [i for i, it in enumerate(items) if it["lemma"] in audit_lemmas]
    position = {k: i for i, k in enumerate(ordered)}
    synset_ids = {s: i for i, s in enumerate(sorted({it["synset"] for it in items}))}
    lemma_ids = [position[items[i]["lemma"]] for i in audit]
    sense_ids = [synset_ids[items[i]["synset"]] for i in audit]
    separability = {layer: sense_separability(states[layer][audit], lemma_ids, sense_ids) for layer in layers}
    layer = max(separability, key=separability.get)
    return {"lemmas": lemmas, "items": items, "targets": states[layer], "p1": p1, "p2": p2, "ontology": onto,
            "lemma_schedule": FrameSchedule.from_frames(lemma_frames), "sense_schedule": FrameSchedule.from_frames(sense_frames),
            "edge_owner": edge_owner, "sense_count": sense_counter, "lemma_sense_ids": lemma_sense_ids,
            "layer": layer, "separability": separability}


def summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    seeds = config["seeds"]
    def paired(a: str, b: str, metric: str, split: str = "heldout_lemmas"):
        get = lambda learner, seed: next(r[metric] for r in rows if r["learner"] == learner and r["seed"] == seed and r["split"] == split)
        return mean_confidence_interval([get(a, s) - get(b, s) for s in seeds])
    primary = "m1_p2"
    gains = {m: paired(primary, "m0_lemma", m) for m in ("variance_explained", "context_mrr")}
    sense = mean_confidence_interval([next(r["sense_accuracy"] - r["mfs_accuracy"] for r in rows if r["learner"] == primary
                                           and r["seed"] == s and r["split"] == "heldout_lemmas") for s in seeds])
    passes = all(g["ci_low"] is not None and g["ci_low"] > 0 for g in gains.values()) and sense["ci_low"] is not None and sense["ci_low"] > 0
    means = defaultdict(dict)
    for r in rows:
        key = (r["learner"], r["split"])
        for metric in ("variance_explained", "context_mrr", "cosine", "sense_accuracy", "mfs_accuracy"):
            if metric in r:
                means[key].setdefault(metric, []).append(r[metric])
    table = {f"{k[0]}|{k[1]}": {m: float(np.mean(v)) for m, v in d.items()} for k, d in means.items()}
    return {"primary": primary, "gain_over_m0_lemma": gains, "sense_accuracy_minus_mfs": sense,
            "gate_passed": passes, "means": table}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    git_at_start = prepare_output_dir(args.output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = prepare(config, device)
    rows = []
    for seed in config["seeds"]:
        log(f"seed {seed}")
        rows += run_seed(config, data, int(seed), device)
    summary = summarize(rows, config)
    summary.update(layer=data["layer"], separability=data["separability"], lemmas=len(data["lemmas"]), items=len(data["items"]))
    (args.output / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    ci = lambda v: f"{v['mean']:+.4f} [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]"
    lines = ["# E1 — contextual composition against frozen contextual anchors", "",
             f"Host {config['host']}, target layer {data['layer']} (separability {json.dumps({k: round(v, 3) for k, v in data['separability'].items()})}); "
             f"{len(data['lemmas'])} lemmas, {len(data['items'])} sentences, seeds {config['seeds']}.", "",
             "| Learner | Split | Variance explained | Context MRR | Cosine | Sense acc. | MFS acc. |", "|---|---|---:|---:|---:|---:|---:|"]
    for key, m in sorted(summary["means"].items()):
        learner, split = key.split("|")
        fmt = lambda name: f"{m[name]:.4f}" if name in m else "—"
        lines.append(f"| {learner} | {split} | {fmt('variance_explained')} | {fmt('context_mrr')} | {fmt('cosine')} | "
                     f"{fmt('sense_accuracy')} | {fmt('mfs_accuracy')} |")
    lines += ["", f"Primary `m1_p2` − `m0_lemma` on held-out lemmas: variance explained {ci(summary['gain_over_m0_lemma']['variance_explained'])}, "
              f"context MRR {ci(summary['gain_over_m0_lemma']['context_mrr'])}; sense accuracy − MFS {ci(summary['sense_accuracy_minus_mfs'])}.",
              f"**E1 gate: {'PASS' if summary['gate_passed'] else 'FAIL'}.**", ""]
    (args.output / "report.md").write_text("\n".join(lines))
    write_run_metadata(args.output, config, git_at_start=git_at_start, device=device)


if __name__ == "__main__":
    main()
