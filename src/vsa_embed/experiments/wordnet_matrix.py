"""Real matrix-only WordNet factorization pilot using an immutable host table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml
from nltk.corpus import wordnet as wn
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from vsa_embed.factorization import OntologyFactorizer, fit_factorizer, geometry_metrics, nearest_recipe_predictions
from vsa_embed.provenance import prepare_output_dir, write_run_metadata


@dataclass(frozen=True)
class Concept:
    synset: str
    lemma: str
    token_id: int
    values: tuple[str, ...]
    taxonomy_degree: int


def immutable_embedding_rows(embedding: nn.Embedding, token_ids: Tensor) -> Tensor:
    """Clone selected rows without retaining a mutable view into host weights."""
    before = embedding.weight.detach().clone()
    rows = embedding(token_ids).detach().clone()
    if not torch.equal(before, embedding.weight.detach()):
        raise RuntimeError("host embedding changed during extraction")
    return rows


def canonical_hypernym_path(synset: Any, depth: int) -> tuple[str, ...] | None:
    """Return a deterministic ancestor path or ``None`` when it is too short."""
    values = [f"lexname:{synset.lexname()}"]
    node = synset
    for _ in range(depth):
        parents = sorted(node.hypernyms() + node.instance_hypernyms(), key=lambda item: item.name())
        if not parents:
            return None
        node = parents[0]
        values.append(f"synset:{node.name()}")
    return tuple(values)


def collect_concepts(tokenizer: Any, *, max_concepts: int, ancestor_depth: int, seed: int) -> list[Concept]:
    """Align unambiguous single-token noun lemmas with WordNet recipes."""
    candidates: dict[int, list[Concept]] = defaultdict(list)
    all_senses: dict[int, set[str]] = defaultdict(set)
    encoded: dict[str, list[int]] = {}
    for synset in wn.all_synsets(pos=wn.NOUN):
        path = canonical_hypernym_path(synset, ancestor_depth)
        for lemma_obj in synset.lemmas():
            lemma = lemma_obj.name().lower()
            if not lemma.isalpha() or lemma != lemma_obj.name():
                continue
            ids = encoded.setdefault(lemma, tokenizer.encode(" " + lemma, add_special_tokens=False))
            if len(ids) != 1 or tokenizer.decode(ids).strip().lower() != lemma:
                continue
            all_senses[ids[0]].add(synset.name())
            if path is not None:
                degree = len(synset.hypernyms()) + len(synset.instance_hypernyms()) + len(synset.hyponyms())
                candidates[ids[0]].append(Concept(synset.name(), lemma, ids[0], path, degree))
    # One token must map to exactly one eligible sense; otherwise a static row
    # cannot be assigned a defensible synset target.
    unambiguous = [items[0] for token, items in candidates.items() if len(items) == 1 and len(all_senses[token]) == 1]
    # A synset can have several monosemous aliases with the same recipe. Keep
    # one deterministic token so an exact recipe twin cannot cross the split.
    by_synset: dict[str, list[Concept]] = defaultdict(list)
    for concept in unambiguous:
        by_synset[concept.synset].append(concept)
    synset_unique = [min(items, key=lambda item: (item.lemma, item.token_id)) for items in by_synset.values()]
    concepts = synset_unique
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(concepts), generator=generator).tolist()
    return [concepts[index] for index in order[:max_concepts]]


def encode_recipes(concepts: Iterable[Concept]) -> tuple[Tensor, list[str]]:
    concepts = list(concepts)
    vocabulary = sorted({value for concept in concepts for value in concept.values})
    ids = {value: index for index, value in enumerate(vocabulary)}
    return torch.tensor([[ids[value] for value in concept.values] for concept in concepts]), vocabulary


def supported_holdout(recipes: Tensor, holdout_count: int, seed: int) -> tuple[Tensor, Tensor]:
    """Create recipe-group-disjoint holdouts with global atomic support."""
    n, _ = recipes.shape
    if not 0 < holdout_count < n:
        raise ValueError("holdout_count must be between zero and node count")
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, recipe in enumerate(recipes.tolist()):
        groups[tuple(recipe)].append(index)
    counts = Counter(recipes.flatten().tolist())
    selected: list[int] = []
    keys = list(groups)
    order = torch.randperm(len(keys), generator=torch.Generator().manual_seed(seed)).tolist()
    for position in order:
        group = groups[keys[position]]
        if len(selected) + len(group) > holdout_count:
            continue
        removed = Counter(recipes[group].flatten().tolist())
        if all(counts[value] > amount for value, amount in removed.items()):
            selected.extend(group)
            counts.subtract(removed)
        if len(selected) == holdout_count:
            break
    if len(selected) != holdout_count:
        raise RuntimeError(f"only {len(selected)} atomically supported holdouts available")
    test = torch.tensor(sorted(selected))
    mask = torch.ones(n, dtype=torch.bool); mask[test] = False
    return mask.nonzero().flatten(), test


def _trained_prediction(
    method: str, recipes: Tensor, targets: Tensor, train: Tensor, test: Tensor,
    config: dict[str, Any], seed: int, device: torch.device,
) -> tuple[Tensor, dict[str, float], OntologyFactorizer]:
    algebra = "map" if method == "typed_map" else "real_hrr"
    typed = method != "additive"
    used = recipes.clone()
    if method == "shuffled_structure":
        generator = torch.Generator().manual_seed(seed + 991)
        for role in range(used.shape[1]):
            used[:, role] = used[torch.randperm(used.shape[0], generator=generator), role]
    torch.manual_seed(seed + 100)
    model = OntologyFactorizer(
        int(recipes.max()) + 1, recipes.shape[1], int(config["vsa_dimension"]), targets.shape[1],
        algebra=algebra, typed=typed,
    ).to(device)
    fit = fit_factorizer(
        model, used.to(device), targets.to(device), train.to(device), steps=int(config["steps"]),
        learning_rate=float(config["learning_rate"]), cosine_weight=float(config["cosine_weight"]),
    )
    with torch.no_grad():
        prediction = model(used[test].to(device)).cpu()
    return prediction, {"initial_loss": fit.initial_loss, "final_loss": fit.final_loss}, model.cpu()


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    git_at_start = prepare_output_dir(output_dir)
    model_id = config["host"]["model"]
    revision = config["host"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
    host = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, local_files_only=True)
    concepts = collect_concepts(
        tokenizer, max_concepts=int(config["data"]["max_concepts"]),
        ancestor_depth=int(config["data"]["ancestor_depth"]), seed=int(config["data"]["selection_seed"]),
    )
    if len(concepts) < int(config["data"]["max_concepts"]):
        raise RuntimeError(f"only {len(concepts)} aligned concepts found")
    token_ids = torch.tensor([concept.token_id for concept in concepts])
    targets = immutable_embedding_rows(host.get_input_embeddings(), token_ids).float()
    host_hash = hashlib.sha256(targets.numpy().tobytes()).hexdigest()
    del host
    recipes, vocabulary = encode_recipes(concepts)
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    saved: OntologyFactorizer | None = None
    methods = config["methods"]
    for seed in config["seeds"]:
        train, test = supported_holdout(recipes, int(config["data"]["holdout_count"]), int(seed))
        split_rows.extend({"seed": seed, "index": int(index), "partition": "train"} for index in train)
        split_rows.extend({"seed": seed, "index": int(index), "partition": "test"} for index in test)
        degree = torch.tensor([concepts[index].taxonomy_degree for index in test])
        low = degree <= degree.median()
        for method in methods:
            diagnostics: dict[str, float] = {}
            model = None
            if method == "train_mean":
                prediction = targets[train].mean(0).expand(test.numel(), -1)
            elif method == "nearest_recipe":
                prediction = nearest_recipe_predictions(recipes, targets, train, test)
            else:
                prediction, diagnostics, model = _trained_prediction(method, recipes, targets, train, test, config["factorization"], int(seed), device)
            for band, selector in (("all", torch.ones_like(low, dtype=torch.bool)), ("low_degree", low), ("high_degree", ~low)):
                if selector.any():
                    metrics = geometry_metrics(prediction[selector], targets[test][selector], k=int(config["evaluation"]["knn_k"]))
                    rows.append({"seed": seed, "method": method, "degree_band": band, "n": int(selector.sum()), **diagnostics, **metrics})
            if method == "typed_hrr" and saved is None:
                saved = model
    _write_csv(output_dir / "metrics.csv", rows)
    _write_csv(output_dir / "splits.csv", split_rows)
    all_rows = [row for row in rows if row["degree_band"] == "all"]
    means = {
        method: {
            metric: sum(float(row[metric]) for row in all_rows if row["method"] == method) / len(config["seeds"])
            for metric in ("row_cosine", "knn_overlap")
        } for method in methods
    }
    controls = [method for method in methods if method != "typed_hrr"]
    knn_control = max(controls, key=lambda method: means[method]["knn_overlap"])
    cosine_control = max(controls, key=lambda method: means[method]["row_cosine"])
    knn_gain = means["typed_hrr"]["knn_overlap"] - means[knn_control]["knn_overlap"]
    cosine_gain = means["typed_hrr"]["row_cosine"] - means[cosine_control]["row_cosine"]
    gate = knn_gain >= float(config["acceptance"]["min_knn_gain"]) and cosine_gain >= float(config["acceptance"]["min_row_cosine_gain"])
    concept_rows = [{"index": i, **concept.__dict__} for i, concept in enumerate(concepts)]
    _write_csv(output_dir / "concepts.csv", concept_rows)
    artifact = {
        "schema_version": 1, "model": model_id, "revision": revision, "host_rows_sha256": host_hash,
        "algebra": "real_hrr", "dimension": config["factorization"]["vsa_dimension"], "values": vocabulary,
        "state_dict": saved.state_dict() if saved is not None else {},
    }
    torch.save(artifact, output_dir / "factorizer.pt")
    write_run_metadata(output_dir, config, git_at_start=git_at_start, device="cpu", wordnet=wn.get_version(), host_rows_sha256=host_hash)
    lines = ["# Experiment 01 — GPT-2 / WordNet matrix-only pilot", "", f"Concepts: **{len(concepts)}**; host revision: `{revision}`; WordNet: **{wn.get_version()}**.", "", "| Method | Held-out row cosine | Held-out kNN overlap |", "|---|---:|---:|", *[f"| {method} | {means[method]['row_cosine']:.3f} | {means[method]['knn_overlap']:.3f} |" for method in methods], "", f"Typed-HRR kNN gain over **{knn_control}**: **{knn_gain:+.3f}**.", f"Typed-HRR row-cosine gain over **{cosine_control}**: **{cosine_gain:+.3f}**.", f"Pilot gate: **{'PASS' if gate else 'FAIL'}**.", "", "This is a real frozen embedding-matrix result, but still only a node-disjoint, monosemous single-token pilot. It is not a behavioral insertion result."]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    return {"gate_passed": gate, "knn_gain": knn_gain, "cosine_gain": cosine_gain, "concepts": len(concepts)}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(yaml.safe_load(args.config.read_text()), args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()