"""T1-open track tasks: item builders (no GPU) for the "open-clinical (MeSH + PubMed)" track.

Reads a T1 build (run folder + data root) and writes small JSONL item files plus `manifest.json`:

1. `mesh_tree_probe.jsonl` — MeSH tree-category probe. Each item is a descriptor linked (ℓ ≥ 2,
   whole word) in the PubMed evaluation split, labelled with its top-level tree categories
   (A anatomy … Z geographicals). Protocol: read the model's word state at the alias's last subtoken
   (the injection position) in the item's context; fit a multinomial linear probe on
   `probe_split == "train"` (seen descriptors, single category) and report accuracy on
   `test_seen` and on `test_heldout` (held-out descriptors, never linked in training: the
   channel composes them with zero update). Prompting form: rank the 16 category names by
   likelihood after the template.
2. `pubmedqa_labeled.jsonl` — PubMedQA pqa_labeled (1,000 expert-labelled yes/no/maybe items,
   MIT licence) with the official 500-item test flag and 10 seeded CV folds over the rest. Prompting
   (cloze) form: score " yes" / " no" / " maybe" after `{context}\nQuestion: {question}\nAnswer:`;
   linear form: probe on the final-token state. BioASQ is not openly downloadable (registration).
3. `rare_neighbours.jsonl` + `rare_neighbours_preregistration.json` — the E5.3 neighbour study:
   40 rare descriptors (20 held-out, 20 seen with training frequency 1–9 at ℓ ≥ 2), chosen by a
   pre-registered rule and seed, each with its gold MeSH neighbourhood (parents, siblings,
   children, see-also, pharmacological actions) for the judge rubric.

No abstract text is committed (NLM: some abstracts may be protected by copyright): contexts are
pointers (PMID, character offsets, sentence sha256) that `resolve_contexts` fills from the extracted
PubMed parquet, and PubMedQA contexts are joined back from the pinned parquet by
`pubmedqa_with_context`. Train frequencies come from the build's `ontology.pt`; a slice build's
items are marked provisional and are regenerated from the full build with the same rule and seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

from vsa_embed.data.pubmed import iter_pubmed, pmid_bucket, text_paths
from vsa_embed.experiments.t1_open_corpus import TRACK_LABEL, build_track_ontology, load_config, pubmed_names
from vsa_embed.span_channel import AliasTable, CausalLinker

CATEGORIES = {
    "A": "Anatomy", "B": "Organisms", "C": "Diseases", "D": "Chemicals and Drugs",
    "E": "Analytical, Diagnostic and Therapeutic Techniques, and Equipment", "F": "Psychiatry and Psychology",
    "G": "Phenomena and Processes", "H": "Disciplines and Occupations",
    "I": "Anthropology, Education, Sociology, and Social Phenomena", "J": "Technology, Industry, and Agriculture",
    "K": "Humanities", "L": "Information Science", "M": "Named Groups", "N": "Health Care",
    "V": "Publication Characteristics", "Z": "Geographicals",
}
NEIGHBOUR_CATEGORIES = tuple("ABCDEFGHIJKLMN")   # concept-bearing branches (not V, Z)
NEIGHBOUR_RUBRIC = {
    "question": "How related is the candidate concept to the target concept in biomedicine?",
    "scale": {"strongly related": "same kind of entity in the same clinical/biological context, or a direct "
                                  "taxonomic neighbour (parent, child, sibling)",
              "related": "a clear but indirect relationship (shared system, mechanism, treatment or field)",
              "unrelated": "no meaningful relationship"},
    "unit": "for each target: the model's top-5 nearest descriptors (by channel/word-state cosine), each rated",
    "endpoint": "number of targets (of 40) whose top neighbours are rated strongly related by majority; "
                "one-tailed Fisher test channel vs C0/C2 (as the paper's 28/40 vs 4/40)",
    "gold": "the MeSH neighbourhood below is reference for the judge, never shown to the model",
}


def frequency_bin(frequency: int, heldout: bool) -> str:
    if heldout:
        return "heldout"
    if frequency == 0:
        return "unseen"
    return "rare" if frequency <= 9 else "mid" if frequency <= 99 else "frequent"


def sentence_bounds(text: str, start: int, end: int, limit: int = 300) -> tuple[int, int]:
    """[left, right) of the sentence containing [start, end) in `text` (see `sentence_around`)."""
    sentence, s, _ = sentence_around(text, start, end, limit)
    return start - s, start - s + len(sentence)


def sentence_around(text: str, start: int, end: int, limit: int = 300) -> tuple[str, int, int]:
    """The sentence containing [start, end) (split after '. ', '? ', '! ' and at newlines), cut to
    `limit` characters around the span; returns (sentence, span start, span end) in its coordinates."""
    left, right = 0, len(text)
    for separator in (". ", "? ", "! ", "\n"):
        before = text.rfind(separator, 0, start)
        if before >= 0:
            left = max(left, before + len(separator))
        after = text.find(separator, end)
        if after >= 0:
            right = min(right, after + (0 if separator == "\n" else 1))
    if right - left > limit:
        half = max(0, (limit - (end - start)) // 2)
        left, right = max(left, start - half), min(right, end + half)
    while left < start and text[left].isspace():
        left += 1
    return text[left:right].rstrip(), start - left, end - left


def eval_occurrences(paths: list[Path], *, eval_buckets: int, limit: int | None, table: AliasTable, tokenizer: Any,
                     min_subtokens: int = 2, per_entry: int = 5) -> dict[int, list[dict[str, Any]]]:
    """Whole-word linked occurrences (ℓ ≥ `min_subtokens`) per entry in the PubMed evaluation split
    (same documents and order as the build's `eval-pubmed`), up to `per_entry` from distinct PMIDs."""
    linker = CausalLinker(table, min_subtokens=min_subtokens)
    found: dict[int, list[dict[str, Any]]] = {}
    seen: set[int] = set()
    produced = 0
    for record in iter_pubmed(paths):
        pmid = int(record["pmid"])
        if pmid in seen or pmid_bucket(pmid) >= eval_buckets:
            continue
        seen.add(pmid)
        if limit is not None and produced >= limit:
            break
        produced += 1
        text = record["text"]
        offsets = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
        for span in linker.link(text, offsets):
            start, end = offsets[span.start_token][0], offsets[span.end_token][1]
            while start < end and text[start].isspace():
                start += 1
            if end < len(text) and (text[end].isalnum() or text[end] == "_"):
                continue                       # inside a longer word ("bank" in "banking")
            rows = found.setdefault(span.entry, [])
            if len(rows) < per_entry and all(r["pmid"] != pmid for r in rows):
                rows.append({"pmid": pmid, "start": start, "end": end, "text": text})
    return found


def _context(occurrence: dict[str, Any]) -> dict[str, Any]:
    """A pointer into the extracted PubMed text (no abstract text is committed; NLM: some abstracts
    may be protected by copyright): PMID, alias and sentence character offsets, sentence sha256."""
    text, start, end = occurrence["text"], occurrence["start"], occurrence["end"]
    left, right = sentence_bounds(text, start, end)
    return {"pmid": occurrence["pmid"], "alias": text[start:end], "start": start, "end": end, "sentence": [left, right],
            "sentence_sha256": hashlib.sha256(text[left:right].encode()).hexdigest()}


def resolve_contexts(items: list[dict[str, Any]], paths: list[Path]) -> list[dict[str, Any]]:
    """Fill `context["text"]` (the sentence) and `context["span"]` (alias offsets in it) from the
    extracted PubMed parquet files, verifying each sentence's sha256."""
    needed = {item["context"]["pmid"] for item in items}
    texts = {}
    for record in iter_pubmed(paths):
        if int(record["pmid"]) in needed:
            texts[int(record["pmid"])] = record["text"]
    resolved = []
    for item in items:
        context = dict(item["context"])
        text = texts[context["pmid"]]
        left, right = context["sentence"]
        sentence = text[left:right]
        if hashlib.sha256(sentence.encode()).hexdigest() != context["sentence_sha256"]:
            raise ValueError(f"PMID {context['pmid']}: sentence differs from the one the item was built on")
        context.update(text=sentence, span=[context["start"] - left, context["end"] - left])
        resolved.append({**item, "context": context})
    return resolved


def tree_probe_items(*, occurrences: dict[int, list[dict[str, Any]]], table: AliasTable, metadata: dict[str, Any],
                     concept_names: list[str], heldout: set[int], frequency: np.ndarray, seed: int,
                     seen_items: int, heldout_items: int, train_fraction: float = 0.8) -> list[dict[str, Any]]:
    """Up to `heldout_items` held-out and `seen_items` seen descriptors (seeded uniform samples);
    seen ones split `train_fraction` / rest into `train` / `test_seen`."""
    rng = np.random.default_rng(seed)
    trees, headings = metadata["trees"], metadata["headings"]
    rows = []
    for entry in sorted(occurrences):
        concepts = table.entry_concepts[entry]
        if len(concepts) != 1:
            continue                           # polysemous surface forms have no single label
        concept = concepts[0]
        categories = sorted({t[0] for t in trees[concept]})
        if not categories:
            continue
        rows.append((entry, concept, categories))
    held_pool = [r for r in rows if r[0] in heldout]
    keep = sorted(rng.choice(len(held_pool), size=min(heldout_items, len(held_pool)), replace=False).tolist()) if held_pool else []
    held_rows = [held_pool[i] for i in keep]
    seen_pool = [r for r in rows if r[0] not in heldout and frequency[r[0]] > 0]
    pick = sorted(rng.choice(len(seen_pool), size=min(seen_items, len(seen_pool)), replace=False).tolist()) if seen_pool else []
    seen_rows = [seen_pool[i] for i in pick]
    items = []
    for entry, concept, categories in held_rows + seen_rows:
        is_held = entry in heldout
        split = "test_heldout" if is_held else ("train" if rng.random() < train_fraction else "test_seen")
        occurrence = occurrences[entry]
        items.append({
            "id": f"tree-{concept_names[concept]}", "ui": concept_names[concept], "heading": headings[concept],
            "entry": int(entry), "label": categories[0], "categories": categories, "single_category": len(categories) == 1,
            "status": "heldout" if is_held else "seen", "train_frequency": int(frequency[entry]),
            "frequency_bin": frequency_bin(int(frequency[entry]), is_held), "probe_split": split,
            "context": _context(occurrence[0]),
            "more_occurrences": [[o["pmid"], o["start"], o["end"]] for o in occurrence[1:]],
        })
    return items


def _pubmedqa_context(row: dict[str, Any]) -> str:
    contexts = row["context"]
    labels = contexts.get("labels") or [None] * len(contexts["contexts"])
    return "\n".join(f"{label.capitalize() if label and label.isupper() else label}: {text}" if label else text
                     for label, text in zip(labels, contexts["contexts"]))


def pubmedqa_items(parquet: Path, ground_truth: Path | None, *, seed: int, folds: int = 10) -> list[dict[str, Any]]:
    """Item index (no abstract text): pubid, question, answer, official-test flag, CV fold, and the
    sha256 of the context that `pubmedqa_with_context` joins back from the pinned parquet."""
    table = pq.read_table(parquet).to_pylist()
    test = set(json.loads(ground_truth.read_text())) if ground_truth and ground_truth.exists() else set()
    items = []
    for row in sorted(table, key=lambda r: int(r["pubid"])):
        pubid = str(row["pubid"])
        official_test = pubid in test
        fold = None if official_test else int(hashlib.sha256(f"{seed}:{pubid}".encode()).hexdigest(), 16) % folds
        items.append({"id": f"pubmedqa-{pubid}", "pubid": int(pubid), "question": row["question"], "answer": row["final_decision"],
                      "official_test": official_test, "fold": fold,
                      "context_sha256": hashlib.sha256(_pubmedqa_context(row).encode()).hexdigest()})
    return items


def pubmedqa_with_context(items: list[dict[str, Any]], parquet: Path) -> list[dict[str, Any]]:
    """Join `context` (labelled abstract sections) and `long_answer` from the pinned parquet, verified."""
    rows = {int(r["pubid"]): r for r in pq.read_table(parquet).to_pylist()}
    joined = []
    for item in items:
        row = rows[item["pubid"]]
        context = _pubmedqa_context(row)
        if hashlib.sha256(context.encode()).hexdigest() != item["context_sha256"]:
            raise ValueError(f"PubMedQA {item['pubid']}: context differs from the pinned revision")
        joined.append({**item, "context": context, "long_answer": row["long_answer"]})
    return joined


def neighbourhood(concept: int, metadata: dict[str, Any], concept_names: list[str], *, cap: int = 25) -> dict[str, Any]:
    index = {ui: i for i, ui in enumerate(concept_names)}
    parents = metadata["parents"]
    children: dict[str, list[str]] = {}
    for i, ps in enumerate(parents):
        for p in ps:
            children.setdefault(p, []).append(concept_names[i])
    ui = concept_names[concept]
    headings = metadata["headings"]

    def named(uis: list[str]) -> list[dict[str, str]]:
        return [{"ui": u, "heading": headings[index[u]]} for u in sorted(set(uis))[:cap] if u in index]

    siblings = sorted({c for p in parents[concept] for c in children.get(p, []) if c != ui})
    return {"parents": named(parents[concept]), "siblings": named(siblings), "siblings_total": len(siblings),
            "children": named(children.get(ui, [])), "see_also": named(metadata["related"][concept]),
            "pharmacological_actions": named(metadata["actions"][concept]), "tree_numbers": metadata["trees"][concept]}


def rare_neighbour_items(*, occurrences: dict[int, list[dict[str, Any]]], table: AliasTable, metadata: dict[str, Any],
                         concept_names: list[str], heldout: set[int], frequency: np.ndarray, entry_length: dict[int, int],
                         seed: int, per_stratum: int = 20) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pre-registered rule: pool = single-descriptor entries with an alias of ≥ 2 GPT-2 subtokens, at
    least one whole-word ℓ ≥ 2 occurrence in the PubMed evaluation split, at least one parent, and
    every tree number in A–N; strata = held-out, and seen with training frequency 1–9 (ℓ ≥ 2);
    `per_stratum` drawn uniformly without replacement from each stratum's pool sorted by UI, with
    `numpy.random.default_rng(seed)`, held-out first."""
    pools: dict[str, list[int]] = {"heldout": [], "rare": []}
    for entry in sorted(occurrences, key=lambda e: concept_names[table.entry_concepts[e][0]]):
        concepts = table.entry_concepts[entry]
        if len(concepts) != 1 or entry_length.get(entry, 0) < 2:
            continue
        concept = concepts[0]
        trees = metadata["trees"][concept]
        if not trees or not metadata["parents"][concept] or any(t[0] not in NEIGHBOUR_CATEGORIES for t in trees):
            continue
        if entry in heldout:
            pools["heldout"].append(entry)
        elif 1 <= frequency[entry] <= 9:
            pools["rare"].append(entry)
    rng = np.random.default_rng(seed)
    items = []
    for stratum in ("heldout", "rare"):
        pool = pools[stratum]
        chosen = rng.choice(len(pool), size=min(per_stratum, len(pool)), replace=False).tolist() if pool else []
        for i in chosen:
            entry = pool[i]
            concept = table.entry_concepts[entry][0]
            items.append({"id": f"neighbour-{concept_names[concept]}", "ui": concept_names[concept],
                          "heading": metadata["headings"][concept], "entry": int(entry), "stratum": stratum,
                          "train_frequency": int(frequency[entry]), "context": _context(occurrences[entry][0]),
                          "gold": neighbourhood(concept, metadata, concept_names),
                          "scope_note_for_judge": metadata["scope_notes"][concept]})
    return items, {"pool_sizes": {k: len(v) for k, v in pools.items()}}


def _write_jsonl(path: Path, items: list[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in items)
    path.write_text(payload)
    return hashlib.sha256(payload.encode()).hexdigest()


def build_items(config: dict[str, Any], run_dir: Path, out_dir: Path, *, seen_items: int = 1200,
                heldout_items: int = 800) -> dict[str, Any]:
    data_root = Path(config["paths"]["data_root"]).expanduser()
    onto_pt = torch.load(data_root / "ontology.pt", weights_only=False)
    summary = json.loads((run_dir / "summary.json").read_text())
    ontology = build_track_ontology(config["ontology"])
    index = ontology.concept_index
    held_names = (run_dir / "holdout_concepts.txt").read_text().split()
    table = AliasTable.from_pairs(ontology.alias_pairs, holdout=[index[n] for n in held_names], include_holdout=True)
    if table.digest() != onto_pt["alias_table_sha256"]:
        raise ValueError("alias table differs from the build's ontology.pt (different ontology or holdout)")
    heldout = set(onto_pt["heldout_entries"])
    frequency = np.asarray(onto_pt["train_frequency"])
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], local_files_only=True)
    lengths = {a: len(tokenizer.encode(" " + a, add_special_tokens=False)) for a in table.alias_to_entry}
    entry_length: dict[int, int] = {}
    for alias, entry in table.alias_to_entry.items():
        entry_length[entry] = max(entry_length.get(entry, 0), lengths[alias])
    paths = text_paths(pubmed_names(config), Path(config["paths"]["pubmed_text"]).expanduser())
    occurrences = eval_occurrences(paths, eval_buckets=int(config["pubmed"]["eval_buckets"]),
                                   limit=config["data"].get("eval_pubmed_docs"), table=table, tokenizer=tokenizer)
    gpt2_train = summary["corpora"][config["tokenizer"]]["corpora"]["train"]["tokens"]
    reference_tokens = int(config["data"].get("reference_train_tokens") or config["data"]["train_tokens"])
    provisional = gpt2_train < 0.99 * reference_tokens
    seed = int(config["seed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    probe = tree_probe_items(occurrences=occurrences, table=table, metadata=ontology.metadata, concept_names=ontology.concept_names,
                             heldout=heldout, frequency=frequency, seed=seed, seen_items=seen_items, heldout_items=heldout_items)
    probe_sha = _write_jsonl(out_dir / "mesh_tree_probe.jsonl", probe)

    qa_cfg = config.get("pubmedqa")
    qa_info = None
    if qa_cfg:
        local = Path(config["paths"]["pubmedqa"]).expanduser()
        qa = pubmedqa_items(local / qa_cfg["file"], local / "test_ground_truth.json", seed=seed)
        qa_info = {"file": "pubmedqa_labeled.jsonl", "sha256": _write_jsonl(out_dir / "pubmedqa_labeled.jsonl", qa),
                   "items": len(qa), "official_test": sum(i["official_test"] for i in qa),
                   "answers": dict(Counter(i["answer"] for i in qa)),
                   "source": f"huggingface.co/datasets/{qa_cfg['repo']}@{qa_cfg['revision']} ({qa_cfg['file']}), MIT licence; "
                             f"official test ids from {qa_cfg.get('test_ground_truth_url')}",
                   "context": "joined from the pinned parquet by t1_items.pubmedqa_with_context (sha256-verified)",
                   "prompt": "{context}\nQuestion: {question}\nAnswer:", "candidates": [" yes", " no", " maybe"],
                   "overlap_with_training_pmids": 0}

    neighbours, pools = rare_neighbour_items(occurrences=occurrences, table=table, metadata=ontology.metadata,
                                             concept_names=ontology.concept_names, heldout=heldout, frequency=frequency,
                                             entry_length=entry_length, seed=seed)
    neighbour_sha = _write_jsonl(out_dir / "rare_neighbours.jsonl", neighbours)
    prereg = {
        "study": "E5.3 rare-concept neighbour study, T1-open", "track": TRACK_LABEL,
        "rule": " ".join(rare_neighbour_items.__doc__.split("Pre-registered rule: ", 1)[1].split()),
        "seed": seed, "per_stratum": 20, "pool_sizes": pools["pool_sizes"], "rubric": NEIGHBOUR_RUBRIC,
        "selected": [i["ui"] for i in neighbours], "items_sha256": neighbour_sha,
        "holdout_sha256": onto_pt["holdout_sha256"], "alias_table_sha256": onto_pt["alias_table_sha256"],
        "train_frequency": "measured on the full training corpus" if not provisional
        else f"PROVISIONAL: slice build ({gpt2_train:,} training tokens); regenerate from the full build",
        "judge": "headless claude -p (judging.claude_cli_runner), model pinned; >= 3 calls x >= 2 paraphrases per item",
    }
    (out_dir / "rare_neighbours_preregistration.json").write_text(json.dumps(prereg, indent=2, ensure_ascii=False) + "\n")

    labels = Counter(i["label"] for i in probe if i["single_category"])
    manifest = {
        "track": TRACK_LABEL, "build": str(run_dir), "provisional": provisional,
        "holdout_sha256": onto_pt["holdout_sha256"], "alias_table_sha256": onto_pt["alias_table_sha256"],
        "mesh_tree_probe": {"file": "mesh_tree_probe.jsonl", "sha256": probe_sha, "items": len(probe),
                            "by_split": dict(Counter(i["probe_split"] for i in probe)),
                            "single_category": sum(i["single_category"] for i in probe),
                            "labels_single_category": dict(sorted(labels.items())), "categories": CATEGORIES,
                            "readout": "hidden state at the alias's last subtoken (injection position) in the context sentence",
                            "contexts": "pointers (PMID, character offsets, sentence sha256) into the extracted PubMed parquet; "
                                        "t1_items.resolve_contexts fills in the sentence (no abstract text is committed)",
                            "probe": "multinomial logistic regression on single-category train items; "
                                     "accuracy on test_seen and test_heldout; paired bootstrap between conditions"},
        "pubmedqa": qa_info,
        "rare_neighbours": {"file": "rare_neighbours.jsonl", "sha256": neighbour_sha, "items": len(neighbours),
                            "preregistration": "rare_neighbours_preregistration.json"},
        "not_built": {"bioasq": "requires registration (not openly downloadable)",
                      "mimic_icd_coding": "MIMIC-IV not available (author decision 2026-10-02: reported as not run)"},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True, help="T1 build run folder (summary.json, holdout_concepts.txt)")
    parser.add_argument("--out", type=Path, default=Path("experiments/t1-open-clinical/items"))
    parser.add_argument("--seen-items", type=int, default=1200)
    parser.add_argument("--heldout-items", type=int, default=800)
    args = parser.parse_args(argv)
    print(json.dumps(build_items(load_config(args.config), args.run, args.out, seen_items=args.seen_items,
                                 heldout_items=args.heldout_items), indent=2)[:3000])


if __name__ == "__main__":
    main()
