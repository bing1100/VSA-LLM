"""E10.1 — the WordNet world for the E10 runner: WordNet frames, frozen GPT-2 multi-template anchors.

Concepts are the E2/E3 selection (single-token-aligned synsets with at least one pointer edge, same
selection seed), frames are the WordNet frame ontology (`build_wordnet_ontology`), and each concept is
*observed* through several prompt templates around its gloss and term; the frozen host's final hidden
state at the term's last token in each template is one observation (E2's anchor is template 0). The
observations are centred on the training concepts' mean and L2-normalised.

Two steps:

1. `anchors` (GPU, queueable): compute and cache the multi-template anchors
   (`python -m vsa_embed.experiments.e10_wordnet anchors --config <cfg> --output <file.pt>`);
2. the E10 runner with `world: {kind: wordnet, anchors: <file.pt>, ...}` builds the world from the cache
   (`build_wordnet_world`); without a cache it computes the anchors in-process.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Hashable

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from vsa_embed.synthetic_ontology import OntologyWorld

TEMPLATES = (
    "Definition: {gloss} Term: {term}",
    "{gloss}. This is called {term}",
    "Meaning: {gloss}. Word: {term}",
    "Q: What do we call this: {gloss}? A: {term}",
    "Gloss: {gloss}\nLemma: {term}",
    "In other words, {gloss}: {term}",
    "The dictionary says \"{gloss}\" for the word {term}",
    "{gloss} -- {term}",
)

GOLD_PROPERTIES = {
    "hypernym": ("antisymmetric",), "instance_hypernym": ("antisymmetric",),
    "part_meronym": ("inverse_of:part_holonym",), "part_holonym": ("inverse_of:part_meronym",),
    "member_meronym": ("inverse_of:member_holonym",), "member_holonym": ("inverse_of:member_meronym",),
    "substance_meronym": ("inverse_of:substance_holonym",), "substance_holonym": ("inverse_of:substance_meronym",),
    "antonym": ("symmetric",), "similar_to": ("symmetric",),
}


def select_concepts(onto: Any, wordnet: Any, tokenizer: Any, *, max_concepts: int, selection_seed: int) -> tuple[list[int], list[str], list[str]]:
    """The E2 WordNet selection (same rule and seed as `e2_frontier.wordnet_dataset`)."""
    from vsa_embed.wordnet_relations import aligned_lemma
    candidates = [cid for cid in range(len(onto.concept_names))
                  if sum(1 for r, _ in onto.frames[cid] if onto.relation_names[r] not in {"lexname", "pos"}) >= 1]
    rng = np.random.default_rng(selection_seed)
    rng.shuffle(candidates)
    chosen, glosses, terms = [], [], []
    for cid in candidates:
        synset = wordnet.synset(onto.concept_names[cid])
        match = aligned_lemma(synset, tokenizer)
        if match is None:
            continue
        chosen.append(cid); glosses.append(synset.definition()); terms.append(match[0])
        if len(chosen) >= max_concepts:
            break
    return chosen, glosses, terms


@torch.no_grad()
def template_anchors(host: Any, tokenizer: Any, glosses: list[str], terms: list[str], templates: tuple[str, ...], *,
                     device: torch.device, max_length: int = 64, batch_size: int = 64) -> torch.Tensor:
    """(concepts, templates, host_dim) final hidden states at the term's last token."""
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    host = host.to(device).eval()
    out = []
    for template in templates:
        prompts = []
        for gloss, term in zip(glosses, terms):
            frame = template.format(gloss="{g}", term=term)
            budget = max(4, max_length - len(tokenizer.encode(frame.replace("{g}", ""), add_special_tokens=False)))
            ids = tokenizer.encode(gloss, add_special_tokens=False)
            short = gloss if len(ids) <= budget else tokenizer.decode(ids[:budget])
            prompts.append(frame.replace("{g}", short))
        rows = []
        for start in range(0, len(prompts), batch_size):
            encoded = tokenizer(prompts[start:start + batch_size], padding=True, truncation=True,
                                max_length=max_length + 24, return_tensors="pt").to(device)
            hidden = host(**encoded, output_hidden_states=True).hidden_states[-1]
            last = encoded["attention_mask"].sum(1) - 1
            rows.append(hidden[torch.arange(hidden.shape[0], device=device), last].float().cpu())
        out.append(torch.cat(rows))
    return torch.stack(out, 1)


def compute_anchors(spec: dict[str, Any]) -> dict[str, Any]:
    from nltk.corpus import wordnet as wn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vsa_embed.ontologies.wordnet import build_wordnet_ontology
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(spec.get("host", "gpt2"), local_files_only=True)
    host = AutoModelForCausalLM.from_pretrained(spec.get("host", "gpt2"), local_files_only=True)
    onto = build_wordnet_ontology(wn, max_atomics=int(spec.get("max_atomics", 8192)))
    chosen, glosses, terms = select_concepts(onto, wn, tokenizer, max_concepts=int(spec.get("max_concepts", 6000)),
                                             selection_seed=int(spec.get("selection_seed", 5)))
    templates = tuple(spec.get("templates", TEMPLATES))
    anchors = template_anchors(host, tokenizer, glosses, terms, templates, device=device)
    return {"concept_names": [onto.concept_names[c] for c in chosen], "terms": terms, "templates": list(templates),
            "anchors": anchors.half(), "host": spec.get("host", "gpt2"), "wordnet_version": wn.get_version(),
            "selection_seed": int(spec.get("selection_seed", 5)), "max_atomics": int(spec.get("max_atomics", 8192))}


def build_wordnet_world(spec: dict[str, Any], seed: int) -> OntologyWorld:
    from nltk.corpus import wordnet as wn
    from vsa_embed.ontologies.wordnet import build_wordnet_ontology
    path = spec.get("anchors")
    cache = torch.load(Path(path).expanduser(), weights_only=False) if path and Path(path).expanduser().exists() \
        else compute_anchors(spec)
    onto = build_wordnet_ontology(wn, max_atomics=int(cache.get("max_atomics", spec.get("max_atomics", 8192))))
    index = onto.concept_index
    names = cache["concept_names"][: int(spec.get("max_concepts", len(cache["concept_names"])))]
    anchors = cache["anchors"][: len(names)].float()
    concept_ids = [index[n] for n in names]
    local = {cid: i for i, cid in enumerate(concept_ids)}
    heads, relations, fillers = [], [], []
    for i, cid in enumerate(concept_ids):
        for r, a in onto.frames[cid]:
            heads.append(i); relations.append(r); fillers.append(a)
    n = len(concept_ids)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    fractions = spec.get("split_fractions", [0.1, 0.1, 0.1])
    counts = [int(round(f * n)) for f in fractions]
    val, test = order[:counts[0]], order[counts[0]:counts[0] + counts[1]]
    new = order[counts[0] + counts[1]:sum(counts)]
    train = order[sum(counts):]
    splits = {k: torch.from_numpy(np.sort(v)).long() for k, v in
              (("train", train), ("validation", val), ("test", test), ("new_word", new))}
    observations = anchors - anchors[splits["train"]].reshape(-1, anchors.shape[-1]).mean(0)
    observations = F.normalize(observations, dim=-1)
    t = anchors.shape[1]
    obs_split = spec.get("observation_split") or {"train": list(range(t // 2)), "val": list(range(t // 2, 3 * t // 4)),
                                                  "audit": list(range(3 * t // 4, t)), "infer": list(range(3 * t // 4)),
                                                  "eval": list(range(3 * t // 4, t))}
    node_of_atomic: list[Hashable] = [a[len("synset:"):] if a.startswith("synset:") else a for a in onto.atomic_names]
    levels = torch.tensor([wn.synset(nm).min_depth() for nm in names])
    probes = {"isa": "hypernym", "transitive": None, "inverse": ["part_meronym", "part_holonym"], "symmetric": "antonym"}
    return OntologyWorld(
        name="wordnet", concept_count=n, atomic_count=len(onto.atomic_names), relation_names=list(onto.relation_names),
        heads=torch.tensor(heads), relations=torch.tensor(relations), fillers=torch.tensor(fillers),
        teacher_weights=torch.ones(len(heads)), filler_node=node_of_atomic, concept_node=list(names),
        concept_level=levels, observations=observations, observation_sources=torch.full(observations.shape[:2], -1),
        splits=splits, observation_splits=obs_split, dimension=int(spec.get("dimension", 256)),
        gold_properties=dict(GOLD_PROPERTIES),
        metadata={"probes": probes, "host": cache.get("host"), "templates": cache.get("templates"),
                  "wordnet_version": cache.get("wordnet_version"), "concepts": n, "edges": len(heads),
                  "edges_per_relation": {nm: int(sum(1 for r in relations if r == i)) for i, nm in enumerate(onto.relation_names)},
                  "concepts_also_fillers": int(sum(1 for a in node_of_atomic if a in set(names)))})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    anchors = sub.add_parser("anchors", help="compute and cache the multi-template anchors (GPU if available)")
    anchors.add_argument("--config", type=Path, required=True)
    anchors.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists; choose a new file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = compute_anchors(config["world"])
    torch.save(cache, args.output)
    print({"concepts": len(cache["concept_names"]), "templates": len(cache["templates"]),
           "shape": list(cache["anchors"].shape), "output": str(args.output)})


if __name__ == "__main__":
    main()
