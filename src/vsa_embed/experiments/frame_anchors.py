"""Frozen-host anchors for ontology concepts and node-disjoint frame-regression splits (E2, E3).

A concept's anchor is the host's final hidden state at the last token of
`Definition: <gloss> Term: <term>` (gloss shortened to fit, so the read position is the term's last
token). Anchors are centred on training concepts only and L2-normalised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from vsa_embed.compose import FrameSchedule
from vsa_embed.ontologies.wordnet import FrameOntology
from vsa_embed.statistics import positive_ranks


@torch.no_grad()
def term_anchors(host: Any, tokenizer: Any, glosses: Sequence[str], terms: Sequence[str], *, device: torch.device,
                 max_length: int = 64, batch_size: int = 64) -> torch.Tensor:
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    host = host.to(device).eval()
    prompts = []
    for gloss, term in zip(glosses, terms):
        prefix, suffix = "Definition: ", f" Term: {term}"
        budget = max(4, max_length - len(tokenizer.encode(prefix + suffix, add_special_tokens=False)))
        ids = tokenizer.encode(gloss, add_special_tokens=False)
        prompts.append(prefix + (gloss if len(ids) <= budget else tokenizer.decode(ids[:budget])) + suffix)
    outputs = []
    for start in range(0, len(prompts), batch_size):
        encoded = tokenizer(prompts[start:start + batch_size], padding=True, truncation=True,
                            max_length=max_length + 16, return_tensors="pt").to(device)
        hidden = host(**encoded, output_hidden_states=True).hidden_states[-1]
        last = encoded["attention_mask"].sum(1) - 1
        outputs.append(hidden[torch.arange(hidden.shape[0], device=device), last].float().cpu())
    return torch.cat(outputs)


@dataclass
class FrameRegressionData:
    name: str
    concept_ids: list[int]          # ontology concept ids, in schedule order
    schedule: FrameSchedule
    anchors: torch.Tensor           # raw host anchors (concepts × host_dim)
    atomic_count: int
    relation_count: int


def build_dataset(ontology: FrameOntology, concept_ids: list[int], anchors: torch.Tensor) -> FrameRegressionData:
    frames = [ontology.frames[c] for c in concept_ids]
    return FrameRegressionData(ontology.name, concept_ids, FrameSchedule.from_frames(frames), anchors,
                               len(ontology.atomic_names), len(ontology.relation_names))


def node_disjoint_split(n: int, *, seed: int, test_fraction: float = 0.2, validation_fraction: float = 0.1):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test, n_val = int(n * test_fraction), int(n * validation_fraction)
    return {"test": np.sort(order[:n_test]), "validation": np.sort(order[n_test:n_test + n_val]),
            "train": np.sort(order[n_test + n_val:])}


def centred_targets(anchors: torch.Tensor, train: np.ndarray) -> torch.Tensor:
    return F.normalize(anchors - anchors[torch.from_numpy(train)].mean(0), dim=-1)


def regression_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Retrieval among the evaluated concepts (shared candidate set, mid-rank ties), cosine, VE."""
    scores = F.normalize(prediction, dim=-1) @ F.normalize(target, dim=-1).T
    ranks = positive_ranks(scores, torch.eye(len(target), dtype=torch.bool), ties="mid")
    sse = float(((prediction - target) ** 2).sum()); sst = float(((target - target.mean(0)) ** 2).sum())
    return {"mrr": float((1 / ranks).mean()), "recall_at_10": float((ranks <= 10).float().mean()),
            "cosine": float(F.cosine_similarity(prediction, target).mean()), "variance_explained": 1 - sse / max(sst, 1e-12)}
