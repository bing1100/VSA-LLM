"""Self-authored ontologies (M5, formulation §5; task B12).

One round:
1. `find_candidates` — multi-token word n-grams frequent in the reading corpus, not covered by the
   alias table, ranked by excess surprisal (their mean per-occurrence surprisal under the host
   minus the mean of spans with the same subtoken length);
2. `author` — sample frames `relation: filler` for each candidate from a generator (the host
   itself, or the Claude teacher) under a closed relation vocabulary; pool over contexts and
   samples, keep edges proposed in ≥ ρ of samples whose filler resolves to a dictionary atom;
3. `edge_utility` — held-out utility of each edge, measured exactly as the loss change on
   validation contexts (disjoint from authoring contexts) when the candidate is linked with the
   edge added to its frame, versus without it; bootstrap lower bound over occurrences;
4. `accept` — edges with a positive lower bound, logged as authoring cards.

(Formulation §5.2 gives a first-order version of step 3; the exact difference is used here because
a new concept starts with an empty frame, where the first-order expansion through the row
normalisation is undefined.)
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .compose import FrameSchedule
from .span_channel import AliasTable, CausalLinker, normalize_alias

WORD = re.compile(r"[A-Za-z][A-Za-z\-]+")


@dataclass
class Candidate:
    surface: str
    occurrences: list[tuple[int, int, int]]          # (text id, char start, char end)
    subtokens: int
    surprisal: float = 0.0                           # mean summed surprisal over occurrences
    excess: float = 0.0                              # surprisal − mean of same-length spans


@dataclass
class AuthoringCard:
    surface: str
    relation: str
    filler: str
    atom: int
    proposals: int
    samples: int
    utility_mean: float
    utility_low: float
    accepted: bool
    round: int
    contexts: list[int] = field(default_factory=list)


def ngram_spans(text: str, max_words: int = 3) -> Iterable[tuple[str, int, int]]:
    words = list(WORD.finditer(text))
    for i in range(len(words)):
        for n in range(1, max_words + 1):
            if i + n > len(words):
                break
            span = text[words[i].start():words[i + n - 1].end()]
            if n > 1 and not re.fullmatch(r"[A-Za-z\-]+( [A-Za-z\-]+)*", span):
                break
            yield span, words[i].start(), words[i + n - 1].end()


@torch.no_grad()
def token_surprisal(model: Any, tokenizer: Any, text: str, device: torch.device) -> tuple[list[tuple[int, int]], torch.Tensor]:
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False, truncation=True, max_length=1024)
    ids = torch.tensor([encoded["input_ids"]], device=device)
    logits = model(input_ids=ids).logits[0].float()
    surprisal = torch.zeros(ids.shape[1])
    if ids.shape[1] > 1:
        surprisal[1:] = -torch.log_softmax(logits[:-1], -1).gather(1, ids[0, 1:, None]).squeeze(1).cpu()
    return encoded["offset_mapping"], surprisal


def find_candidates(model: Any, tokenizer: Any, texts: Sequence[str], table: AliasTable, device: torch.device, *,
                    min_count: int = 5, min_subtokens: int = 2, max_candidates: int = 200, max_words: int = 3) -> list[Candidate]:
    counts: Counter[str] = Counter()
    places: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for t, text in enumerate(texts):
        for span, start, end in ngram_spans(text, max_words):
            key = normalize_alias(span)
            if key in table.alias_to_entry:
                continue
            counts[key] += 1
            places[key].append((t, start, end))
    frequent = [k for k, c in counts.items() if c >= min_count]
    lengths = {k: len(tokenizer.encode(" " + k, add_special_tokens=False)) for k in frequent}
    frequent = [k for k in frequent if lengths[k] >= min_subtokens]
    needed = sorted({t for k in frequent for t, _, _ in places[k]})
    cache = {t: token_surprisal(model, tokenizer, texts[t], device) for t in needed}
    candidates = []
    for k in frequent:
        values = []
        for t, start, end in places[k]:
            offsets, surprisal = cache[t]
            inside = [i for i, (s, e) in enumerate(offsets) if s < end and e > start and e > s]
            if inside:
                values.append(float(surprisal[inside].sum()))
        if values:
            candidates.append(Candidate(k, places[k], lengths[k], float(np.mean(values))))
    by_length = defaultdict(list)
    for c in candidates:
        by_length[c.subtokens].append(c.surprisal)
    for c in candidates:
        c.excess = c.surprisal - float(np.mean(by_length[c.subtokens]))
    candidates.sort(key=lambda c: -c.excess)
    return candidates[:max_candidates]


def authoring_prompt(surface: str, context: str, relations: Sequence[str]) -> str:
    return (f"Text: {context.strip()}\n"
            f"In this text, \"{surface}\" is a concept. Describe it with relations chosen from: {', '.join(relations)}.\n"
            f"Write one relation per line as `relation: related concept`.\n")


def parse_proposals(text: str, relations: Sequence[str]) -> list[tuple[str, str]]:
    allowed = {r.lower(): r for r in relations}
    proposals = []
    for line in text.splitlines():
        if ":" not in line:
            continue
        relation, filler = (part.strip(" -*`\t.") for part in line.split(":", 1))
        relation = relation.lower().replace(" ", "_")
        filler = normalize_alias(filler.split(",")[0].split(";")[0])
        if relation in allowed and filler and len(filler) < 60:
            proposals.append((allowed[relation], filler))
    return proposals


def author(candidate: Candidate, texts: Sequence[str], generate: Callable[[str], list[str]],
           relations: Sequence[str], resolve_filler: Callable[[str], int | None], *, contexts: Sequence[int],
           min_share: float = 0.3) -> list[dict[str, Any]]:
    """Pool proposals over contexts and samples; keep edges with a resolvable filler proposed often enough."""
    votes: Counter[tuple[str, str]] = Counter()
    samples = 0
    for t in contexts:
        for completion in generate(authoring_prompt(candidate.surface, texts[t], relations)):
            samples += 1
            votes.update(set(parse_proposals(completion, relations)))
    edges = []
    for (relation, filler), n in votes.most_common():
        atom = resolve_filler(filler)
        if atom is not None and n / max(1, samples) >= min_share:
            edges.append({"relation": relation, "filler": filler, "atom": atom, "proposals": n, "samples": samples})
    return edges


def bootstrap_lower(values: Sequence[float], *, resamples: int = 1000, seed: int = 0, quantile: float = 0.05) -> float:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return float("-inf")
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, size=(resamples, data.size))].mean(1)
    return float(np.quantile(means, quantile))


@torch.no_grad()
def occurrence_losses(lm: Any, tokenizer: Any, texts: Sequence[str], occurrences: Sequence[tuple[int, int, int]],
                      entry: int | None, device: torch.device, *, after: int = 8) -> list[float]:
    """Mean next-token loss over the `after` tokens following each occurrence, with the span linked
    to `entry` at its last subtoken (or not linked when `entry` is None)."""
    losses = []
    for t, start, end in occurrences:
        encoded = tokenizer(texts[t], return_offsets_mapping=True, add_special_tokens=False, truncation=True, max_length=1024)
        offsets = encoded["offset_mapping"]
        inside = [i for i, (s, e) in enumerate(offsets) if s < end and e > start and e > s]
        if not inside or inside[-1] + 1 >= len(offsets):
            continue
        last = inside[-1]
        ids = torch.tensor([encoded["input_ids"]], device=device)
        spans = None
        if entry is not None:
            spans = {"batch": torch.tensor([0], device=device), "start": torch.tensor([inside[0]], device=device),
                     "end": torch.tensor([last], device=device), "inject": torch.tensor([last], device=device),
                     "entry": torch.tensor([entry], device=device), "confidence": torch.ones(1, device=device),
                     "length": torch.tensor([len(inside)], device=device)}
        per_token = lm(ids, spans=spans, labels=ids, reduction="none")["loss"][0]
        window = per_token[last:min(per_token.numel(), last + after)]
        losses.append(float(window.mean()))
    return losses


def edge_utility(lm: Any, tokenizer: Any, texts: Sequence[str], validation: Sequence[tuple[int, int, int]],
                 entry: int, base_frame: list[tuple[int, int]], edge: tuple[int, int], device: torch.device,
                 *, seed: int = 0) -> dict[str, float]:
    """Loss decrease from adding `edge` to the entry's frame, per validation occurrence."""
    composer = lm.channel.composer
    original = composer.schedule

    def with_frame(frame: list[tuple[int, int]] | None) -> list[float]:
        if frame is None:
            return occurrence_losses(lm, tokenizer, texts, validation, None, device)
        offsets, relations, fillers = original.offsets, original.relations, original.fillers
        start, end = int(offsets[entry]), int(offsets[entry + 1])
        new_rel = torch.cat([relations[:start], torch.tensor([r for r, _ in frame], device=relations.device), relations[end:]])
        new_fil = torch.cat([fillers[:start], torch.tensor([a for _, a in frame], device=fillers.device), fillers[end:]])
        shift = len(frame) - (end - start)
        new_off = offsets.clone(); new_off[entry + 1:] += shift
        composer.set_schedule(FrameSchedule(new_off, new_rel, new_fil))
        try:
            return occurrence_losses(lm, tokenizer, texts, validation, entry, device)
        finally:
            composer.set_schedule(original)

    before = with_frame(base_frame if base_frame else None)
    after = with_frame(base_frame + [edge])
    deltas = [b - a for b, a in zip(before, after)]
    return {"mean": float(np.mean(deltas)) if deltas else float("nan"), "low": bootstrap_lower(deltas, seed=seed),
            "n": len(deltas)}
