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

Corpus-scale pieces for E7 (WP-E7; new names, the functions above are unchanged):

- `span_occurrences` / `discover_candidates` — streaming discovery over a reading corpus of
  millions of tokens: n-grams not starting or ending with a stopword, counted by hash, surprisal
  from batched sliding windows over a capped sample of each candidate's occurrences;
- `authoring_prompt_fewshot`, `cut_completion`, `pool_proposals`, `self_consistent` — the
  constrained template with demonstrations (base LMs follow it far better than a bare
  instruction), and the self-consistency vote of step 3(i);
- `split_contexts` — authoring and validation documents of a candidate, disjoint by document;
- `ComputeLedger` — FLOPs of discovery, authoring and verification, converted into the extra
  tokens of the compute-matched control (formulation §5.4, §5.6);
- `ValidationWindow`, `window_losses`, `verify_frames` — batched exact utility with greedy,
  sequential acceptance (each edge is scored against the frame accepted so far);
- `first_order_edge_utility`, `edge_mass_losses` — the first-order utility of §5.2,
  `U = −∂L/∂β` for `z_j + β v_e` (a TracIn/LESS-style inner product), for concepts whose frame is
  not empty; it equals the finite-difference loss change for small β (tested).
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Container, Iterable, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .algebra import normalize
from .compose import FrameSchedule, _occurrence_edges
from .span_channel import AliasTable, CausalLinker, normalize_alias

WORD = re.compile(r"[A-Za-z][A-Za-z\-]+")


@dataclass
class Candidate:
    surface: str
    occurrences: list[tuple[int, int, int]]          # (text id, char start, char end)
    subtokens: int
    surprisal: float = 0.0                           # mean summed surprisal over occurrences
    excess: float = 0.0                              # surprisal − mean of same-length spans
    count: int = 0                                   # occurrences in the corpus (discover_candidates)


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


# ---------------------------------------------------------------------------------------------
# Corpus-scale discovery (E7)

STOPWORDS = frozenset("""
a an the and or but nor of in on at to for from by with without about as into onto over under between among
through during before after above below up down out off again further then once here there when where why how
all any both each few more most other some such no not only own same so than too very can will just should now
is are was were be been being have has had having do does did doing i me my we our you your he him his she her
it its they them their what which who whom whose this that these those am would could might must shall may also
however therefore thus if while because until unless per via vs etc one two three many much several
""".split())
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*[A-Za-z0-9]")


def span_occurrences(text: str, *, max_words: int = 3, pattern: re.Pattern = WORD, stopwords: Container[str] = STOPWORDS,
                     normalize_key: Callable[[str], str] = normalize_alias) -> Iterable[tuple[str, int, int]]:
    """Word n-grams of 1..max_words words joined by single spaces that neither start nor end with a
    stopword: `(normalized key, char start, char end)`."""
    words = list(pattern.finditer(text))
    lowered = [w.group().lower() for w in words]
    for i in range(len(words)):
        if lowered[i] in stopwords:
            continue
        for n in range(1, max_words + 1):
            j = i + n - 1
            if j >= len(words):
                break
            if n > 1 and text[words[j - 1].end():words[j].start()] != " ":
                break
            if lowered[j] in stopwords:
                continue
            yield normalize_key(text[words[i].start():words[j].end()]), words[i].start(), words[j].end()


@torch.no_grad()
def batched_surprisal(model: Any, tokenizer: Any, texts: Sequence[str], device: torch.device, *, window: int = 1024,
                      stride: int = 768, batch: int = 8) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], int]:
    """Per text: (token start offsets, token end offsets, surprisal) with `−log p(x_t | x_<t)` from
    sliding windows (each token scored with ≥ window − stride tokens of context, except at the start;
    the first token of a text scores 0). Returns the per-text results and the tokens forwarded."""
    from .integrations.transformers import chunked_causal_lm_loss
    encoded = tokenizer(list(texts), return_offsets_mapping=True, add_special_tokens=False)
    results, jobs = [], []
    for t, (ids, offsets) in enumerate(zip(encoded["input_ids"], encoded["offset_mapping"])):
        offsets = np.asarray(offsets, dtype=np.int64).reshape(-1, 2)
        results.append((offsets[:, 0], offsets[:, 1], np.zeros(len(ids), dtype=np.float32)))
        start = 0
        while True:
            jobs.append((t, start, ids[start:start + window], 0 if start == 0 else window - stride))
            if start + window >= len(ids):
                break
            start += stride
    jobs.sort(key=lambda job: -len(job[2]))
    head = model.get_output_embeddings()
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    forwarded = 0
    for b in range(0, len(jobs), batch):
        chunk = [j for j in jobs[b:b + batch] if len(j[2]) > 1]
        if not chunk:
            continue
        length = max(len(j[2]) for j in chunk)
        ids = torch.full((len(chunk), length), pad, dtype=torch.long)
        mask = torch.zeros((len(chunk), length), dtype=torch.long)
        for row, (_, _, piece, _) in enumerate(chunk):
            ids[row, :len(piece)] = torch.tensor(piece); mask[row, :len(piece)] = 1
        ids, mask = ids.to(device), mask.to(device)
        forwarded += int(mask.sum())
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hidden = model.base_model(input_ids=ids, attention_mask=mask).last_hidden_state
        labels = ids.masked_fill(mask == 0, -100)
        bias = getattr(head, "bias", None)
        losses = chunked_causal_lm_loss(hidden.float(), head.weight.float(), labels, reduction="none",
                                        output_bias=None if bias is None else bias.float()).cpu().numpy()
        for row, (t, start, piece, keep_from) in enumerate(chunk):
            values = losses[row, :len(piece) - 1]               # value j scores token j + 1 of the window
            first = max(1, keep_from)
            results[t][2][start + first:start + len(piece)] = values[first - 1:]
    return results, forwarded


def discover_candidates(model: Any, tokenizer: Any, texts: Sequence[str], device: torch.device, *, exclude: Container[str],
                        min_count: int = 8, min_subtokens: int = 2, max_candidates: int = 5000, max_words: int = 3,
                        max_occurrences: int = 32, pattern: re.Pattern = WORD, stopwords: Container[str] = STOPWORDS,
                        normalize_key: Callable[[str], str] = normalize_alias, window: int = 1024, stride: int = 768,
                        batch: int = 8, texts_per_pass: int = 256) -> tuple[list[Candidate], dict[str, Any]]:
    """Streaming version of `find_candidates` for a large reading corpus.

    Spans not in `exclude` (the visible linker and any excluded aliases) occurring ≥ `min_count` times
    with ≥ `min_subtokens` host subtokens are scored by the mean summed surprisal of up to
    `max_occurrences` occurrences (the first ones in corpus order); `excess` subtracts the mean of
    the scored spans with the same subtoken length; the `max_candidates` largest are returned with
    their total `count`."""
    hashes: list[int] = []
    for text in texts:
        hashes.extend(hash(key) for key, _, _ in span_occurrences(text, max_words=max_words, pattern=pattern,
                                                                  stopwords=stopwords, normalize_key=normalize_key)
                      if key not in exclude)
    values, counts = np.unique(np.asarray(hashes, dtype=np.int64), return_counts=True)
    occurrences_total = len(hashes)
    del hashes
    frequent = dict(zip(values[counts >= min_count].tolist(), counts[counts >= min_count].tolist()))
    surfaces: dict[int, str] = {}
    places: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for t, text in enumerate(texts):
        for key, start, end in span_occurrences(text, max_words=max_words, pattern=pattern, stopwords=stopwords,
                                                normalize_key=normalize_key):
            h = hash(key)
            if h in frequent and key not in exclude and len(places[h]) < max_occurrences:
                surfaces.setdefault(h, key)
                places[h].append((t, start, end))
    keys = sorted(surfaces, key=lambda h: surfaces[h])
    lengths: dict[int, int] = {}
    for i in range(0, len(keys), 4096):
        chunk = keys[i:i + 4096]
        encoded = tokenizer([" " + surfaces[h] for h in chunk], add_special_tokens=False)["input_ids"]
        lengths.update({h: len(ids) for h, ids in zip(chunk, encoded)})
    keys = [h for h in keys if lengths[h] >= min_subtokens]
    by_text: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for h in keys:
        for t, start, end in places[h]:
            by_text[t].append((h, start, end))
    sums: dict[int, list[float]] = defaultdict(list)
    needed = sorted(by_text)
    forwarded = 0
    for i in range(0, len(needed), texts_per_pass):
        group = needed[i:i + texts_per_pass]
        scored, tokens = batched_surprisal(model, tokenizer, [texts[t] for t in group], device, window=window,
                                           stride=stride, batch=batch)
        forwarded += tokens
        for t, (starts, ends, surprisal) in zip(group, scored):
            for h, start, end in by_text[t]:
                lo, hi = int(np.searchsorted(ends, start, side="right")), int(np.searchsorted(starts, end, side="left"))
                inside = [k for k in range(lo, hi) if ends[k] > starts[k]]
                if inside:
                    sums[h].append(float(surprisal[inside].sum()))
    candidates = [Candidate(surfaces[h], places[h], lengths[h], float(np.mean(sums[h])), count=int(frequent[h]))
                  for h in keys if sums[h]]
    by_length: dict[int, list[float]] = defaultdict(list)
    for c in candidates:
        by_length[c.subtokens].append(c.surprisal)
    for c in candidates:
        c.excess = c.surprisal - float(np.mean(by_length[c.subtokens]))
    candidates.sort(key=lambda c: (-c.excess, c.surface))
    stats = {"span_occurrences": occurrences_total, "distinct_spans": int(values.size), "frequent_spans": len(frequent),
             "scored_candidates": len(candidates), "forward_tokens": forwarded, "texts_scored": len(needed),
             "min_count": min_count, "min_subtokens": min_subtokens, "max_occurrences": max_occurrences}
    return candidates[:max_candidates], stats


def excerpt(text: str, start: int, end: int, *, chars: int = 400) -> str:
    """About `chars` characters of `text` around [start, end), cut at word boundaries, whitespace collapsed."""
    left, right = max(0, start - chars // 2), min(len(text), end + chars // 2)
    if left > 0:
        space = text.find(" ", left, start)
        left = space + 1 if space >= 0 else left
    if right < len(text):
        space = text.rfind(" ", end, right)
        right = space if space >= 0 else right
    return " ".join(text[left:right].split())


# ---------------------------------------------------------------------------------------------
# Authoring template, sample pooling, context splits

def authoring_prompt_fewshot(surface: str, context: str, relations: Sequence[str],
                             demonstrations: Sequence[dict[str, Any]] = (), descriptions: dict[str, str] | None = None) -> str:
    """The constrained template of `authoring_prompt` with worked demonstrations (`context`,
    `surface`, `edges` as (relation, filler) pairs); the host continues after `Concept: <surface>`."""
    vocabulary = ", ".join(f"{r} ({descriptions[r]})" if descriptions and r in descriptions else r for r in relations)
    blocks = [f"Each example names a concept from a text and lists facts about it, one per line as "
              f"`relation: related concept`, using only these relations: {vocabulary}. "
              f"A related concept is a short general noun phrase."]
    for demo in demonstrations:
        facts = "\n".join(f"{relation}: {filler}" for relation, filler in demo["edges"])
        blocks.append(f"Text: {' '.join(demo['context'].split())}\nConcept: {demo['surface']}\n{facts}")
    blocks.append(f"Text: {' '.join(context.split())}\nConcept: {surface}\n")
    return "\n\n".join(blocks)


def cut_completion(text: str) -> str:
    """The completion up to the end of its fact list (a blank line or the next example)."""
    for marker in ("\n\n", "\nText:", "\nConcept:"):
        position = text.find(marker)
        if position >= 0:
            text = text[:position]
    return text


def pool_proposals(completions: Sequence[Sequence[str]], relations: Sequence[str]) -> tuple[Counter, int]:
    """Votes per (relation, filler) — at most one per sample — over contexts × samples, and the sample count."""
    votes: Counter = Counter()
    samples = 0
    for per_context in completions:
        for completion in per_context:
            samples += 1
            votes.update(set(parse_proposals(cut_completion(completion), relations)))
    return votes, samples


def self_consistent(votes: Counter, samples: int, *, min_share: float) -> list[dict[str, Any]]:
    """Edges proposed in ≥ `min_share` of the samples (formulation §5.1 step 3(i)), most votes first."""
    return [{"relation": relation, "filler": filler, "votes": n, "samples": samples}
            for (relation, filler), n in sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
            if samples and n / samples >= min_share]


def stable_seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()[:8], "little")


def split_contexts(documents: Sequence[int], *, authoring: int, key: str, seed: int) -> tuple[list[int], list[int]]:
    """Disjoint authoring and validation documents of one candidate: a seeded permutation of the
    distinct documents it occurs in; at most half of them (and at most `authoring`) author."""
    unique = sorted(set(int(d) for d in documents))
    take = min(authoring, max(1, len(unique) // 2)) if len(unique) > 1 else len(unique)
    order = np.random.default_rng(stable_seed(seed, key)).permutation(len(unique))
    chosen = sorted(unique[i] for i in order[:take])
    return chosen, sorted(unique[i] for i in order[take:])


# ---------------------------------------------------------------------------------------------
# Compute accounting (compute-matched control)

def training_flops_per_token(parameters: float, host_mode: str, trainable: float = 0.0) -> float:
    """FLOPs per training token: `6N` for a trained host; a frozen host still back-propagates through
    its activations to the channel (`2N` forward + `2N` backward, experiments.md §Compute: `≈ 4·N·D`);
    LoRA adds the adapters' weight gradients (`2·trainable`)."""
    if host_mode == "train":
        return 6.0 * parameters
    return 4.0 * parameters + (2.0 * trainable if host_mode == "lora" else 0.0)


@dataclass
class ComputeLedger:
    """FLOPs spent outside training, by stage: `2N` per forwarded, prompt or generated token."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(self, stage: str, *, parameters: float, forward_tokens: int = 0, prompt_tokens: int = 0,
            generated_tokens: int = 0, seconds: float | None = None, note: str = "") -> dict[str, Any]:
        entry = {"stage": stage, "parameters": float(parameters), "forward_tokens": int(forward_tokens),
                 "prompt_tokens": int(prompt_tokens), "generated_tokens": int(generated_tokens),
                 "flops": 2.0 * float(parameters) * (forward_tokens + prompt_tokens + generated_tokens),
                 "seconds": seconds, "note": note}
        self.entries.append(entry)
        return entry

    def flops(self, stages: Iterable[str] | None = None) -> float:
        wanted = None if stages is None else set(stages)
        return float(sum(e["flops"] for e in self.entries if wanted is None or e["stage"] in wanted))

    def extend(self, other: "ComputeLedger") -> "ComputeLedger":
        self.entries.extend(other.entries)
        return self

    def to_json(self) -> list[dict[str, Any]]:
        return list(self.entries)

    @classmethod
    def from_json(cls, entries: Iterable[dict[str, Any]]) -> "ComputeLedger":
        return cls([dict(e) for e in entries])


def compute_matched_tokens(round_tokens: int, ledger: ComputeLedger, *, parameters: float, host_mode: str,
                           stages: Iterable[str] | None = None, trainable: float = 0.0) -> dict[str, Any]:
    """Training tokens of the compute-matched control: the round's tokens plus the ledger's FLOPs
    (selected stages) divided by the training FLOPs per token of the consumer."""
    per_token = training_flops_per_token(parameters, host_mode, trainable)
    selected = None if stages is None else sorted(stages)
    flops = ledger.flops(selected)
    extra = int(math.ceil(flops / per_token)) if flops else 0
    return {"round_tokens": int(round_tokens), "extra_tokens": extra, "total_tokens": int(round_tokens) + extra,
            "flops": flops, "flops_per_training_token": per_token, "stages": selected}


# ---------------------------------------------------------------------------------------------
# Batched held-out utility (verification)

@dataclass
class ValidationWindow:
    """A fixed-length token window ending `after` tokens past one occurrence of a candidate span."""

    ids: np.ndarray                         # (L,) token ids
    spans: dict[str, np.ndarray]            # other linked spans (start, end, inject, entry, length, confidence), local
    candidate: int                          # candidate index
    start: int                              # first subtoken of the candidate span (local)
    end: int                                # last subtoken (local)
    document: int = -1


def _window_batch(windows: Sequence[ValidationWindow], entries: Sequence[int | None]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ids = torch.from_numpy(np.stack([w.ids for w in windows]).astype(np.int64))
    columns: dict[str, list] = {k: [] for k in ("batch", "start", "end", "inject", "entry", "confidence", "length")}
    for b, (window, entry) in enumerate(zip(windows, entries)):
        spans = window.spans
        keep = (spans["inject"] != window.end) if entry is not None else np.ones(spans["inject"].shape, dtype=bool)
        for key in ("start", "end", "inject", "entry", "length"):
            columns[key].append(spans[key][keep].astype(np.int64))
        columns["confidence"].append(spans["confidence"][keep].astype(np.float32))
        columns["batch"].append(np.full(int(keep.sum()), b, dtype=np.int64))
        if entry is not None:
            for key, value in (("start", window.start), ("end", window.end), ("inject", window.end), ("entry", entry),
                               ("length", window.end - window.start + 1), ("batch", b)):
                columns[key].append(np.asarray([value], dtype=np.int64))
            columns["confidence"].append(np.ones(1, dtype=np.float32))
    spans_t = {k: torch.from_numpy(np.concatenate(v)) for k, v in columns.items()}
    order = torch.argsort(spans_t["batch"] * (ids.shape[1] + 1) + spans_t["inject"], stable=True)
    return ids, {k: v[order] for k, v in spans_t.items()}


def window_losses(lm: Any, windows: Sequence[ValidationWindow], entries: Sequence[int | None], device: torch.device, *,
                  after: int = 8, batch: int = 16) -> tuple[np.ndarray, int]:
    """Mean next-token loss over the `after` tokens following each window's candidate span, the span
    linked to `entries[i]` (None = not linked); other spans of the window are linked as given.
    Returns the losses and the tokens forwarded."""
    out = np.zeros(len(windows), dtype=np.float64)
    forwarded = 0
    lm.eval()
    with torch.no_grad():
        for b in range(0, len(windows), batch):
            chunk, chunk_entries = windows[b:b + batch], entries[b:b + batch]
            ids, spans = _window_batch(chunk, chunk_entries)
            ids = ids.to(device)
            forwarded += int(ids.numel())
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                per_token = lm(ids, spans={k: v.to(device) for k, v in spans.items()} if lm.channel else None,
                               labels=ids, reduction="none")["loss"].float().cpu()
            for row, window in enumerate(chunk):
                out[b + row] = float(per_token[row, window.end:window.end + after].mean())
    return out, forwarded


def replace_frames(schedule: FrameSchedule, frames: dict[int, Sequence[tuple[int, int]]]) -> FrameSchedule:
    """`schedule` with the frames of the given concepts replaced (vectorized CSR rebuild)."""
    if not frames:
        return schedule
    offsets = schedule.offsets.cpu()
    degrees = (offsets[1:] - offsets[:-1]).numpy()
    owner = np.repeat(np.arange(degrees.size), degrees)
    replaced = np.zeros(degrees.size, dtype=bool)
    replaced[list(frames)] = True
    keep = ~replaced[owner]
    new_owner = np.asarray([c for c, frame in frames.items() for _ in frame], dtype=np.int64)
    new_rel = np.asarray([r for frame in frames.values() for r, _ in frame], dtype=np.int64)
    new_fil = np.asarray([a for frame in frames.values() for _, a in frame], dtype=np.int64)
    owner_all = np.concatenate([owner[keep], new_owner])
    relations = np.concatenate([schedule.relations.cpu().numpy()[keep], new_rel])
    fillers = np.concatenate([schedule.fillers.cpu().numpy()[keep], new_fil])
    order = np.argsort(owner_all, kind="stable")
    counts = np.bincount(owner_all, minlength=degrees.size)
    device = schedule.offsets.device
    return FrameSchedule(torch.from_numpy(np.concatenate([[0], np.cumsum(counts)])).to(device),
                         torch.from_numpy(relations[order]).to(device), torch.from_numpy(fillers[order]).to(device))


def verify_frames(lm: Any, windows: dict[int, Sequence[ValidationWindow]], proposals: dict[int, Sequence[tuple[int, int]]],
                  entry_of: dict[int, int], device: torch.device, *, min_validation: int = 4, quantile: float = 0.05,
                  resamples: int = 1000, seed: int = 0, after: int = 8, batch: int = 16) -> tuple[dict[int, list[dict[str, Any]]], int]:
    """Held-out utility of proposed edges with greedy, sequential acceptance.

    Candidate `c` (composer entry `entry_of[c]`, empty frame) is scored on its validation `windows[c]`:
    phase k adds its k-th proposed edge to the frame accepted so far; the utility per occurrence is
    the loss decrease over the `after` tokens following the span; the edge is accepted when the
    bootstrap `quantile` bound of the mean utility is > 0 (formulation §5.1 step 3(ii)). All candidates
    advance one phase per pass, so one batched pass scores one edge of every candidate. Candidates with
    fewer than `min_validation` windows are not verifiable (every edge rejected). The composer schedule
    is restored afterwards. Returns per-candidate edge records and the tokens forwarded."""
    composer = lm.channel.composer
    original = composer.schedule
    accepted: dict[int, list[tuple[int, int]]] = {c: [] for c in proposals}
    records: dict[int, list[dict[str, Any]]] = {c: [] for c in proposals}
    verifiable = [c for c in proposals if len(windows.get(c, ())) >= min_validation and proposals[c]]
    for c in proposals:
        if c not in verifiable:
            records[c] = [{"relation": int(r), "atom": int(a), "mean": float("nan"), "low": float("-inf"),
                           "n": len(windows.get(c, ())), "accepted": False, "phase": k, "reason": "too few validation contexts"}
                          for k, (r, a) in enumerate(proposals[c])]
    forwarded = 0

    def evaluate(active: list[int], frames: dict[int, list[tuple[int, int]]] | None) -> dict[int, np.ndarray]:
        nonlocal forwarded
        flat = [(c, w) for c in active for w in windows[c]]
        if frames is not None:
            composer.set_schedule(replace_frames(original, {entry_of[c]: frames[c] for c in active}))
        try:
            losses, tokens = window_losses(lm, [w for _, w in flat],
                                           [None if frames is None else entry_of[c] for c, _ in flat], device,
                                           after=after, batch=batch)
        finally:
            composer.set_schedule(original)
        forwarded += tokens
        out: dict[int, list[float]] = defaultdict(list)
        for (c, _), value in zip(flat, losses):
            out[c].append(float(value))
        return {c: np.asarray(v) for c, v in out.items()}

    before = evaluate(verifiable, None) if verifiable else {}
    phase = 0
    while True:
        active = [c for c in verifiable if phase < len(proposals[c])]
        if not active:
            break
        trial = {c: accepted[c] + [tuple(proposals[c][phase])] for c in active}
        after_losses = evaluate(active, trial)
        for c in active:
            deltas = before[c] - after_losses[c]
            low = bootstrap_lower(deltas.tolist(), resamples=resamples, seed=seed + phase, quantile=quantile)
            relation, atom = proposals[c][phase]
            ok = low > 0
            records[c].append({"relation": int(relation), "atom": int(atom), "mean": float(deltas.mean()), "low": float(low),
                               "n": int(deltas.size), "accepted": bool(ok), "phase": phase})
            if ok:
                accepted[c] = trial[c]
                before[c] = after_losses[c]
        phase += 1
    return records, forwarded


# ---------------------------------------------------------------------------------------------
# First-order utility (formulation §5.2)

class _EdgeMass:
    """Context manager: the composition of `entry` becomes `N(z_entry + β_i v_e)` for its i-th occurrence."""

    def __init__(self, composer: Any, entry: int, edge: tuple[int, int], beta: torch.Tensor) -> None:
        self.composer, self.entry, self.edge, self.beta = composer, entry, edge, beta

    def __enter__(self) -> "_EdgeMass":
        composer, entry, beta = self.composer, self.entry, self.beta
        relation, filler = self.edge

        def compose(concept_ids: torch.Tensor, context: torch.Tensor | None = None, *, return_weights: bool = False):
            edge_index, segments = _occurrence_edges(composer.schedule, concept_ids)
            bound = composer.bound_edges(edge_index)
            weights = composer.edge_weights(edge_index, segments, concept_ids, bound, context)
            summed = bound.new_zeros(concept_ids.numel(), bound.shape[-1]).index_add(0, segments, weights.unsqueeze(-1) * bound)
            hit = (concept_ids == entry).nonzero().flatten()
            if hit.numel() != beta.numel():
                raise ValueError(f"entry {entry} occurs {hit.numel()} times in the batch, expected {beta.numel()}")
            extra = composer.transform(torch.tensor([relation], device=summed.device),
                                       composer.atomic_vectors()[torch.tensor([filler], device=summed.device)])
            summed = summed.index_add(0, hit, beta.to(summed.dtype)[:, None] * extra)
            return normalize(summed)

        composer.compose = compose
        return self

    def __exit__(self, *exc: Any) -> None:
        del self.composer.compose


def _mass_losses(lm: Any, windows: Sequence[ValidationWindow], entry: int, edge: tuple[int, int], beta: torch.Tensor,
                 device: torch.device, after: int) -> torch.Tensor:
    ids, spans = _window_batch(windows, [entry] * len(windows))
    ids = ids.to(device)
    with _EdgeMass(lm.channel.composer, entry, edge, beta):
        per_token = lm(ids, spans={k: v.to(device) for k, v in spans.items()}, labels=ids, reduction="none")["loss"]
    return torch.stack([per_token[i, w.end:w.end + after].mean() for i, w in enumerate(windows)])


def edge_mass_losses(lm: Any, windows: Sequence[ValidationWindow], entry: int, edge: tuple[int, int], beta: float,
                     device: torch.device, *, after: int = 8) -> np.ndarray:
    """Per-occurrence loss with the edge added at mass `beta` (`z_j + β v_e`; β = 0 is the frame as is)."""
    dtype = next(lm.parameters()).dtype
    with torch.no_grad():
        values = _mass_losses(lm, windows, entry, edge, torch.full((len(windows),), float(beta), dtype=dtype, device=device),
                              device, after)
    return values.double().cpu().numpy()


def first_order_edge_utility(lm: Any, windows: Sequence[ValidationWindow], entry: int, edge: tuple[int, int],
                             device: torch.device, *, after: int = 8, seed: int = 0, quantile: float = 0.05) -> dict[str, Any]:
    """`U_{j,e} = −∂L/∂β |_{β=0}` per validation occurrence (one β per occurrence, one backward pass):
    a TracIn/LESS-style inner product `−⟨g_j, ∂z_j/∂β⟩` through the row normalisation. Needs a
    non-empty frame for `entry`."""
    dtype = next(lm.parameters()).dtype
    beta = torch.zeros(len(windows), dtype=dtype, device=device, requires_grad=True)
    losses = _mass_losses(lm, windows, entry, edge, beta, device, after)
    (gradient,) = torch.autograd.grad(losses.sum(), beta)
    values = (-gradient).double().cpu().numpy()
    return {"mean": float(values.mean()), "low": bootstrap_lower(values.tolist(), seed=seed, quantile=quantile),
            "n": int(values.size), "values": values}


def card_from_record(surface: str, relation: str, filler: str, record: dict[str, Any], *, proposals: int, samples: int,
                     round_index: int, contexts: Sequence[int]) -> dict[str, Any]:
    """An authoring card (formulation §5.3) as a JSON-ready dict."""
    card = AuthoringCard(surface, relation, filler, int(record["atom"]), proposals, samples, float(record["mean"]),
                         float(record["low"]), bool(record["accepted"]), round_index, list(contexts))
    return {**asdict(card), "n": int(record["n"]), "phase": record.get("phase"), "reason": record.get("reason")}
