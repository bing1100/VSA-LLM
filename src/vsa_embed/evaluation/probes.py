"""Lexical-semantics probes for small causal LMs (B8), in prompting and linear-probe form.

- LAMBADA (prompting): last-word accuracy and mean last-word loss.
- WiC: *prompting* — cosine of the target word's contextual states in the two sentences, with the
  threshold chosen on the training split; *linear probe* — logistic regression on
  `[|h₁ − h₂|, h₁ ⊙ h₂]` trained on the training split. Both are evaluated on the dev split (the
  public test labels are used only as a secondary report).
- Rare-word similarity (CARD-660, Stanford Rare Words): Spearman correlation between human scores
  and cosine of word representations (last-subtoken state of " <word>" after a neutral prefix).

Every probe takes a `represent(texts) -> (hidden states, offsets)`-style model adapter so the same
code serves from-scratch checkpoints and pretrained hosts, with or without the span channel.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.nn import functional as F


@dataclass
class ModelAdapter:
    """Wraps a causal LM + tokenizer; optional `spans_fn(texts, offsets)` adds channel spans."""

    model: Any                       # ChannelLM or HF causal LM
    tokenizer: Any
    device: torch.device
    layer: int = -1                  # hidden layer for representations (−1 = final)
    spans_fn: Callable[[Sequence[str], Sequence[Sequence[tuple[int, int]]]], dict[str, torch.Tensor]] | None = None
    batch_size: int = 32
    max_length: int = 256
    need_logits: bool = False

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda")

    def _run_batch(self, batch: list[str]) -> tuple[Any, Any, Any]:
        """Tokenize one batch (right padding) and run the model: (encoding, outputs with hidden states, output head)."""
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        encoded = self.tokenizer(batch, return_offsets_mapping=True, add_special_tokens=False, padding=True,
                                 truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device); mask = encoded["attention_mask"].to(self.device)
        spans = None
        if self.spans_fn:
            offsets_list = [[tuple(o) for o, m in zip(offs.tolist(), msk.tolist()) if m] for offs, msk in
                            zip(encoded["offset_mapping"], encoded["attention_mask"])]
            spans = {k: v.to(self.device) for k, v in self.spans_fn(batch, offsets_list).items()}
        with torch.no_grad(), self._autocast():
            if hasattr(self.model, "channel") and hasattr(self.model, "embed"):
                embeddings = self.model.embed(input_ids, spans)
                out = self.model.base(inputs_embeds=embeddings, attention_mask=mask, output_hidden_states=True)
                head = self.model.model.get_output_embeddings()
            else:
                out = self.model.base_model(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)
                head = self.model.get_output_embeddings()
        return encoded, out, head

    def _forward(self, texts: Sequence[str]) -> tuple[list[torch.Tensor], list[list[tuple[int, int]]], list[torch.Tensor], list[list[int]]]:
        """Per text: hidden states (T, d), offsets, next-token logits (T, V), ids. Batched with right padding."""
        states, offsets_all, logits_all, ids_all = [], [], [], []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            encoded, out, head = self._run_batch(batch)
            with torch.no_grad(), self._autocast():
                final, chosen = out.hidden_states[-1].float(), out.hidden_states[self.layer].float()
                logits = F.linear(final, head.weight.float()) if self.need_logits else None
            for row in range(len(batch)):
                n = int(encoded["attention_mask"][row].sum())
                states.append(chosen[row, :n].cpu())
                offsets_all.append([tuple(o) for o in encoded["offset_mapping"][row, :n].tolist()])
                logits_all.append(logits[row, :n].cpu() if logits is not None else None)
                ids_all.append(encoded["input_ids"][row, :n].tolist())
        return states, offsets_all, logits_all, ids_all

    def token_logprobs(self, texts: Sequence[str]) -> tuple[list[torch.Tensor], list[list[tuple[int, int]]]]:
        """Per text: `log p(x_t | x_<t)` for t = 1..T−1 (float32, length T−1) and the offsets.

        The log-softmax runs on the device, so only one number per token leaves it (cheap for
        likelihood prompting; `_forward` with `need_logits` copies full vocabulary rows).
        """
        scores, offsets_all = [], []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            encoded, out, head = self._run_batch(batch)
            with torch.no_grad():
                with self._autocast():
                    logits = F.linear(out.hidden_states[-1].float(), head.weight.float())
                ids = encoded["input_ids"].to(logits.device)
                picked = torch.log_softmax(logits[:, :-1].float(), -1).gather(-1, ids[:, 1:, None]).squeeze(-1).cpu()
            for row in range(len(batch)):
                n = int(encoded["attention_mask"][row].sum())
                scores.append(picked[row, :max(0, n - 1)])
                offsets_all.append([tuple(o) for o in encoded["offset_mapping"][row, :n].tolist()])
        return scores, offsets_all

    def word_state(self, texts: Sequence[str], char_spans: Sequence[tuple[int, int]]) -> torch.Tensor:
        """State at the last subtoken overlapping each character span."""
        states, offsets, _, _ = self._forward(texts)
        rows = []
        for state, offs, (start, end) in zip(states, offsets, char_spans):
            index = max((i for i, (s, e) in enumerate(offs) if s < end and e > start), default=len(offs) - 1)
            rows.append(state[index])
        return torch.stack(rows)


# -- LAMBADA ------------------------------------------------------------------------------------

def lambada(adapter: ModelAdapter, path: Path, *, limit: int | None = None) -> dict[str, float]:
    rows = [json.loads(line) for line in path.read_text().splitlines()][:limit]
    texts, word_ids_all, context_lengths = [], [], []
    for row in rows:
        context, word = row["text"].rsplit(" ", 1)
        context_lengths.append(len(adapter.tokenizer(context, add_special_tokens=False)["input_ids"]))
        word_ids_all.append(adapter.tokenizer(" " + word, add_special_tokens=False)["input_ids"])
        texts.append(context + " " + word)
    adapter.need_logits = True
    correct, losses = 0, []
    try:
        for start in range(0, len(texts), adapter.batch_size):
            _, _, logits_batch, ids_batch = adapter._forward(texts[start:start + adapter.batch_size])
            for k, (logits, ids) in enumerate(zip(logits_batch, ids_batch)):
                i = start + k
                word_ids, n_context = word_ids_all[i], context_lengths[i]
                positions = list(range(n_context - 1, n_context - 1 + len(word_ids)))
                if positions[-1] >= logits.shape[0] or ids[n_context:n_context + len(word_ids)] != word_ids:
                    continue      # tokenization of the joined text differs from context + word
                predicted = logits[positions].argmax(-1).tolist()
                correct += int(predicted == word_ids)
                log_probs = torch.log_softmax(logits[positions], -1)
                losses.append(-float(log_probs[torch.arange(len(word_ids)), torch.tensor(word_ids)].mean()))
    finally:
        adapter.need_logits = False
    return {"lambada_accuracy": correct / max(1, len(losses)), "lambada_word_loss": float(np.mean(losses)), "n": len(losses)}


# -- WiC ----------------------------------------------------------------------------------------

def _wic_split(root: Path, split: str) -> list[dict[str, Any]]:
    data = (root / split / f"{split}.data.txt").read_text().splitlines()
    gold = (root / split / f"{split}.gold.txt").read_text().splitlines()
    items = []
    for line, label in zip(data, gold):
        lemma, _, indices, s1, s2 = line.split("\t")
        i1, i2 = (int(x) for x in indices.split("-"))
        items.append({"s1": s1, "s2": s2, "i1": i1, "i2": i2, "label": label.strip() == "T"})
    return items


def _token_char_span(sentence: str, index: int) -> tuple[int, int]:
    words, position = sentence.split(" "), 0
    for i, word in enumerate(words):
        if i == index:
            return position, position + len(word)
        position += len(word) + 1
    return position, position


def _wic_features(adapter: ModelAdapter, items: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    h1 = adapter.word_state([it["s1"] for it in items], [_token_char_span(it["s1"], it["i1"]) for it in items])
    h2 = adapter.word_state([it["s2"] for it in items], [_token_char_span(it["s2"], it["i2"]) for it in items])
    labels = np.asarray([it["label"] for it in items])
    return h1, h2, labels


def _fit_logistic(x: np.ndarray, y: np.ndarray, *, steps: int = 400, l2: float = 1e-2) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mean, std = x.mean(0), x.std(0) + 1e-6
    xt = torch.tensor((x - mean) / std, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    w = torch.zeros(xt.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(xt @ w + b, yt) + l2 * w.square().sum()
        loss.backward(); return loss
    optimizer.step(closure)
    return w.detach().numpy(), float(b.detach()), mean, std


def wic(adapter: ModelAdapter, root: Path, *, train_limit: int | None = None) -> dict[str, float]:
    train, dev = _wic_split(root, "train")[:train_limit], _wic_split(root, "dev")
    t1, t2, ty = _wic_features(adapter, train)
    d1, d2, dy = _wic_features(adapter, dev)
    train_cos = F.cosine_similarity(t1, t2).numpy(); dev_cos = F.cosine_similarity(d1, d2).numpy()
    thresholds = np.unique(train_cos)
    accuracies = [((train_cos >= t) == ty).mean() for t in thresholds]
    threshold = float(thresholds[int(np.argmax(accuracies))])
    features = lambda a, b: torch.cat([(a - b).abs(), a * b], -1).numpy()
    w, b, mean, std = _fit_logistic(features(t1, t2), ty.astype(np.float64))
    dev_pred = (((features(d1, d2) - mean) / std) @ w + b) > 0
    return {"wic_prompt_accuracy": float(((dev_cos >= threshold) == dy).mean()),
            "wic_probe_accuracy": float((dev_pred == dy).mean()), "wic_majority": float(max(dy.mean(), 1 - dy.mean())),
            "n_dev": int(len(dy))}


# -- rare-word similarity -------------------------------------------------------------------------

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def word_similarity(adapter: ModelAdapter, pairs: list[tuple[str, str, float]], *, prefix: str = "The word") -> dict[str, float]:
    words = sorted({w for a, b, _ in pairs for w in (a, b)})
    texts = [f"{prefix} {w.replace('_', ' ')}" for w in words]
    spans = [(len(prefix) + 1, len(t)) for t in texts]
    states = adapter.word_state(texts, spans)
    index = {w: i for i, w in enumerate(words)}
    cos = np.asarray([float(F.cosine_similarity(states[index[a]], states[index[b]], dim=0)) for a, b, _ in pairs])
    gold = np.asarray([s for _, _, s in pairs])
    return {"spearman": _spearman(cos, gold), "n": len(pairs)}


def load_card660(path: Path) -> list[tuple[str, str, float]]:
    rows = [line.split("\t") for line in path.read_text().splitlines() if line.strip()]
    return [(r[0], r[1], float(r[2])) for r in rows if len(r) >= 3]


def load_rare_words(path: Path) -> list[tuple[str, str, float]]:
    rows = [line.split("\t") for line in path.read_text().splitlines() if line.strip()]
    return [(r[0], r[1], float(r[2])) for r in rows if len(r) >= 3]


def run_probes(adapter: ModelAdapter, root: Path, *, lambada_limit: int | None = 1000,
               wic_train_limit: int | None = None) -> dict[str, Any]:
    results: dict[str, Any] = {}
    results.update(lambada(adapter, root / "lambada" / "data" / "lambada_test_en.jsonl", limit=lambada_limit))
    results.update(wic(adapter, root / "wic", train_limit=wic_train_limit))
    results["card660"] = word_similarity(adapter, load_card660(root / "card660.tsv"))
    results["rare_words"] = word_similarity(adapter, load_rare_words(root / "rw" / "rw" / "rw.txt"))
    return results
