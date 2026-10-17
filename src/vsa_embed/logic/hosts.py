"""Frozen host language models for E14 scoring (plan §1 hosts H-135, H-360, H-q06, H-q17): continuation log-probabilities
(forced choice, cloze, scored choice of reader and prover steps) and greedy generation (the Qwen3 ReAct and chain-of-thought
policies). Plain `transformers` models from the local cache (`local_files_only`); nothing trains.

`HOSTS` maps the plan's host ids to Hugging Face names; any other name or a local path is accepted as is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

HOSTS = {"H-135": "HuggingFaceTB/SmolLM2-135M", "H-360": "HuggingFaceTB/SmolLM2-360M", "H-q06": "Qwen/Qwen3-0.6B-Base",
         "H-q17": "Qwen/Qwen3-1.7B-Base"}


@dataclass
class Scores:
    logprob: np.ndarray          # summed log-probability of each continuation
    tokens: np.ndarray           # its token count

    @property
    def mean(self) -> np.ndarray:
        return self.logprob / np.maximum(self.tokens, 1)


class HostLM:
    def __init__(self, name: str, *, device: str | None = None, dtype: str = "auto", batch_size: int = 16,
                 max_length: int | None = None) -> None:
        import transformers
        self.name = HOSTS.get(name, name)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        kwargs = {"local_files_only": True}
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.name, **kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        torch_dtype = {"auto": torch.bfloat16 if self.device.type == "cuda" else torch.float32, "float32": torch.float32,
                       "bfloat16": torch.bfloat16}[dtype]
        self.model = transformers.AutoModelForCausalLM.from_pretrained(self.name, torch_dtype=torch_dtype, **kwargs).to(self.device)
        self.model.eval()
        config = self.model.config
        self.max_length = max_length or int(getattr(config, "max_position_embeddings", None) or getattr(config, "n_positions", 2048))
        self.batch_size = int(batch_size)

    def describe(self) -> dict[str, object]:
        return {"host": self.name, "device": str(self.device), "dtype": str(next(self.model.parameters()).dtype),
                "max_length": self.max_length}

    def count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _ids(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def score(self, prompts: Sequence[str], continuations: Sequence[str]) -> Scores:
        """log p(continuation | prompt) for each pair (prompt and continuation tokenized separately and joined, so the
        continuation's tokens are the same under every prompt). Over-long inputs keep their right end. Sequences are
        left-padded (explicit position ids), so every continuation ends at the last position and only the last positions'
        logits are computed (`logits_to_keep`): memory stays small for large vocabularies."""
        if len(prompts) != len(continuations):
            raise ValueError("prompts and continuations differ in length")
        pairs = [(self._ids(p), self._ids(c)) for p, c in zip(prompts, continuations)]
        bos = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None and "Qwen" not in self.name else []
        out_lp = np.zeros(len(pairs)); out_n = np.zeros(len(pairs), dtype=int)
        order = sorted(range(len(pairs)), key=lambda i: -(len(pairs[i][0]) + len(pairs[i][1])))
        for start in range(0, len(order), self.batch_size):
            batch = order[start:start + self.batch_size]
            seqs, cont = [], []
            for i in batch:
                p, c = pairs[i]
                ids = (bos + p + c)[-self.max_length:]
                seqs.append(ids)
                cont.append(min(len(c), len(ids) - 1))
            width = max(len(s) for s in seqs)
            input_ids = torch.full((len(seqs), width), self.tokenizer.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(seqs), width), dtype=torch.long)
            for row, s in enumerate(seqs):
                input_ids[row, width - len(s):] = torch.tensor(s)
                mask[row, width - len(s):] = 1
            positions = (mask.cumsum(-1) - 1).clamp_min(0)
            keep = max(cont) + 1
            kwargs = {"input_ids": input_ids.to(self.device), "attention_mask": mask.to(self.device),
                      "position_ids": positions.to(self.device)}
            try:
                logits = self.model(**kwargs, logits_to_keep=keep).logits
            except TypeError:
                logits = self.model(**kwargs).logits[:, -keep:]
            logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)          # predicts the last keep-1 tokens
            targets = input_ids[:, width - keep + 1:].to(self.device)
            token_lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).cpu().numpy()
            for row, i in enumerate(batch):
                n = cont[row]
                out_lp[i] = float(token_lp[row, token_lp.shape[1] - n:].sum()) if n else 0.0
                out_n[i] = n
        return Scores(out_lp, out_n)

    def choose(self, prompt: str, options: Sequence[str], *, normalize: bool = True, null: str | None = None) -> tuple[int, np.ndarray]:
        """Index of the best option after `prompt` (length-normalised log-probability; with `null`, minus each option's
        log-probability after the null prompt: the PMI calibration E12 uses) and the option scores."""
        s = self.score([prompt] * len(options), list(options))
        values = s.mean if normalize else s.logprob
        if null is not None:
            n = self.score([null] * len(options), list(options))
            values = values - (n.mean if normalize else n.logprob)
        return int(np.argmax(values)), values

    @torch.no_grad()
    def generate(self, prompts: Sequence[str], *, max_new_tokens: int = 64, stop: Sequence[str] = ("\n\n",)) -> list[str]:
        """Greedy continuations, cut at the first stop string."""
        out = []
        tok = self.tokenizer
        tok.padding_side = "left"
        for start in range(0, len(prompts), self.batch_size):
            batch = list(prompts[start:start + self.batch_size])
            enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False, truncation=True,
                      max_length=self.max_length - max_new_tokens).to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            for row in gen[:, enc["input_ids"].shape[1]:]:
                text = tok.decode(row, skip_special_tokens=True)
                cut = min([text.find(s) for s in stop if s in text] or [len(text)])
                out.append(text[:cut])
        tok.padding_side = "right"
        return out
