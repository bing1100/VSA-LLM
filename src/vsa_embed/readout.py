"""The unbinding readout (decision 60, step 2; pre-registration `experiments/e9-retrofit/preregistration-binding.md` §13).

A head that reads a role out of the most recent linked concept's frame store and injects the filler into the host's
residual stream. At every position t of a sequence:

1. **Concept.** The most recent linked span whose injection position p satisfies p ≤ t < p + `window` gives the concept
   e_t (prefix-causal: the span was linked from the text up to p, so position t never sees later text). Positions without
   one get nothing.
2. **Query.** A linear map of the host's hidden state h_t at layer L (the input of decoder layer L, read by a forward
   pre-hook) predicts a distribution over the R relations plus "no query".
3. **Unbind and clean up.** Every relation r is unbound from e_t's frame store with the composer's own operator
   (`FrameComposer.unbind`: learned HRR → correlation, unitary → conjugate, translation → subtraction, untyped → the
   bundle readout, which ignores the role) and cleaned up softly against the shared atomic dictionary
   (`cleanup.SoftCleanup`, a modern-Hopfield step with a learned inverse temperature; `typed`: restricted to the atomics
   observed under r in the frames). The store is the static frame bundle `Σ_e T_{r_e}(a_e)` (`source="static"`, every
   edge weight 1; `"attentive"`: the composer's attention weights without context).
4. **Inject.** The cleaned fillers are mixed by the predicted distribution ("no query" adds nothing), projected to model
   width and added to h_t through a gate `g = σ(w·[h_t; P f_t] + b)` with `w = 0` and `b` = the gate bias at
   initialization (0, as C5's channel gate), so the readout starts at half strength.

The readout is trained with the LM loss only. It shares the composer (atomics, operator) with the span channel; it is a
submodule of the channel (`SpanChannel.readout`), so it is saved and restored with it, and `ChannelLM` installs the hook.
"""

from __future__ import annotations

import contextlib
import math
from typing import Any, Iterator

import torch
from torch import Tensor, nn

from .cleanup import SoftCleanup, relation_candidates
from .relations import readout_method

SOURCES = ("static", "attentive")


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    """The decoder layers of a Hugging Face causal LM (Llama / Qwen: `model.layers`; GPT-2: `transformer.h`)."""
    inner = getattr(model, "model", None)
    if inner is not None and isinstance(getattr(inner, "layers", None), nn.ModuleList):
        return inner.layers
    transformer = getattr(model, "transformer", None)
    if transformer is not None and isinstance(getattr(transformer, "h", None), nn.ModuleList):
        return transformer.h
    raise ValueError(f"cannot find the decoder layers of {type(model).__name__}")


def readout_layer(spec: int | str, host: nn.Module | None) -> int:
    """The decoder layer whose input the readout reads and writes: an integer, or `"third"` = round(depth / 3)."""
    if isinstance(spec, int) or (isinstance(spec, str) and spec.lstrip("-").isdigit()):
        return int(spec)
    if spec != "third":
        raise ValueError("readout.layer must be an integer or 'third'")
    if host is None:
        raise ValueError("readout.layer 'third' needs the host model")
    depth = len(decoder_layers(host))
    return max(1, min(depth - 1, round(depth / 3)))


class UnbindingReadout(nn.Module):
    """See the module docstring. `composer` is shared with the span channel and is not registered here (its parameters
    belong to the channel's state)."""

    def __init__(self, composer: Any, model_dimension: int, *, layer: int, window: int = 32, gate_bias: float = 0.0,
                 beta: float = 16.0, steps: int = 1, typed: bool = True, source: str = "static") -> None:
        super().__init__()
        if source not in SOURCES:
            raise ValueError(f"readout source must be one of {SOURCES}")
        if window < 1:
            raise ValueError("readout window must be positive")
        self.__dict__["_composer"] = composer
        self.layer, self.window, self.typed, self.source = int(layer), int(window), bool(typed), source
        self.method = readout_method(composer.transform)
        relations, dimension = composer.relation_count, composer.atomics.shape[1]
        self.query = nn.Linear(model_dimension, relations + 1)
        nn.init.normal_(self.query.weight, std=model_dimension ** -0.5 * 0.1)
        nn.init.zeros_(self.query.bias)
        self.cleanup = SoftCleanup(beta=beta, steps=steps)
        self.projector = nn.Linear(dimension, model_dimension, bias=False)
        self.gate = nn.Linear(2 * model_dimension, 1)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, float(gate_bias))
        self.current: dict[str, Tensor] | None = None       # spans of the running forward (`ChannelLM.embed`)
        self.enabled = True                                  # False: the readout adds nothing (its gate off; evaluation)
        self.blocked: Tensor | None = None                  # entries whose rows are overridden at evaluation (no readout)
        self.capture: list[dict[str, Tensor]] | None = None  # diagnostics: per forward, the query and the readout
        self._mask_key: tuple[int, int, int] | None = None
        self._mask: Tensor | None = None

    @property
    def composer(self) -> Any:
        return self.__dict__["_composer"]

    def candidate_mask(self) -> Tensor:
        """`(relations, atomics)`: atomics observed under each relation in the composer's frames (a relation without
        any edge allows every atomic). Cached per schedule."""
        composer = self.composer
        key = (id(composer.frame_fillers), composer.frame_fillers.numel(), composer.atomics.shape[0])
        if self._mask is None or self._mask_key != key:
            mask = relation_candidates(composer.frame_relations, composer.frame_fillers, composer.relation_count,
                                       composer.atomics.shape[0])
            mask[~mask.any(-1)] = True
            self._mask, self._mask_key = mask, key
        return self._mask

    def filler_table(self, entries: Tensor) -> Tensor:
        """`(entries, relations, dimension)`: every relation unbound from each entry's store and cleaned up."""
        composer = self.composer
        bundle, *_ = composer.raw_bundle(entries, None, uniform=self.source == "static")
        count = composer.relation_count
        relation_ids = torch.arange(count, device=entries.device).repeat(entries.numel())
        unbound = composer.unbind(relation_ids, bundle.float().repeat_interleave(count, 0), method=self.method).float()
        mask = self.candidate_mask().to(entries.device)[relation_ids] if self.typed else None
        atomics = composer.atomic_vectors().float()
        slot_masks = composer.slot_masks() if hasattr(composer, "slot_masks") else None
        if slot_masks is None:
            cleaned = self.cleanup(unbound, atomics, mask)
        else:                                  # slotted layout: each relation cleans up within its own slot
            cleaned = torch.zeros_like(unbound)
            slots = composer.slot_of().to(entries.device)[relation_ids]
            for g in range(slot_masks.shape[0]):
                rows = (slots == g).nonzero(as_tuple=True)[0]
                if rows.numel():
                    dictionary = atomics * slot_masks[g].to(atomics.device)
                    cleaned[rows] = self.cleanup(unbound[rows], dictionary, None if mask is None else mask[rows]).to(cleaned.dtype)
        return cleaned.view(entries.numel(), count, -1)

    def locate(self, batch: int, length: int, spans: dict[str, Tensor], device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
        """(rows, columns, span index) of every position with a linked span injected at most `window − 1` tokens before
        it (the most recent one)."""
        inject = spans["inject"].to(device)
        rows_of = spans["batch"].to(device)
        keep = inject < length
        if self.blocked is not None:
            keep &= ~torch.isin(spans["entry"].to(device), self.blocked.to(device))
        span_at = torch.full((batch, length), -1, dtype=torch.long, device=device)
        span_at[rows_of[keep], inject[keep]] = torch.arange(inject.numel(), device=device)[keep]
        steps = torch.arange(length, device=device).expand(batch, length)
        recent = torch.cummax(torch.where(span_at >= 0, steps, torch.full_like(steps, -1)), dim=1).values
        valid = (recent >= 0) & (steps - recent < self.window)
        rows, cols = valid.nonzero(as_tuple=True)
        return rows, cols, span_at[rows, recent[rows, cols]]

    def forward(self, hidden: Tensor) -> Tensor | None:
        """The addition to `hidden` `(batch, time, width)` (None when no position has a recent concept)."""
        spans = self.current
        if spans is None or spans["entry"].numel() == 0:
            return None
        rows, cols, span = self.locate(hidden.shape[0], hidden.shape[1], spans, hidden.device)
        if rows.numel() == 0:
            return None
        concepts = spans["entry"].to(hidden.device)[span]
        unique, inverse = torch.unique(concepts, return_inverse=True)
        fillers = self.filler_table(unique)
        states = hidden[rows, cols]
        probabilities = torch.softmax(self.query(states).float(), -1)
        mixed = torch.einsum("nr,nrd->nd", probabilities[:, :-1], fillers[inverse])
        projected = self.projector(mixed.to(self.projector.weight.dtype))
        gate = torch.sigmoid(self.gate(torch.cat([states, projected.to(states.dtype)], -1)))
        addition = torch.zeros_like(hidden)
        addition[rows, cols] = (gate * projected).to(hidden.dtype)
        if self.capture is not None:
            self.capture.append({"rows": rows.detach(), "cols": cols.detach(), "concepts": concepts.detach(),
                                 "probabilities": probabilities.detach(), "mixed": mixed.detach(), "gate": gate.detach().squeeze(-1)})
        return addition


def readout_from_config(settings: dict[str, Any], composer: Any, width: int, host: nn.Module | None, *, seed: int) -> UnbindingReadout:
    """Build the readout of `channel.readout` (opt-in). Its initialization draws from its own generator, so the global
    random stream — and with it every other initialization of the run — is the same as without the readout."""
    spec = dict(settings)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) * 7919 + 17)
        return UnbindingReadout(composer, width, layer=readout_layer(spec.get("layer", "third"), host),
                                window=int(spec.get("window", 32)), gate_bias=float(spec.get("gate_bias", 0.0)),
                                beta=float(spec.get("beta", 16.0)), steps=int(spec.get("steps", 1)),
                                typed=bool(spec.get("typed", True)), source=str(spec.get("source", "static")))


@contextlib.contextmanager
def readout_disabled(model: nn.Module) -> Iterator[bool]:
    """Within the block the model's readout adds nothing (its gate switched off: the operator's loss without the
    readout, decision 61); yields False when the model has no readout."""
    readout = getattr(getattr(model, "channel", None), "readout", None)
    if readout is None:
        yield False
        return
    previous = readout.enabled
    readout.enabled = False
    try:
        yield True
    finally:
        readout.enabled = previous


def mean_reciprocal(rank: Tensor) -> float:
    return float((1.0 / rank).mean()) if rank.numel() else math.nan
