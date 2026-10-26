"""HRR as a training target (E15 "HT"; pre-registration `experiments/e15-hrr-target/preregistration.md`).

In C5 the composed HRR vector is only an input feature: no loss touches its structure from the model's side, so the
host learns "this vector identifies the term" (C5 ≈ C5sh ≈ C2w, status 06.3–06.4; the readout arms of 09.4 confirm it).
`HRRTarget` adds two losses that only the algebra can satisfy, computed from the hidden state `h` at the term's last
subtoken at the input of decoder layer ℓ (the readout layer of 09.4: `readout.readout_layer("third")`, 11 of 32 for
SmolLM2-360M), captured by a forward pre-hook that `ChannelLM` installs (it changes nothing in the forward):

1. **Write loss.** A linear map `W` (host width → store dimension) must reproduce the term's static frame bundle
   `b = N(Σ_e T_{r_e}(a_e))` (`FrameComposer.raw_bundle(uniform=True)`, the store the readout arms read):
   `L_write = 1 − cos(W h, b)`.
2. **Read loss.** For every edge `(r, f)` of the term's frame, `û = unbind_r(W h)` with the store's own operator
   (`FrameComposer.unbind`), and `L_read = CE(softmax(N(û) · Aᵀ / τ), f)` over the unit-normalised atom table `A` —
   Plate's cleanup memory as a trained objective (τ fixed, 0.1 by default; chance = ln |A|).

`detach_target` (default True) treats the store as the target: the bundle, the roles and the atoms carry no gradient
from these losses (the host and `W` learn to match the store, the store does not collapse to the model). The store
still trains through the LM loss as in C5, and gradients of the two losses reach the channel through the injected row
(`h` depends on it). A compose channel shares its composer as the decoder; a free channel (C2+R, the key control) gets
its own frozen random HRR decoder built from the ontology's frames (`own_decoder`): a fixed code the host must learn to
write its free row into — possible for seen terms by memorisation, impossible to generalise to a term never seen.

The total objective is `L_LM + λ_w L_write + λ_r L_read` (`write_weight`, `read_weight`; `training.lm`). `W` draws its
initialisation from its own generator, so every other initialisation of the run equals the run without the block.

**Stage 1e, "read, then inject" (`channel.target.inject`, opt-in; pre-registration "Stage 1e").** In stages 1–1d the
decoded facts went to the external head `W` and nowhere else, so the LM loss could not learn to use them. The injection
path feeds them back: from the same captured state `h` (input of layer ℓ), for every relation `r` present in the term's
frame (the frame the run injects; the roles are known at the term, the fillers are what is decoded — they are never
read), `cleanup(unbind_r(W h)) = softmax(N(û) · Aᵀ / τ) · A` (the soft cleanup of the read loss, τ the same; gradient
flows to `W`, and the content stays in atom space), **summed over the relations** (frame-size independent), projected by
a learned linear map `P` (store dimension → host width, **zero-initialised**) and added to the hidden state at the term's
position **at the input of decoder layer ℓ + 1** through a scalar gate `g = σ(b)` with `b` initialised to −4
(`gate_bias`; σ(−4) = 0.018, near-closed; `gate: false` fixes g ≡ 1). Nothing on this path is detached: the LM loss
reaches `P`, `b`, `W`, the host (through `h`) and the store (through the cleanup, as it does through the injected row in
C5); the write and read losses keep `detach_target` and keep `W` a decoder. `FactInjection` holds `P` and `b`;
`HRRTarget.injection` computes the addition; `ChannelLM` installs the pre-hook on layer ℓ + 1 (the capture at layer ℓ is
therefore always pre-injection, and `e15_eval read` reads layer ℓ unchanged). `set_injection(model, False)` closes the
gate at evaluation (`--gate-closed`: the injection ablation on the trained model).
"""

from __future__ import annotations

import contextlib
import math
from typing import Any, Iterator, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .algebra import normalize
from .compose import FrameComposer, FrameSchedule, _occurrence_edges


@contextlib.contextmanager
def frozen(module: nn.Module) -> Iterator[None]:
    """Within the block the module's parameters record no gradient (operations on them are treated as constants);
    their `requires_grad` flags are restored on exit. Earlier operations of the same graph keep their gradient."""
    flags = [(p, p.requires_grad) for p in module.parameters()]
    for p, _ in flags:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)


class FactInjection(nn.Module):
    """Stage 1e: the projection `P` (store dimension → host width, zero-initialised) and the scalar gate `g = σ(b)` that
    add the decoded content to the hidden state (`gated=False`: g ≡ 1, no parameter — the `nogate` arm). `enabled`
    False switches the addition off (gate closed at evaluation)."""

    def __init__(self, dimension: int, model_dimension: int, *, gate_bias: float = -4.0, gated: bool = True) -> None:
        super().__init__()
        if not math.isfinite(gate_bias):
            raise ValueError("inject.gate_bias must be finite")
        self.project = nn.Linear(int(dimension), int(model_dimension), bias=False)
        nn.init.zeros_(self.project.weight)
        self.gated, self.gate_bias = bool(gated), float(gate_bias)
        if self.gated:
            self.gate_logit = nn.Parameter(torch.tensor(float(gate_bias)))
        self.enabled = True

    def gate(self) -> Tensor:
        """`g` as a scalar tensor (differentiable when gated)."""
        return torch.sigmoid(self.gate_logit) if self.gated else self.project.weight.new_ones(())

    def gate_value(self) -> float:
        return float(self.gate().detach())

    def forward(self, content: Tensor) -> Tensor:
        """`g · P content` in float32 (`content`: `(occurrences, dimension)`)."""
        projected = self.project(content.to(self.project.weight.dtype)).float()
        return self.gate().float() * projected

    def record(self) -> dict[str, Any]:
        return {"gate": self.gate_value(), "gate_bias": self.gate_bias, "gated": self.gated, "enabled": bool(self.enabled)}


class HRRTarget(nn.Module):
    """See the module docstring. `decoder`: the channel's composer (shared, not registered here) or, with
    `own_decoder=True`, a composer owned by this module (registered, saved with the channel's state). `inject` (stage
    1e, optional; set after construction by `target_from_config` so that `W` draws the same values as without it) adds
    the decoded fillers back into the residual stream at layer ℓ + 1."""

    def __init__(self, decoder: FrameComposer, model_dimension: int, *, layer: int, tau: float = 0.1,
                 write_weight: float = 0.0, read_weight: float = 0.0, detach_target: bool = True,
                 own_decoder: bool = False) -> None:
        super().__init__()
        if tau <= 0:
            raise ValueError("tau must be positive")
        if write_weight < 0 or read_weight < 0 or write_weight == read_weight == 0:
            raise ValueError("write_weight and read_weight must be non-negative and not both zero")
        self.layer, self.tau = int(layer), float(tau)
        self.write_weight, self.read_weight, self.detach_target = float(write_weight), float(read_weight), bool(detach_target)
        self.own_decoder = bool(own_decoder)
        if own_decoder:
            self.decoder_module = decoder
        else:
            self.__dict__["_decoder"] = decoder
        dimension = decoder.atomics.shape[1]
        self.write = nn.Linear(model_dimension, dimension, bias=False)
        nn.init.normal_(self.write.weight, std=model_dimension ** -0.5)
        self.current: dict[str, Tensor] | None = None       # spans of the running forward (`ChannelLM.embed`)
        self.captured: dict[str, Tensor] | None = None      # states at the linked terms' injection positions
        self.inject: FactInjection | None = None            # stage 1e (opt-in): the "read, then inject" path

    @property
    def decoder(self) -> FrameComposer:
        return self.decoder_module if self.own_decoder else self.__dict__["_decoder"]

    # -- capture (the forward pre-hook of decoder layer ℓ) ---------------------------------------------------------

    def capture(self, hidden: Tensor) -> None:
        """Keep `h` at every linked span's injection position of this forward (batch, inject from `current`)."""
        spans = self.current
        if spans is None or spans["entry"].numel() == 0:
            self.captured = None
            return
        rows, cols = spans["batch"].to(hidden.device), spans["inject"].to(hidden.device)
        keep = cols < hidden.shape[1]
        self.captured = {"states": hidden[rows[keep], cols[keep]], "entries": spans["entry"].to(hidden.device)[keep],
                         "rows": rows[keep], "cols": cols[keep]}

    # -- pieces -------------------------------------------------------------------------------------------------------

    def codes(self, states: Tensor) -> Tensor:
        """`W h` in float32."""
        return self.write(states.to(self.write.weight.dtype)).float()

    def bundles(self, entries: Tensor) -> Tensor:
        """The static frame bundle of each entry, unit-normalised (float32)."""
        summed, *_ = self.decoder.raw_bundle(entries, None, uniform=True)
        return normalize(summed.float())

    def frame_edges(self, entries: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """(relations, fillers, occurrence index) of every edge of the entries' frames in the decoder's schedule."""
        schedule = self.decoder.schedule
        edge_index, segments = _occurrence_edges(schedule, entries)
        return schedule.relations[edge_index], schedule.fillers[edge_index], segments

    def read_logits(self, codes: Tensor, relations: Tensor, segments: Tensor) -> Tensor:
        """`(edges, atoms)` cleanup logits: cosine of the unbound code with every atom, over τ."""
        unbound = self.decoder.unbind(relations, codes[segments]).float()
        atoms = normalize(self.decoder.atomic_vectors().float())
        return normalize(unbound) @ atoms.T / self.tau

    # -- stage 1e: read, then inject -----------------------------------------------------------------------------------

    def decoded_content(self, states: Tensor, entries: Tensor) -> Tensor:
        """`(occurrences, dimension)`: for every relation of each entry's frame (the decoder's schedule; one unbinding
        per distinct relation), the soft cleanup `softmax(N(unbind_r(W h)) · Aᵀ / τ) · A`, summed over the relations.
        The frame's fillers are not read. Entries outside the schedule (zero-shot insertions of a free channel) get a
        zero row. Differentiable end to end (W, the store, and `h`)."""
        codes = self.codes(states)
        schedule = self.decoder.schedule
        inside = entries < schedule.concept_count
        content = codes.new_zeros(entries.numel(), codes.shape[1])
        if not bool(inside.any()):
            return content
        index = inside.nonzero(as_tuple=True)[0]
        relations, _, segments = self.frame_edges(entries[index])
        if relations.numel() == 0:
            return content
        count = int(self.decoder.relation_count)
        pairs = torch.unique(segments * count + relations)                 # one unbinding per (occurrence, relation)
        segments, relations = pairs // count, pairs % count
        logits = self.read_logits(codes[index], relations, segments)
        atoms = normalize(self.decoder.atomic_vectors().float())
        cleaned = torch.softmax(logits, -1) @ atoms
        # `codes` (and so `content`) follow the host's autocast dtype on the GPU (bf16); the cleanup is computed in fp32.
        return content.index_add(0, index[segments], cleaned.to(content.dtype))

    def injection(self, hidden: Tensor) -> Tensor | None:
        """The addition to the input of layer ℓ + 1 `(batch, time, width)` from the capture of this forward: zeros
        except `g · P content` at the linked terms' positions; None without injection, with the gate closed, or without
        a linked span."""
        inject, captured = self.inject, self.captured
        if inject is None or not inject.enabled or captured is None or captured["entries"].numel() == 0 or "rows" not in captured:
            return None
        content = self.decoded_content(captured["states"], captured["entries"])
        addition = torch.zeros_like(hidden)
        addition.index_put_((captured["rows"], captured["cols"]), inject(content).to(hidden.dtype), accumulate=True)
        return addition

    @staticmethod
    def edges_of(frames: Sequence[Sequence[tuple[int, int]]], device: torch.device | str) -> tuple[Tensor, Tensor, Tensor]:
        """(relations, fillers, occurrence index) of explicit frames (evaluation of terms outside the schedule)."""
        relations, fillers, segments = [], [], []
        for i, frame in enumerate(frames):
            for r, f in frame:
                relations.append(int(r)); fillers.append(int(f)); segments.append(i)
        make = lambda v: torch.tensor(v, dtype=torch.long, device=device)  # noqa: E731
        return make(relations), make(fillers), make(segments)

    # -- losses -------------------------------------------------------------------------------------------------------

    def losses(self, captured: dict[str, Tensor] | None = None) -> dict[str, Tensor]:
        """`{"write", "read", "occurrences", "edges"}` of the captured forward (zeros when no span was linked); the
        capture is consumed. Under `detach_target` the decoder's parameters are constants here."""
        captured = self.captured if captured is None else captured
        self.captured = None
        zero = self.write.weight.new_zeros(())
        if captured is None or captured["entries"].numel() == 0:
            return {"write": zero, "read": zero, "occurrences": 0, "edges": 0}
        with frozen(self.decoder) if self.detach_target else contextlib.nullcontext():
            return self._losses(captured["states"], captured["entries"])

    def _losses(self, states: Tensor, entries: Tensor) -> dict[str, Tensor]:
        codes = self.codes(states)
        write = (1.0 - F.cosine_similarity(codes, self.bundles(entries), dim=-1)).mean() if self.write_weight else codes.new_zeros(())
        relations, fillers, segments = self.frame_edges(entries)
        read = (F.cross_entropy(self.read_logits(codes, relations, segments), fillers) if self.read_weight
                else codes.new_zeros(()))
        return {"write": write, "read": read, "occurrences": int(entries.numel()), "edges": int(fillers.numel())}

    # -- evaluation ---------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def read_frames(self, states: Tensor, frames: Sequence[Sequence[tuple[int, int]]], *, codes: Tensor | None = None,
                    candidates: Tensor | None = None) -> dict[str, Tensor]:
        """Filler recovery from `W h` (or given `codes`) for explicit frames, one state per frame: per edge the
        relation, the gold filler, the top-1 atom over the whole table, the gold atom's rank (1 = best) and, with a
        `(relations, atoms)` candidate mask, the typed top-1."""
        codes = self.codes(states) if codes is None else codes.float()
        relations, fillers, segments = self.edges_of(frames, codes.device)
        if fillers.numel() == 0:
            empty = torch.zeros(0, dtype=torch.long, device=codes.device)
            return {"relation": empty, "filler": empty, "top1": empty, "rank": empty, "typed_top1": empty, "occurrence": empty}
        logits = self.read_logits(codes, relations, segments)
        gold = logits.gather(1, fillers[:, None])
        out = {"relation": relations, "filler": fillers, "top1": logits.argmax(1), "rank": 1 + (logits > gold).sum(1),
               "occurrence": segments}
        if candidates is not None:
            masked = logits.masked_fill(~candidates.to(logits.device)[relations], float("-inf"))
            out["typed_top1"] = masked.argmax(1)
        return out


def injection_of(model: nn.Module) -> FactInjection | None:
    """The stage-1e injection of a `ChannelLM` (None when the run has none)."""
    target = getattr(getattr(model, "channel", None), "target", None)
    return getattr(target, "inject", None) if target is not None else None


def set_injection(model: nn.Module, enabled: bool) -> bool:
    """Switch the injection on or off (`enabled=False`: the gate closed at evaluation); False when the run has none."""
    inject = injection_of(model)
    if inject is None:
        return False
    inject.enabled = bool(enabled)
    return True


@contextlib.contextmanager
def injection_closed(model: nn.Module) -> Iterator[bool]:
    """Within the block the injection adds nothing; yields False when the run has none."""
    inject = injection_of(model)
    if inject is None:
        yield False
        return
    previous = inject.enabled
    inject.enabled = False
    try:
        yield True
    finally:
        inject.enabled = previous


def injection_record(model: nn.Module) -> dict[str, Any] | None:
    """`{"gate", "gate_bias", "gated", "enabled", "layer"}` of the run's injection (layer: the decoder layer whose input
    receives it), None without one."""
    inject = injection_of(model)
    if inject is None:
        return None
    return {**inject.record(), "layer": int(model.channel.target.layer) + 1}


def ontology_decoder(ontology: dict[str, Any], dimension: int, *, seed: int) -> FrameComposer:
    """A frozen random HRR decoder over the ontology's frames (the free channel's fixed code; seeded on its own)."""
    schedule = FrameSchedule(ontology["offsets"], ontology["relations"], ontology["fillers"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) * 7919 + 29)
        decoder = FrameComposer(schedule, int(ontology["atomic_count"]), int(ontology["relation_count"]), int(dimension),
                                operator="hrr", mode="bundle")
    decoder.requires_grad_(False)
    return decoder


def target_from_config(settings: dict[str, Any], channel: Any, ontology: dict[str, Any], width: int,
                       host: nn.Module | None, *, seed: int) -> HRRTarget:
    """Build the target of `channel.target` (opt-in): `{layer: "third" | int, tau: 0.1, write_weight, read_weight,
    detach_target: true, dimension: 256 (free channels: the decoder's width), inject: null}`. A compose channel shares
    its composer; a free channel gets `ontology_decoder`. `W` draws from its own generator (see the module docstring).
    `inject` (stage 1e; `{gate_bias: -4.0, gate: true}` or `true`) adds the "read, then inject" path, built after `W` in
    the same generator so that `W` equals the run without it."""
    from .readout import readout_layer
    spec = dict(settings)
    layer = readout_layer(spec.get("layer", "third"), host)
    if channel.mode == "compose":
        decoder, own = channel.composer, False
    elif channel.mode == "free":
        decoder, own = ontology_decoder(ontology, int(spec.get("dimension", 256)), seed=seed), True
    else:
        raise ValueError(f"channel.target supports the compose and free channels, not {channel.mode!r}")
    inject = spec.get("inject")
    if inject is not None and not isinstance(inject, (dict, bool)):
        raise ValueError("channel.target.inject must be a mapping ({gate_bias, gate}), true, or absent")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) * 7919 + 23)
        target = HRRTarget(decoder, width, layer=layer, tau=float(spec.get("tau", 0.1)),
                           write_weight=float(spec.get("write_weight", 0.0)), read_weight=float(spec.get("read_weight", 0.0)),
                           detach_target=bool(spec.get("detach_target", True)), own_decoder=own)
        if inject:
            options = inject if isinstance(inject, dict) else {}
            target.inject = FactInjection(decoder.atomics.shape[1], width, gate_bias=float(options.get("gate_bias", -4.0)),
                                          gated=bool(options.get("gate", True)))
        return target
