"""Weight-editing baselines for E9 dimension 3 (claim C): ROME, MEMIT and AlphaEdit.

Minimal re-implementations of the three locate-then-edit methods on the MLP output projection of a
causal LM — `mlp.down_proj` of LLaMA-style hosts (SmolLM2, Qwen3, Qwen3.5; `LoRALinear`-wrapped
projections are edited through their base weight) and `mlp.c_proj` of GPT-2 (tests). They follow the
reference algorithms:

- **ROME** (Meng, Bau, Andonian & Belinkov, *Locating and Editing Factual Associations in GPT*,
  NeurIPS 2022, arXiv:2202.05262): one rank-one update `ΔW = (v* − W k*) (C⁻¹ k̄)ᵀ / ((C⁻¹ k̄)ᵀ k*)`
  of one layer's output projection. `k̄` is the projection's input at the subject's last token averaged
  over context prefixes, `k*` the same on the plain prompt, `C = E[k kᵀ]` the uncentred second moment of
  the keys on sample text, and `v* = W k* + δ*`, where `δ*` (added to the projection's output at the
  subject's last token) minimizes the target's negative log-likelihood over the context prefixes plus
  `kl_factor ·` KL of the next-token distribution after "{subject} is a" and a relative weight decay,
  with `‖δ‖ ≤ clamp_norm_factor · ‖W k*‖`.
- **MEMIT** (Meng, Sen Sharma, Andonian, Belinkov & Bau, *Mass-Editing Memory in a Transformer*, ICLR
  2023, arXiv:2210.07229): a batch of edits spread over a range of layers. Per edit a target residual
  `z = h_L + δ` is optimized at the output of the last layer `L` of the range (same loss); then, layer
  by layer, `ΔW_l = R_l K_lᵀ (λ C_l + K_l K_lᵀ)⁻¹` with keys `K_l` (averaged over context prefixes) and
  the residual `R_l = (z − h_L) / (layers left)` recomputed after each layer's update.
- **AlphaEdit** (Fang et al., *AlphaEdit: Null-Space Constrained
  Knowledge Editing for Language Models*, ICLR 2025, arXiv:2410.02355): MEMIT's targets and keys, with
  the update projected onto the null space of the preserved keys, `P = U₀U₀ᵀ` over the eigenvectors of
  `C_l` with eigenvalue below `nullspace_threshold`: `ΔW_l = R_l K_lᵀ P (K_l K_lᵀ P + l2 · I)⁻¹`
  (batch editing, no earlier edits to protect).

Defaults are EasyEdit's LLaMA-7B hyper-parameters (Wang et al., *EasyEdit*, arXiv:2308.07269), with the
edited layers scaled to the host's depth (`default_layers`: ROME layer 5 of 32, MEMIT/AlphaEdit layers
4–8 of 32).

**Deviations from the reference code** (recorded in every output by `EditorSettings.record`):
- context prefixes are a fixed list (`CONTEXT_PREFIXES`) instead of ten texts sampled from the model,
  so edits are deterministic;
- the second moment `C` is estimated from `cov_tokens` tokens of the run's own training stream (ROME:
  100,000 Wikipedia texts), with a ridge of `1e-5 · mean(diag C)` before ROME's and MEMIT's inversions
  (the reference solves MEMIT's `λC + KKᵀ` without one, which is singular when `C` is rank-deficient);
  AlphaEdit's null space is taken from the unridged `C` (its solve has `l2 · I`);
- `δ` is optimized in float32 without autocast; scoring then runs as every other E9 evaluation;
- AlphaEdit's absolute eigenvalue threshold is kept (2e-2) but the null-space dimension is recorded, and
  `nullspace_relative` switches to a threshold relative to the largest eigenvalue for hosts whose key
  scale differs from LLaMA-3's.
"""

from __future__ import annotations

import contextlib
import dataclasses
from typing import Any, Callable, Iterator, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .evaluation.channel_probes import read_index

# "{}" is the plain prompt; the others stand in for ROME's model-generated context texts ("<text>. {}").
CONTEXT_PREFIXES = ("{}", "The meeting was moved to Monday. {}", "Therefore, the plan changed. {}",
                    "Because of this, we waited. {}", "I read the notes again. {}", "You can ask the office. {}",
                    "The report was short and clear. {}", "In the end, it worked. {}", "She checked the list twice. {}",
                    "We agreed on the next steps. {}")
METHODS = ("rome", "memit", "alphaedit")


@dataclasses.dataclass(frozen=True)
class EditRequest:
    """One edit: the rewrite prompt (`{x}` marks the subject), the subject and the new target
    (a continuation with its leading space)."""

    prompt: str
    subject: str
    target: str
    key: str = ""


@dataclasses.dataclass
class EditorSettings:
    method: str = "rome"
    layers: tuple[int, ...] | None = None          # None: `default_layers`
    v_steps: int = 25
    v_lr: float = 0.5
    v_weight_decay: float = 1e-3
    clamp_norm_factor: float = 4.0
    kl_factor: float = 0.0625
    kl_prompt: str = "{x} is a"
    mom2_update_weight: float = 15000.0           # MEMIT λ
    nullspace_threshold: float = 2e-2             # AlphaEdit
    nullspace_relative: float | None = None       # e.g. 1e-2: eigenvalues below 1% of the largest
    l2: float = 10.0                              # AlphaEdit
    context_prefixes: tuple[str, ...] = CONTEXT_PREFIXES
    early_stop: float = 5e-2
    batch_size: int = 16

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown editor {self.method!r}; choose from {', '.join(METHODS)}")
        if not self.context_prefixes or self.context_prefixes[0] != "{}":
            raise ValueError("the first context prefix must be the plain prompt '{}'")

    def record(self, layers: Sequence[int]) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["layers"] = list(layers)
        out["context_prefixes"] = list(self.context_prefixes)
        out["deviations"] = ["fixed context prefixes (not model-generated)", "second moment from the run's training stream",
                             "delta optimized in float32 without autocast",
                             "ridge 1e-5·mean(diag C) on C before inversion (ROME C⁻¹, MEMIT λC + KKᵀ; not AlphaEdit's null space)"]
        return out


def default_layers(method: str, depth: int) -> tuple[int, ...]:
    """EasyEdit's LLaMA-7B layers (ROME 5; MEMIT/AlphaEdit 4–8, of 32) scaled to `depth` layers."""
    scale = depth / 32
    if method == "rome":
        return (min(depth - 1, max(0, round(5 * scale))),)
    lo, hi = max(0, round(4 * scale)), min(depth - 1, max(0, round(8 * scale)))
    return tuple(range(lo, max(lo, hi) + 1))


# -- the host behind an evaluation adapter ----------------------------------------------------------

class EditableHost:
    """The causal LM of a probe adapter (`ModelAdapter` / `ChannelModelAdapter`): decoder layers, MLP
    output projections, and a differentiable forward that goes through the span channel when there is
    one (so a channel model is edited with its channel live)."""

    def __init__(self, adapter: Any) -> None:
        if (getattr(adapter, "info", None) or {}).get("quantization"):
            raise ValueError("weight editing needs the unquantized model")
        self.adapter = adapter
        model = adapter.model
        self.channel_model = hasattr(model, "channel") and hasattr(model, "embed")
        self.host = model.model if self.channel_model else model
        self.device = adapter.device

    @property
    def layers(self) -> Any:
        inner = getattr(self.host, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            return inner.layers
        transformer = getattr(self.host, "transformer", None)
        if transformer is not None and hasattr(transformer, "h"):
            return transformer.h
        raise ValueError(f"cannot find the decoder layers of {type(self.host).__name__}")

    @property
    def depth(self) -> int:
        return len(self.layers)

    def site(self, layer: int) -> torch.nn.Module:
        """The MLP output projection of `layer` (the module ROME/MEMIT rewrite)."""
        mlp = self.layers[layer].mlp
        for name in ("down_proj", "c_proj"):
            if hasattr(mlp, name):
                return getattr(mlp, name)
        raise ValueError(f"layer {layer} has no MLP output projection")

    def weight(self, layer: int) -> tuple[torch.nn.Parameter, bool]:
        """(the 2-D weight the update is added to, transposed?) — Linear (out, in); GPT-2 Conv1D (in, out)."""
        site = self.site(layer)
        base = getattr(site, "base", site)               # LoRALinear: the frozen base projection
        return base.weight, type(base).__name__ == "Conv1D"

    @torch.no_grad()
    def add_update(self, layer: int, update: Tensor) -> None:
        """W ← W + update, `update` in (out, in) orientation."""
        weight, transposed = self.weight(layer)
        weight.add_((update.T if transposed else update).to(weight.device, weight.dtype))

    def snapshot(self, layers: Sequence[int]) -> dict[int, Tensor]:
        return {l: self.weight(l)[0].detach().clone() for l in layers}

    @torch.no_grad()
    def restore(self, saved: dict[int, Tensor]) -> None:
        for layer, value in saved.items():
            self.weight(layer)[0].copy_(value)

    # -- forward --------------------------------------------------------------------------------

    def encode(self, texts: Sequence[str]) -> tuple[dict[str, Tensor], list[list[tuple[int, int]]], dict[str, Tensor] | None]:
        tokenizer = self.adapter.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        encoded = tokenizer(list(texts), return_offsets_mapping=True, add_special_tokens=False, padding=True,
                            truncation=True, max_length=self.adapter.max_length, return_tensors="pt")
        offsets = [[tuple(o) for o, m in zip(offs.tolist(), mask.tolist()) if m]
                   for offs, mask in zip(encoded["offset_mapping"], encoded["attention_mask"])]
        spans = None
        if self.channel_model and self.adapter.spans_fn is not None:
            spans = {k: v.to(self.device) for k, v in self.adapter.spans_fn(list(texts), offsets).items()}
        tensors = {"input_ids": encoded["input_ids"].to(self.device), "attention_mask": encoded["attention_mask"].to(self.device)}
        return tensors, offsets, spans

    def hidden(self, tensors: dict[str, Tensor], spans: dict[str, Tensor] | None) -> Tensor:
        """Final hidden states (after the final norm), float32, no autocast."""
        model = self.adapter.model
        if self.channel_model:
            embeddings = model.embed(tensors["input_ids"], spans)
            out = model.base(inputs_embeds=embeddings, attention_mask=tensors["attention_mask"])
        else:
            out = model.base_model(input_ids=tensors["input_ids"], attention_mask=tensors["attention_mask"])
        return (out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]).float()

    def head(self) -> Tensor:
        return self.host.get_output_embeddings().weight


@contextlib.contextmanager
def _hooks(*registrations: Any) -> Iterator[None]:
    try:
        yield
    finally:
        for handle in registrations:
            handle.remove()


def _replace_output(output: Any, value: Tensor) -> Any:
    if isinstance(output, tuple):
        return (value, *output[1:])
    return value


def _output_tensor(output: Any) -> Tensor:
    return output[0] if isinstance(output, tuple) else output


# -- prompt bookkeeping ---------------------------------------------------------------------------------

@dataclasses.dataclass
class _Prompt:
    text: str
    subject_span: tuple[int, int]
    prompt_length: int                 # characters before the target


def _build(prefix: str, template: str, subject: str, target: str = "") -> _Prompt:
    head, tail = prefix.split("{}", 1)
    t_head, t_tail = template.split("{x}", 1)
    prompt = head + t_head + subject + t_tail + tail
    start = len(head) + len(t_head)
    return _Prompt(prompt + target, (start, start + len(subject)), len(prompt))


def _positions(prompts: Sequence[_Prompt], offsets: Sequence[Sequence[tuple[int, int]]]) -> tuple[list[int], list[list[int]], list[int]]:
    """Per prompt: the subject's last token, the target tokens, and the last prompt token."""
    subject_last, targets, last = [], [], []
    for prompt, offs in zip(prompts, offsets):
        subject_last.append(read_index(list(offs), *prompt.subject_span))
        targets.append([t for t in range(1, len(offs)) if offs[t][1] > prompt.prompt_length])
        last.append(max(t for t in range(len(offs)) if offs[t][0] < prompt.prompt_length))
    return subject_last, targets, last


# -- statistics ------------------------------------------------------------------------------------------

@torch.no_grad()
def second_moments(host: EditableHost, texts: Sequence[str], layers: Sequence[int], *, batch_size: int = 8) -> dict[int, Tensor]:
    """Uncentred second moment `E[k kᵀ]` (float64, on the CPU) of the output projection's input at
    `layers`, over every non-padding token of `texts`."""
    sums: dict[int, Tensor] = {}
    count = 0
    mask_holder: dict[str, Tensor] = {}

    def capture(layer: int) -> Callable:
        def hook(module, inputs):
            keys = inputs[0].detach().float()
            keys = keys[mask_holder["mask"]].double()
            sums[layer] = sums.get(layer, 0) + keys.T @ keys
        return hook

    handles = [host.site(l).register_forward_pre_hook(capture(l)) for l in layers]
    with _hooks(*handles):
        for start in range(0, len(texts), batch_size):
            tensors, _, spans = host.encode(texts[start:start + batch_size])
            mask_holder["mask"] = tensors["attention_mask"].bool()
            count += int(mask_holder["mask"].sum())
            host.hidden(tensors, spans)
    if not count:
        raise ValueError("no tokens for the second moment")
    return {l: (sums[l] / count).cpu() for l in layers}


RIDGE = 1e-5          # relative to mean(diag C)


def ridged(moment: Tensor) -> Tensor:
    """`C + RIDGE · mean(diag C) · I` (float64): the second moment as ROME and MEMIT invert it. A moment estimated
    from a finite sample is rank-deficient when the sample has fewer distinct keys than key dimensions, and is
    ill-conditioned along rarely used directions; the ridge makes every inversion well-posed."""
    c = moment.double()
    return c + RIDGE * float(torch.diagonal(c).mean()) * torch.eye(c.shape[0], dtype=c.dtype)


def _inverse_solver(moment: Tensor) -> Callable[[Tensor], Tensor]:
    factor = torch.linalg.cholesky(ridged(moment))
    return lambda x: torch.cholesky_solve(x.double().reshape(factor.shape[0], -1), factor).reshape(x.shape)


# -- keys and targets -------------------------------------------------------------------------------------

@torch.no_grad()
def _site_io(host: EditableHost, layer: int, prompts: Sequence[_Prompt], *, residual: int | None = None) -> tuple[Tensor, Tensor]:
    """(keys, outputs) at the subject's last token: the output projection's input and output at
    `layer`, or (with `residual`) the decoder layer's output at that layer instead of the projection's."""
    keys, outputs = [], []
    store: dict[str, Tensor] = {}
    site = host.site(layer)
    handles = [site.register_forward_pre_hook(lambda m, i: store.__setitem__("k", i[0].detach().float())),
               (host.layers[residual] if residual is not None else site).register_forward_hook(
                   lambda m, i, o: store.__setitem__("v", _output_tensor(o).detach().float()))]
    with _hooks(*handles):
        for start in range(0, len(prompts), 16):
            batch = prompts[start:start + 16]
            tensors, offsets, spans = host.encode([p.text for p in batch])
            index, _, _ = _positions(batch, offsets)
            host.hidden(tensors, spans)
            rows = torch.arange(len(batch), device=store["k"].device)
            cols = torch.tensor(index, device=store["k"].device)
            keys.append(store["k"][rows, cols].cpu()); outputs.append(store["v"][rows, cols].cpu())
    return torch.cat(keys), torch.cat(outputs)


def _mean_keys(host: EditableHost, layer: int, requests: Sequence[EditRequest], settings: EditorSettings) -> Tensor:
    """Per request: the key at the subject's last token averaged over the context prefixes (n, d_in)."""
    prompts = [_build(prefix, r.prompt, r.subject) for r in requests for prefix in settings.context_prefixes]
    keys, _ = _site_io(host, layer, prompts)
    return keys.view(len(requests), len(settings.context_prefixes), -1).mean(1)


def optimize_delta(host: EditableHost, request: EditRequest, settings: EditorSettings, *, layer: int,
                   at_residual: bool) -> tuple[Tensor, Tensor, dict[str, float]]:
    """(target value, value before, record): `δ*` added at the subject's last token to the output of
    layer `layer`'s output projection (ROME) or of the decoder layer itself (MEMIT/AlphaEdit's z)."""
    module = host.layers[layer] if at_residual else host.site(layer)
    rewrite = [_build(prefix, request.prompt, request.subject, request.target) for prefix in settings.context_prefixes]
    kl = [_build("{}", settings.kl_prompt, request.subject)]
    prompts = rewrite + kl
    tensors, offsets, spans = host.encode([p.text for p in prompts])
    subject_last, targets, last = _positions(prompts, offsets)
    if any(not t for t in targets[:len(rewrite)]):
        raise ValueError(f"no target tokens for edit {request.key!r}")
    delta: Tensor | None = None
    state: dict[str, Any] = {}

    def hook(_module, _inputs, output):
        value = _output_tensor(output)
        if "init" not in state:
            state["init"] = value[0, subject_last[0]].detach().float().clone()
        if delta is None:
            return output
        rows = torch.arange(value.shape[0], device=value.device)
        cols = torch.tensor(subject_last, device=value.device)
        added = value.clone()
        added[rows, cols] = added[rows, cols] + delta.to(value.dtype)
        return _replace_output(output, added)

    head = host.head()
    target_ids = tensors["input_ids"]
    for parameter in host.adapter.model.parameters():
        parameter.requires_grad_(False)
    record: dict[str, float] = {}
    with _hooks(module.register_forward_hook(hook)):
        with torch.no_grad():
            hidden = host.hidden(tensors, spans)
            kl_init = F.log_softmax(F.linear(hidden[len(rewrite):, :][torch.arange(len(kl)), torch.tensor(last[len(rewrite):])],
                                             head.float()), -1)
        init = state["init"]
        delta = torch.zeros_like(init, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=settings.v_lr)
        max_norm = settings.clamp_norm_factor * float(init.norm())
        for step in range(settings.v_steps):
            optimizer.zero_grad()
            with torch.enable_grad():
                hidden = host.hidden(tensors, spans)
                nll = []
                for row, positions in enumerate(targets[:len(rewrite)]):
                    pos = torch.tensor(positions, device=hidden.device)
                    logits = F.linear(hidden[row, pos - 1], head.float())
                    nll.append(-F.log_softmax(logits, -1).gather(-1, target_ids[row, pos][:, None]).mean())
                nll_loss = torch.stack(nll).mean()
                kl_now = F.log_softmax(F.linear(hidden[len(rewrite):, :][torch.arange(len(kl)), torch.tensor(last[len(rewrite):])],
                                                head.float()), -1)
                kl_loss = settings.kl_factor * F.kl_div(kl_init, kl_now, log_target=True, reduction="batchmean")
                decay = settings.v_weight_decay * (delta.norm() / init.norm() ** 2)
                loss = nll_loss + kl_loss + decay
            record = {"steps": step + 1, "nll": float(nll_loss.detach()), "kl": float(kl_loss.detach()), "loss": float(loss.detach())}
            if record["nll"] < settings.early_stop:
                break
            if step == settings.v_steps - 1:
                break
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                if float(delta.norm()) > max_norm:
                    delta.mul_(max_norm / float(delta.norm()))
    record.update(delta_norm=float(delta.detach().norm()), init_norm=float(init.norm()))
    return (init + delta).detach().cpu(), init.cpu(), record


# -- editors ------------------------------------------------------------------------------------------------

def apply_rome(host: EditableHost, request: EditRequest, settings: EditorSettings, moments: dict[int, Tensor]) -> dict[str, Any]:
    """One ROME edit (in place); returns its record."""
    (layer,) = settings.layers if settings.layers is not None else default_layers("rome", host.depth)
    solve = _inverse_solver(moments[layer])
    mean_key = _mean_keys(host, layer, [request], settings)[0]
    left = solve(mean_key).float()
    left = left / left.norm()
    target, _, record = optimize_delta(host, request, settings, layer=layer, at_residual=False)
    key, current = _site_io(host, layer, [_build("{}", request.prompt, request.subject)])
    right = (target - current[0]) / torch.dot(key[0], left)
    host.add_update(layer, torch.outer(right, left))
    return {"method": "rome", "layer": layer, **record, "update_norm": float(right.norm() * left.norm())}


def apply_memit(host: EditableHost, requests: Sequence[EditRequest], settings: EditorSettings, moments: dict[int, Tensor]) -> dict[str, Any]:
    """MEMIT or AlphaEdit (`settings.method`) on a batch of edits (in place); returns the record."""
    layers = list(settings.layers if settings.layers is not None else default_layers(settings.method, host.depth))
    z_layer = layers[-1]
    targets, deltas = [], []
    for request in requests:
        z, _, record = optimize_delta(host, request, settings, layer=z_layer, at_residual=True)
        targets.append(z); deltas.append(record)
    targets_z = torch.stack(targets)                                  # (n, d)
    plain = [_build("{}", r.prompt, r.subject) for r in requests]
    info: dict[str, Any] = {"method": settings.method, "layers": layers, "edits": len(requests),
                            "delta_nll_mean": float(sum(d["nll"] for d in deltas) / max(1, len(deltas))), "per_layer": []}
    for i, layer in enumerate(layers):
        keys = _mean_keys(host, layer, requests, settings).double()   # (n, d_in)
        _, current = _site_io(host, layer, plain, residual=z_layer)
        residual = ((targets_z - current).double() / (len(layers) - i)).T     # (d, n)
        k = keys.T                                                     # (d_in, n)
        if settings.method == "memit":
            # The ridged C, as ROME's: with a rank-deficient C, λC + KKᵀ is singular and LU returns an arbitrary
            # null-space component, which the update then writes onto every key outside span(C) + span(K) (the
            # edited subject's own plain-prompt key among them), so the residual it leaves can grow instead of shrink.
            adjusted = torch.linalg.solve(settings.mom2_update_weight * ridged(moments[layer]) + k @ k.T, k)   # (d_in, n)
            update = residual @ adjusted.T                             # (d, d_in)
            null_dim = None
        else:
            eigenvalues, eigenvectors = torch.linalg.eigh(moments[layer].double())
            threshold = settings.nullspace_threshold if settings.nullspace_relative is None \
                else settings.nullspace_relative * float(eigenvalues.max())
            small = eigenvalues < threshold
            null_dim = int(small.sum())
            projection = eigenvectors[:, small] @ eigenvectors[:, small].T
            eye = torch.eye(k.shape[0], dtype=k.dtype)
            update = torch.linalg.solve(projection @ (k @ k.T) + settings.l2 * eye, projection @ k @ residual.T).T
        host.add_update(layer, update.float())
        info["per_layer"].append({"layer": layer, "z_error": float((targets_z - current).norm(dim=1).mean()),
                                  "residual_norm": float(residual.norm(dim=0).mean()),
                                  "update_norm": float(update.norm()), **({"null_dim": null_dim} if null_dim is not None else {})})
    return info


def apply_edits(host: EditableHost, requests: Sequence[EditRequest], settings: EditorSettings,
                moments: dict[int, Tensor]) -> dict[str, Any]:
    """ROME edits one request at a time (several requests: applied in sequence); MEMIT/AlphaEdit as a batch."""
    if settings.method == "rome":
        return {"method": "rome", "edits": [apply_rome(host, r, settings, moments) for r in requests]}
    return apply_memit(host, requests, settings, moments)


def edit_layers(host: EditableHost, settings: EditorSettings) -> tuple[int, ...]:
    return tuple(settings.layers) if settings.layers is not None else default_layers(settings.method, host.depth)


def sequence_logprob(host: EditableHost, prompt: str, continuation: str) -> float:
    """Summed `log p(continuation | prompt)` (float32; for checks and the toy edit)."""
    p = _Prompt(prompt + continuation, (0, 0), len(prompt))
    tensors, offsets, spans = host.encode([p.text])
    _, targets, _ = _positions([p], offsets)
    with torch.no_grad():
        hidden = host.hidden(tensors, spans)
        pos = torch.tensor(targets[0], device=hidden.device)
        logits = F.linear(hidden[0, pos - 1], host.head().float())
        return float(F.log_softmax(logits, -1).gather(-1, tensors["input_ids"][0, pos][:, None]).sum())


__all__ = ["CONTEXT_PREFIXES", "EditRequest", "EditableHost", "EditorSettings", "METHODS", "RIDGE", "apply_edits", "apply_memit",
           "apply_rome", "default_layers", "edit_layers", "optimize_delta", "ridged", "second_moments", "sequence_logprob"]
