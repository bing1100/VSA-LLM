"""Identity-preserving relation operators for experiment 01c."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .algebra import HRRAlgebra


class ResidualHRRRelation(nn.Module):
    """Preserve the source and learn a gated HRR correction, optionally with an offset/basis."""

    def __init__(
        self,
        relation_count: int,
        dimension: int,
        *,
        use_offset: bool = False,
        input_gated: bool = False,
        diagonal_basis: bool = False,
        max_residual_scale: float = 0.25,
        max_log_basis_scale: float = 0.25,
        binding: str = "hrr",
    ) -> None:
        super().__init__()
        self.relation_count = relation_count
        self.dimension = dimension
        self.use_offset = use_offset
        self.input_gated = input_gated
        self.diagonal_basis = diagonal_basis
        self.max_residual_scale = float(max_residual_scale)
        self.max_log_basis_scale = float(max_log_basis_scale)
        self.binding = binding
        if self.max_residual_scale <= 0:
            raise ValueError("max_residual_scale must be positive")
        if self.max_log_basis_scale <= 0:
            raise ValueError("max_log_basis_scale must be positive")
        if binding not in {"hrr", "diagonal"}:
            raise ValueError("binding must be 'hrr' or 'diagonal'")
        self.roles = nn.Parameter(torch.randn(relation_count, dimension) / dimension**0.5)
        self.residual_scale_logits = nn.Parameter(torch.zeros(relation_count))
        self.offset = nn.Parameter(torch.zeros(relation_count, dimension)) if use_offset else None
        self.gate_weight = nn.Parameter(torch.zeros(relation_count, dimension)) if input_gated else None
        self.gate_bias = nn.Parameter(torch.zeros(relation_count)) if input_gated else None
        self.log_basis_scale = nn.Parameter(torch.zeros(dimension)) if diagonal_basis else None
        self.algebra = HRRAlgebra()

    def _validate(self, relation_ids: Tensor, sources: Tensor) -> None:
        if relation_ids.shape != sources.shape[:-1] or sources.shape[-1] != self.dimension:
            raise ValueError("relation_ids must match source leading dimensions")

    def components(self, sources: Tensor, relation_ids: Tensor) -> dict[str, Tensor]:
        """Return source, offset, raw HRR, coefficient, and correction for auditing."""
        self._validate(relation_ids, sources)
        if self.log_basis_scale is None:
            encoded = sources
            decode_scale = None
        else:
            bounded_log_scale = self.max_log_basis_scale * torch.tanh(self.log_basis_scale)
            basis_scale = bounded_log_scale.exp()
            encoded = sources * basis_scale
            decode_scale = basis_scale.reciprocal()
        roles = F.normalize(self.roles[relation_ids], dim=-1)
        bound = self.algebra.bind(roles, encoded) if self.binding == "hrr" else roles * encoded
        raw_hrr = F.normalize(bound, dim=-1)
        coefficient = self.max_residual_scale * torch.tanh(self.residual_scale_logits[relation_ids])
        if self.gate_weight is not None and self.gate_bias is not None:
            content_gate = torch.sigmoid(
                (self.gate_weight[relation_ids] * encoded).sum(-1) + self.gate_bias[relation_ids]
            )
            coefficient = coefficient * content_gate
        offset = torch.zeros_like(encoded) if self.offset is None else self.offset[relation_ids]
        if decode_scale is not None:
            offset = offset * decode_scale
            raw_hrr = F.normalize(raw_hrr * decode_scale, dim=-1)
        correction = coefficient[..., None] * raw_hrr
        transformed = sources + offset + correction
        return {
            "source": sources,
            "offset": offset,
            "raw_hrr": raw_hrr,
            "coefficient": coefficient,
            "correction": correction,
            "output": transformed,
        }

    def forward(self, sources: Tensor, relation_ids: Tensor) -> Tensor:
        return self.components(sources, relation_ids)["output"]

    def diagnostics(self, sources: Tensor, relation_ids: Tensor) -> dict[str, float]:
        with torch.no_grad():
            parts = self.components(sources, relation_ids)
            return {
                "mean_abs_hrr_coefficient": float(parts["coefficient"].abs().mean()),
                "mean_hrr_correction_norm": float(parts["correction"].norm(dim=-1).mean()),
                "mean_offset_norm": float(parts["offset"].norm(dim=-1).mean()),
                "mean_log_basis_scale_abs": (
                    float(self.log_basis_scale.abs().mean()) if self.log_basis_scale is not None else 0.0
                ),
            }

    def regularization(
        self, sources: Tensor, relation_ids: Tensor, *, correction_weight: float = 0.0,
        offset_weight: float = 0.0, basis_weight: float = 0.0,
    ) -> Tensor:
        parts = self.components(sources, relation_ids)
        loss = correction_weight * parts["correction"].square().sum(-1).mean()
        if self.offset is not None:
            loss = loss + offset_weight * self.offset.square().sum(-1).mean()
        if self.log_basis_scale is not None:
            loss = loss + basis_weight * self.log_basis_scale.square().mean()
        return loss


def create_residual_relation(
    family: str, relation_count: int, dimension: int, *, max_residual_scale: float = 0.25,
    max_log_basis_scale: float = 0.25,
) -> ResidualHRRRelation:
    options = {
        "residual_hrr": {},
        "offset_residual_hrr": {"use_offset": True},
        "gated_offset_residual_hrr": {"use_offset": True, "input_gated": True},
        "basis_offset_residual_hrr": {"use_offset": True, "diagonal_basis": True},
        "basis_offset_diagonal_control": {
            "use_offset": True, "diagonal_basis": True, "binding": "diagonal",
        },
    }
    try:
        return ResidualHRRRelation(
            relation_count, dimension, max_residual_scale=max_residual_scale,
            max_log_basis_scale=max_log_basis_scale, **options[family]
        )
    except KeyError as error:
        raise ValueError(f"unsupported residual relation family {family!r}") from error