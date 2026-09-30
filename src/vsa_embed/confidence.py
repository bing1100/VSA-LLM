"""Held-out confidence calibration and selective-prediction utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class LogisticCalibrator:
    margin_mean: float
    margin_std: float
    weight: float
    bias: float

    def predict(self, margins: Tensor) -> Tensor:
        standardized = (margins - self.margin_mean) / self.margin_std
        return torch.sigmoid(self.weight * standardized + self.bias)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def fit_logistic_calibrator(margins: Tensor, correct: Tensor, *, steps: int = 400) -> LogisticCalibrator:
    """Fit scalar Platt scaling on calibration-only observations."""
    margins = margins.detach().to(torch.float64)
    labels = correct.detach().to(torch.float64)
    if margins.numel() < 2:
        raise ValueError("calibration requires at least two samples")
    mean = margins.mean()
    std = margins.std().clamp_min(1e-8)
    if labels.unique().numel() == 1:
        # Smoothed constant model keeps perfect/failing low-load conditions
        # representable without emitting exactly 0 or 1 confidence.
        prevalence = (labels.sum() + 1.0) / (labels.numel() + 2.0)
        return LogisticCalibrator(mean.item(), std.item(), 0.0, torch.logit(prevalence).item())
    x = (margins - mean) / std
    weight = torch.ones((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=steps, line_search_fn="strong_wolfe")

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(weight * x + bias, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return LogisticCalibrator(mean.item(), std.item(), weight.item(), bias.item())


def expected_calibration_error(confidence: Tensor, correct: Tensor, bins: int = 10) -> float:
    """Equal-frequency ECE, robust to concentrated confidence values."""
    if confidence.numel() != correct.numel() or confidence.numel() == 0:
        raise ValueError("confidence and correct must be non-empty and equal length")
    order = confidence.argsort()
    chunks = torch.tensor_split(order, min(bins, order.numel()))
    total = confidence.numel()
    return sum(
        chunk.numel() / total
        * abs(confidence[chunk].mean().item() - correct[chunk].float().mean().item())
        for chunk in chunks if chunk.numel()
    )


def precision_threshold(
    confidence: Tensor, correct: Tensor, target_accuracy: float, *, method: str = "point",
) -> tuple[float, float, float]:
    """Choose maximum-coverage calibration threshold meeting target accuracy.

    `method="point"` requires the calibration point estimate to reach the target, which puts
    calibration accuracy right at the target and lets held-out accuracy fall below it about
    half the time. `method="wilson_lower"` requires the Wilson 95% lower bound of the accepted
    prefix to reach the target instead, trading coverage for a held-out margin.
    """
    if method not in {"point", "wilson_lower"}:
        raise ValueError("method must be 'point' or 'wilson_lower'")
    order = confidence.argsort(descending=True)
    ordered_conf = confidence[order]
    counts = torch.arange(1, order.numel() + 1, device=confidence.device)
    cumulative_correct = correct[order].float().cumsum(0)
    cumulative_accuracy = cumulative_correct / counts
    if method == "wilson_lower":
        z = 1.96
        p, n = cumulative_accuracy, counts.to(cumulative_accuracy.dtype)
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * torch.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        criterion = centre - half
    else:
        criterion = cumulative_accuracy
    # Thresholding includes all confidence ties. Only evaluate prefixes ending
    # at a tie boundary so reported calibration coverage is realizable.
    tie_ends = torch.ones_like(ordered_conf, dtype=torch.bool)
    tie_ends[:-1] = ordered_conf[:-1] != ordered_conf[1:]
    valid = ((criterion >= target_accuracy) & tie_ends).nonzero(as_tuple=False).flatten()
    if valid.numel() == 0:
        return 1.0, 0.0, 0.0
    selected = int(valid[-1])
    threshold = ordered_conf[selected].item()
    return threshold, (selected + 1) / order.numel(), cumulative_accuracy[selected].item()
