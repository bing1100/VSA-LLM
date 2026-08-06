import torch
import pytest

from vsa_embed.confidence import expected_calibration_error, fit_logistic_calibrator, precision_threshold


def test_calibrator_orders_confidence_by_margin() -> None:
    margins = torch.tensor([0.01, 0.03, 0.05, 0.20, 0.30, 0.40])
    correct = torch.tensor([0, 0, 0, 1, 1, 1])
    model = fit_logistic_calibrator(margins, correct)
    confidence = model.predict(margins)
    assert torch.all(confidence[1:] >= confidence[:-1])
    assert expected_calibration_error(confidence, correct.bool(), bins=3) < 0.35


def test_constant_outcome_calibration_is_smoothed() -> None:
    model = fit_logistic_calibrator(torch.arange(5.0), torch.ones(5))
    confidence = model.predict(torch.tensor([-100.0, 100.0]))
    assert torch.all((confidence > 0) & (confidence < 1))
    torch.testing.assert_close(confidence[0], confidence[1])


def test_precision_threshold_respects_ties() -> None:
    confidence = torch.tensor([0.9, 0.8, 0.8, 0.1])
    correct = torch.tensor([True, True, False, False])
    threshold, coverage, accuracy = precision_threshold(confidence, correct, 0.75)
    assert threshold == pytest.approx(0.9)
    assert coverage == 0.25
    assert accuracy == 1.0
