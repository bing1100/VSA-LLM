import numpy as np
import pytest

from vsa_embed.statistics import holm_adjust, paired_ratio_bootstrap


def test_holm_matches_the_textbook_step_down() -> None:
    # p sorted: 0.01·4, 0.02·3, 0.03·2, 0.04·1 → monotone 0.04, 0.06, 0.06, 0.06.
    assert holm_adjust([0.03, 0.01, 0.04, 0.02]) == pytest.approx([0.06, 0.04, 0.06, 0.06])
    assert holm_adjust([0.5, 0.9]) == [1.0, 1.0]
    assert holm_adjust([]) == []


def test_ratio_bootstrap_is_token_weighted_and_detects_a_shift() -> None:
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 40, 300).astype(float)
    baseline = 3.0 * counts + rng.normal(0, 1, 300) * np.sqrt(counts)
    shifted = paired_ratio_bootstrap(-0.05 * counts + rng.normal(0, 0.1, 300) * np.sqrt(counts), counts, baseline, resamples=1000)
    assert shifted["mean"] == pytest.approx(-0.05, abs=0.01)
    assert shifted["ci_high"] < 0 and shifted["p_value"] < 0.01
    assert shifted["relative"] == pytest.approx(-0.05 / 3.0, abs=0.004)
    assert shifted["relative_ci_low"] < shifted["relative"] < shifted["relative_ci_high"] < 0
    noise = rng.normal(0, 0.1, 300) * np.sqrt(counts)      # exactly zero-mean null: each cluster has a mirror
    null = paired_ratio_bootstrap(np.concatenate([noise, -noise]), np.concatenate([counts, counts]), resamples=1000)
    assert null["mean"] == pytest.approx(0.0, abs=1e-12)
    assert null["ci_low"] < 0 < null["ci_high"] and null["p_value"] > 0.05
    # Weighted, not a mean of per-cluster ratios: one heavy cluster dominates.
    assert paired_ratio_bootstrap([10.0, 1.0], [100.0, 1.0], resamples=50)["mean"] == pytest.approx(11 / 101)


def test_ratio_bootstrap_reuses_resamples_and_rejects_bad_input() -> None:
    a = paired_ratio_bootstrap([1.0, -2.0, 0.5, 0.2], [1, 2, 3, 4], resamples=200, seed=3)
    b = paired_ratio_bootstrap([1.0, -2.0, 0.5, 0.2], [1, 2, 3, 4], resamples=200, seed=3)
    assert a == b
    with pytest.raises(ValueError):
        paired_ratio_bootstrap([1.0], [0.0])
    with pytest.raises(ValueError):
        paired_ratio_bootstrap([1.0, 2.0], [1.0])
