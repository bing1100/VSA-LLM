import math

import pytest

from vsa_embed.convergence import (
    compare, data_multiplier, escalation_verdict, fit_power_law, project, tokens_to_loss,
)

TOKENS = [1e7 * 2**i for i in range(8)]


def curve(e: float, b: float, beta: float, multiplier: float = 1.0, noise: float = 0.0, seed: int = 0):
    import random
    rng = random.Random(seed)
    return [(t, e + b * (multiplier * t / 1e7) ** (-beta) + rng.gauss(0, noise)) for t in TOKENS]


def test_tokens_to_loss_interpolates_in_log_tokens() -> None:
    c = [(1e7, 4.0), (4e7, 3.0)]
    assert tokens_to_loss(c, 3.5) == pytest.approx(2e7)
    assert tokens_to_loss(c, 2.0) is None


def test_data_multiplier_is_recovered_from_shifted_curves() -> None:
    base, fast = curve(2.0, 2.0, 0.4), curve(2.0, 2.0, 0.4, multiplier=1.5)
    assert data_multiplier(base, fast, 2.9) == pytest.approx(1.5, rel=0.03)


def test_power_law_fit_recovers_parameters_and_projects() -> None:
    fit = fit_power_law(curve(2.0, 2.0, 0.4))
    assert fit["E"] == pytest.approx(2.0, abs=0.05) and fit["beta"] == pytest.approx(0.4, abs=0.05)
    assert project(fit, 1e10) == pytest.approx(2.0 + 2.0 * 1000 ** -0.4, abs=0.02)


def test_compare_and_escalation_on_a_real_and_a_null_effect() -> None:
    bases = [curve(2.0, 2.0, 0.4, noise=0.003, seed=s) for s in range(3)]
    better = [curve(1.9, 2.0, 0.4, multiplier=1.6, noise=0.003, seed=10 + s) for s in range(3)]
    same = [curve(2.0, 2.0, 0.4, noise=0.003, seed=20 + s) for s in range(3)]
    real = compare(bases, better, projection_tokens=2.5e9)
    null = compare(bases, same, projection_tokens=2.5e9)
    controls = {"C1": compare(same, better, projection_tokens=2.5e9)}
    assert escalation_verdict(real, controls)["escalate"]
    verdict = escalation_verdict(null, {"C1": compare(bases, same, projection_tokens=2.5e9)})
    assert not verdict["escalate"] and not verdict["rule_i"]
