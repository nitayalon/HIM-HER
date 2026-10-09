from __future__ import annotations

import numpy as np
import pytest

from himher.domains.handoff.construct import construct_bernoulli_model
from himher.domains.handoff.env import Hand


def test_matches_hand_computed_posterior_mean():
    # context 0: 3 R, 1 L; context 1: 1 R, 3 L; alpha = 1
    window = [(0, Hand.R)] * 3 + [(0, Hand.L)] * 1 + [(1, Hand.R)] * 1 + [(1, Hand.L)] * 3
    model = construct_bernoulli_model(window, alpha=1.0)

    assert model.theta0 == pytest.approx((3 + 1) / (4 + 2))
    assert model.theta1 == pytest.approx((1 + 1) / (4 + 2))


def test_empty_window_falls_back_to_prior_mean():
    model = construct_bernoulli_model([], alpha=1.0)
    assert model.theta0 == pytest.approx(0.5)
    assert model.theta1 == pytest.approx(0.5)


def test_missing_context_falls_back_to_prior_mean_for_that_context_only():
    window = [(0, Hand.R)] * 10  # context 1 never observed
    model = construct_bernoulli_model(window, alpha=1.0)
    assert model.theta1 == pytest.approx(0.5)
    assert model.theta0 == pytest.approx((10 + 1) / (10 + 2))


def test_converges_to_the_true_theta_with_enough_data():
    rng = np.random.default_rng(42)
    true_theta0, true_theta1 = 0.2, 0.8
    n = 5000
    window = [
        (0, Hand.R if rng.random() < true_theta0 else Hand.L) for _ in range(n)
    ] + [(1, Hand.R if rng.random() < true_theta1 else Hand.L) for _ in range(n)]

    model = construct_bernoulli_model(window, alpha=1.0)
    assert model.theta0 == pytest.approx(true_theta0, abs=0.02)
    assert model.theta1 == pytest.approx(true_theta1, abs=0.02)
