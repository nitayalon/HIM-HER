from __future__ import annotations

import numpy as np
import pytest

from himher.core.detector import mismatch_detected, surprise_statistic
from himher.core.history import RecentHistory
from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Hand


def test_empty_window_has_zero_statistic():
    history = RecentHistory(window=10)
    model = BernoulliHandModel(0.1, 0.9)
    assert surprise_statistic(history, model) == 0.0


def test_statistic_is_zero_surprise_in_the_limit_of_a_perfectly_matching_model():
    history = RecentHistory(window=10)
    model = BernoulliHandModel(theta0=1.0, theta1=1.0)
    for _ in range(10):
        history.append(0, Hand.R)

    # -log(1.0) == 0: a perfectly predicted, deterministic sequence is
    # exactly zero surprise, not just small.
    assert surprise_statistic(history, model) == pytest.approx(0.0)


def test_statistic_matches_manual_mean_surprise():
    history = RecentHistory(window=10)
    model = BernoulliHandModel(theta0=0.1, theta1=0.9)
    observations = [(0, Hand.R), (0, Hand.L), (1, Hand.R), (1, Hand.L)]
    for context, action in observations:
        history.append(context, action)

    expected = -np.mean([model.log_prob(c, a) for c, a in observations])
    assert surprise_statistic(history, model) == pytest.approx(expected)


def test_statistic_reflects_whichever_model_is_passed_in():
    # D_t must be recomputed against the *current* incumbent, not cached
    # from whatever model was active when observations were appended.
    history = RecentHistory(window=10)
    for _ in range(10):
        history.append(0, Hand.R)

    matching_model = BernoulliHandModel(1.0, 1.0)
    mismatched_model = BernoulliHandModel(0.0, 0.0)

    assert surprise_statistic(history, matching_model) < surprise_statistic(
        history, mismatched_model
    )


@pytest.mark.parametrize(
    "statistic,kappa,expected", [(0.5, 1.0, False), (1.5, 1.0, True), (1.0, 1.0, False)]
)
def test_mismatch_detected_is_a_strict_threshold(statistic, kappa, expected):
    assert mismatch_detected(statistic, kappa) is expected
