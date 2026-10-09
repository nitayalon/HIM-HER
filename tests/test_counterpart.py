from __future__ import annotations

import numpy as np
import pytest

from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Hand


@pytest.mark.parametrize("theta0,theta1", [(-0.1, 0.5), (0.5, 1.1), (1.5, -1.0)])
def test_rejects_theta_outside_unit_interval(theta0, theta1):
    with pytest.raises(ValueError):
        BernoulliHandModel(theta0, theta1)


def test_predict_proba_matches_theta_per_context():
    model = BernoulliHandModel(theta0=0.3, theta1=0.8)

    proba_x0 = model.predict_proba(0)
    proba_x1 = model.predict_proba(1)

    assert proba_x0[Hand.R] == pytest.approx(0.3)
    assert proba_x0[Hand.L] == pytest.approx(0.7)
    assert proba_x1[Hand.R] == pytest.approx(0.8)
    assert proba_x1[Hand.L] == pytest.approx(0.2)


def test_predict_proba_sums_to_one():
    model = BernoulliHandModel(theta0=0.3, theta1=0.8)
    assert model.predict_proba(0).sum() == pytest.approx(1.0)
    assert model.predict_proba(1).sum() == pytest.approx(1.0)


def test_log_prob_matches_log_theta():
    model = BernoulliHandModel(theta0=0.3, theta1=0.8)

    assert model.log_prob(0, Hand.R) == pytest.approx(np.log(0.3))
    assert model.log_prob(0, Hand.L) == pytest.approx(np.log(0.7))
    assert model.log_prob(1, Hand.R) == pytest.approx(np.log(0.8))
    assert model.log_prob(1, Hand.L) == pytest.approx(np.log(0.2))


@pytest.mark.parametrize("theta0,theta1", [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
def test_log_prob_is_finite_at_the_simplex_boundary(theta0, theta1):
    model = BernoulliHandModel(theta0, theta1)
    for context in (0, 1):
        for action in (Hand.L, Hand.R):
            assert np.isfinite(model.log_prob(context, action))


def test_sample_matches_theta_in_expectation():
    model = BernoulliHandModel(theta0=0.2, theta1=0.9)
    rng = np.random.default_rng(0)

    n = 20_000
    draws_x0 = [model.sample(0, rng) for _ in range(n)]
    draws_x1 = [model.sample(1, rng) for _ in range(n)]

    rate_x0 = np.mean([a == Hand.R for a in draws_x0])
    rate_x1 = np.mean([a == Hand.R for a in draws_x1])

    assert rate_x0 == pytest.approx(0.2, abs=0.02)
    assert rate_x1 == pytest.approx(0.9, abs=0.02)


def test_theta_property_round_trips_constructor_args():
    model = BernoulliHandModel(theta0=0.4, theta1=0.6)
    assert model.theta == (0.4, 0.6)


def test_model_is_hashable_and_usable_in_a_set():
    # Frozen dataclasses should be hashable so the model library (built
    # later) can deduplicate/store models by value.
    a = BernoulliHandModel(0.1, 0.9)
    b = BernoulliHandModel(0.1, 0.9)
    c = BernoulliHandModel(0.1, 0.1)
    assert a == b
    assert {a, b, c} == {a, c}
