"""Tests for the domain-agnostic helpers in himher.core.model.

Exercised against BernoulliHandModel, since the protocol needs a concrete
implementation to test against -- the point of these tests is that
history_log_likelihood and model_divergence never touch theta directly,
only predict_proba/log_prob, so they would work unchanged against any
other CounterpartModel implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from himher.core.model import (
    CounterpartModel,
    history_log_likelihood,
    model_divergence,
    total_variation_distance,
)
from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Hand


def test_counterpart_model_is_structurally_recognized():
    assert isinstance(BernoulliHandModel(0.1, 0.9), CounterpartModel)


def test_counterpart_model_rejects_objects_missing_methods():
    class NotAModel:
        pass

    assert not isinstance(NotAModel(), CounterpartModel)


def test_history_log_likelihood_matches_manual_sum():
    model = BernoulliHandModel(theta0=0.1, theta1=0.9)
    history = [(0, Hand.L), (0, Hand.R), (1, Hand.R), (1, Hand.L)]

    expected = sum(model.log_prob(x, a) for x, a in history)
    assert history_log_likelihood(model, history) == pytest.approx(expected)


def test_history_log_likelihood_prefers_the_matching_model():
    left_leaning = BernoulliHandModel(theta0=0.1, theta1=0.1)
    right_leaning = BernoulliHandModel(theta0=0.9, theta1=0.9)
    history_from_right_leaning = [(0, Hand.R)] * 20 + [(1, Hand.R)] * 20

    ll_right = history_log_likelihood(right_leaning, history_from_right_leaning)
    ll_left = history_log_likelihood(left_leaning, history_from_right_leaning)
    assert ll_right > ll_left


def test_total_variation_distance_identical_distributions_is_zero():
    p = np.array([0.3, 0.7])
    assert total_variation_distance(p, p) == pytest.approx(0.0)


def test_total_variation_distance_disjoint_point_masses_is_one():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert total_variation_distance(p, q) == pytest.approx(1.0)


def test_total_variation_distance_is_symmetric():
    p = np.array([0.2, 0.8])
    q = np.array([0.6, 0.4])
    assert total_variation_distance(p, q) == pytest.approx(total_variation_distance(q, p))


def test_model_divergence_is_zero_for_identical_models():
    m = BernoulliHandModel(0.2, 0.8)
    m_prime = BernoulliHandModel(0.2, 0.8)
    assert model_divergence(m, m_prime, contexts=[0, 1]) == pytest.approx(0.0)


def test_model_divergence_grows_with_parameter_distance():
    base = BernoulliHandModel(0.1, 0.1)
    near = BernoulliHandModel(0.2, 0.2)
    far = BernoulliHandModel(0.9, 0.9)

    d_near = model_divergence(base, near, contexts=[0, 1])
    d_far = model_divergence(base, far, contexts=[0, 1])
    assert 0.0 < d_near < d_far


def test_model_divergence_averages_over_contexts():
    # theta0 identical, theta1 differs -> divergence should be half of the
    # single-context (x=1) TV distance, since the other context contributes 0.
    m = BernoulliHandModel(0.5, 0.1)
    m_prime = BernoulliHandModel(0.5, 0.9)

    single_context_tv = total_variation_distance(m.predict_proba(1), m_prime.predict_proba(1))
    averaged = model_divergence(m, m_prime, contexts=[0, 1])
    assert averaged == pytest.approx(single_context_tv / 2)


def test_model_divergence_requires_at_least_one_context():
    m = BernoulliHandModel(0.1, 0.9)
    with pytest.raises(ValueError):
        model_divergence(m, m, contexts=[])
