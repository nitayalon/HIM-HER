from __future__ import annotations

import numpy as np
import pytest

from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Hand
from himher.domains.handoff.policy import TabularQPolicy, Transition


def test_q_value_defaults_to_zero_for_an_unseen_key():
    policy = TabularQPolicy()
    model = BernoulliHandModel(0.1, 0.9)
    assert policy.q_value(0, model, Hand.L) == 0.0


def test_update_moves_q_toward_the_observed_reward():
    policy = TabularQPolicy(learning_rate=0.5)
    model = BernoulliHandModel(0.1, 0.9)
    transition = Transition(context=0, ego_action=Hand.R, counterpart_action=Hand.R, reward=1.0)

    policy.update(transition, model)
    first = policy.q_value(0, model, Hand.R)
    policy.update(transition, model)
    second = policy.q_value(0, model, Hand.R)

    assert 0.0 < first < second < 1.0


def test_q_converges_to_the_empirical_match_rate():
    policy = TabularQPolicy(learning_rate=0.1)
    model = BernoulliHandModel(0.1, 0.9)  # theta1 = P(R | context=1) = 0.9
    rng = np.random.default_rng(7)

    for _ in range(5000):
        counterpart_action = model.sample(1, rng)
        transition = Transition(1, Hand.R, counterpart_action, 1.0 if counterpart_action == Hand.R else 0.0)
        policy.update(transition, model)

    assert policy.q_value(1, model, Hand.R) == pytest.approx(0.9, abs=0.05)


def test_greedy_action_matches_analytic_best_response_after_training():
    policy = TabularQPolicy(learning_rate=0.1)
    model = BernoulliHandModel(0.2, 0.8)
    rng = np.random.default_rng(3)

    for context in (0, 1):
        for _ in range(2000):
            for ego_action in (Hand.L, Hand.R):
                counterpart_action = model.sample(context, rng)
                reward = 1.0 if ego_action == counterpart_action else 0.0
                policy.update(Transition(context, ego_action, counterpart_action, reward), model)

    for context in (0, 1):
        assert policy.greedy_action(context, model) == TabularQPolicy.analytic_best_response(
            context, model
        )


def test_act_is_greedy_when_epsilon_is_zero():
    policy = TabularQPolicy(epsilon=0.0)
    model = BernoulliHandModel(0.1, 0.9)
    rng = np.random.default_rng(0)

    policy.update(Transition(0, Hand.R, Hand.R, 1.0), model)
    policy.update(Transition(0, Hand.L, Hand.R, 0.0), model)

    for _ in range(20):
        assert policy.act(0, model, rng) == Hand.R


def test_act_is_uniformly_random_when_epsilon_is_one():
    policy = TabularQPolicy(epsilon=1.0)
    model = BernoulliHandModel(0.1, 0.9)
    rng = np.random.default_rng(0)

    actions = [policy.act(0, model, rng) for _ in range(5000)]
    rate_right = np.mean([a == Hand.R for a in actions])
    assert rate_right == pytest.approx(0.5, abs=0.03)


def test_analytic_best_response_matches_theta_threshold():
    model = BernoulliHandModel(theta0=0.3, theta1=0.7)
    assert TabularQPolicy.analytic_best_response(0, model) == Hand.L
    assert TabularQPolicy.analytic_best_response(1, model) == Hand.R


def test_update_count_is_zero_before_any_update():
    policy = TabularQPolicy()
    model = BernoulliHandModel(0.1, 0.9)
    assert policy.update_count(model) == 0
    assert policy.total_updates == 0


def test_update_count_tracks_each_model_independently():
    policy = TabularQPolicy()
    model_a = BernoulliHandModel(0.1, 0.9)
    model_b = BernoulliHandModel(0.9, 0.1)

    policy.update(Transition(0, Hand.R, Hand.R, 1.0), model_a)
    policy.update(Transition(0, Hand.L, Hand.R, 0.0), model_a)
    policy.update(Transition(1, Hand.R, Hand.L, 0.0), model_b)

    assert policy.update_count(model_a) == 2
    assert policy.update_count(model_b) == 1
    assert policy.total_updates == 3
    assert policy.update_counts == {model_a: 2, model_b: 1}


def test_relabel_and_replay_also_increments_update_counts():
    from himher.core.her import ReplayBuffer

    policy = TabularQPolicy()
    model = BernoulliHandModel(0.1, 0.9)
    buffer: ReplayBuffer[Transition] = ReplayBuffer()
    for t in (Transition(0, Hand.R, Hand.R, 1.0), Transition(1, Hand.L, Hand.L, 1.0)):
        buffer.append(t)

    buffer.relabel_and_replay(model, policy.update)

    assert policy.update_count(model) == 2
