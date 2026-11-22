import math

import numpy as np
import pytest

from himher.baselines.bpr_plus_mdp import (
    BPRPlusMDPAgent, ContextMDP, gauss_pdf, performance_model, policy_scores, value_iteration)
from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Hand
from himher.experiment import conditions as cond
from himher.experiment import instrumentation as inst
from himher.experiment.rebuttal_methods import METHODS
from himher.experiment.runner import run_episode

LIB = cond.LIBRARY_0


def agent(**kw):
    d = dict(rho=0.0, n=7, epsilon=0.0, learn_rounds=20, belief_floor=0.01)
    d.update(kw)
    return BPRPlusMDPAgent(LIB, np.random.default_rng(0), **d)


@pytest.mark.parametrize("theta", [(0.1, 0.1), (0.9, 0.9), (0.1, 0.9), (0.2, 0.8), (0.7, 0.3), (0.5, 0.6)])
def test_value_iteration_recovers_best_response(theta):
    pi, sweeps = value_iteration(ContextMDP.from_model(BernoulliHandModel(*theta)))
    assert list(pi) == [int(t > 0.5) for t in theta] and sweeps > 0


def test_performance_model_moments():
    M = ContextMDP.from_model(cond.M_L)
    assert performance_model(M, np.array([0, 0])) == pytest.approx((0.9, 0.3))
    assert performance_model(M, np.array([0, 1])) == pytest.approx((0.5, 0.5))
    assert performance_model(ContextMDP.from_model(BernoulliHandModel(1.0, 1.0)), np.array([1, 1]))[1] == 0.05


def test_mdp_learned_from_counts():
    steps = [(0, 1, 1.0), (1, 0, 0.0), (0, 1, 0.0), (0, 0, 1.0)]  # (s, a, r)
    M = ContextMDP.from_counts(steps)
    assert M.R[0, 1] == 0.5 and M.R[1, 0] == 0.0 and M.R[0, 0] == 1.0
    assert M.R[1, 1] == 0.5  # unvisited -> 0.5
    assert np.allclose(M.T[0, 1], [0.5, 0.5])  # 0->1 then 0->0 under a=1
    assert np.allclose(M.T[1, 0], [1.0, 0.0])
    assert np.allclose(M.T[1, 1], [0.5, 0.5])  # unvisited -> uniform
    # implied theta_0: R played twice (1 match), L once (1 match -> counterpart L)
    assert M.implied_theta[0] == pytest.approx(1 / 3)


def test_belief_update_is_gaussian_bayes_with_floor():
    a = agent()
    act = a.act(0)
    j = a._current_policy
    a.update(0, act, Hand.L, 1.0)
    p = np.array([gauss_pdf(1.0, a._mu[i, j], a._sigma[i, j]) for i in range(3)])
    post = np.maximum(p / p.sum() * 1.0, 0.01)
    assert np.allclose(a.posterior, post / post.sum())


def test_probability_of_improvement_prefers_policy_of_believed_model():
    a = agent()
    for k in range(3):
        a._belief = np.full(3, 0.01)
        a._belief[k] = 0.98
        a._belief /= a._belief.sum()
        assert a._pi_choice() == k
    s = policy_scores(np.array([1.0]), np.array([[0.5]]), np.array([0.5]))
    assert s[0] == pytest.approx(0.5 * math.erfc((0.01 / 0.5) / math.sqrt(2)))


def test_detection_learning_and_library_growth():
    a = agent(rho=10.0, learn_rounds=5)
    for t in range(7):
        x = t % 2
        act = a.act(x)
        a.update(x, act, Hand.R, float(act == Hand.R))
    assert a.detection_times == [6] and a.learning
    for t in range(5):
        x = t % 2
        act = a.act(x)
        a.update(x, act, Hand.R, float(act == Hand.R))
    assert not a.learning and len(a.hypotheses) == 4 and len(a.policies) == 4
    assert a.posterior[-1] == pytest.approx(1 - 0.03)
    assert a._mu.shape == (4, 4)


def test_epsilon_explores_over_policies():
    a = agent(epsilon=1.0)
    chosen = []
    for _ in range(600):
        a.act(0)
        chosen.append(a._current_policy)
    counts = np.bincount(chosen, minlength=3) / 600
    assert np.all(np.abs(counts - 1 / 3) < 0.07)


def test_registered_bpr_is_mdp_version_and_deterministic():
    spec = inst.condition_by_label("outside_library_switch_m2")
    a1, a2 = METHODS["BPR+"].factory(2, spec), METHODS["BPR+"].factory(2, spec)
    assert type(a1).__name__ == "BPRPlusMDPAgent"
    r1 = [l.reward for l in run_episode(a1, spec.condition, 2)]
    r2 = [l.reward for l in run_episode(a2, spec.condition, 2)]
    assert r1 == r2
    res = inst.run_instrumented(METHODS["BPR+"].factory(2, spec), spec, 2, "BPR+", log_level="full")
    assert [float(x) for x in res.per_step["ego_reward"]] == r1
    assert res.summary["n_detections"] == len(a1.detection_times)
    assert type(METHODS["BPR+bern"].factory(2, spec)).__name__ == "BPRPlusAgent"
