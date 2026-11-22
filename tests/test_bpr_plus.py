import numpy as np
import pytest

from himher.baselines.bpr_plus import BPRPlusAgent, best_response, reward_prob
from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Hand
from himher.experiment import conditions as cond
from himher.experiment import instrumentation as inst
from himher.experiment.rebuttal_methods import METHODS, make_bpr_agent
from himher.experiment.runner import run_episode

LIB = cond.LIBRARY_0


def agent(**kw):
    defaults = dict(rho=0.0, n=7, epsilon=0.0, learn_rounds=20, belief_floor=0.01)
    defaults.update(kw)
    return BPRPlusAgent(LIB, np.random.default_rng(0), **defaults)


def test_reward_prob_is_bernoulli_performance_model():
    m = BernoulliHandModel(0.2, 0.7)
    assert reward_prob(m, 0, Hand.R, 1.0) == pytest.approx(0.2)
    assert reward_prob(m, 0, Hand.L, 1.0) == pytest.approx(0.8)
    assert reward_prob(m, 1, Hand.R, 0.0) == pytest.approx(0.3)
    for x in (0, 1):
        for a in Hand:
            assert reward_prob(m, x, a, 1.0) + reward_prob(m, x, a, 0.0) == pytest.approx(1.0)


def test_belief_update_is_bayes_rule_with_floor():
    a = agent()
    a.update(1, Hand.R, Hand.L, 0.0)  # counterpart L in context 1
    p = np.array([0.9, 0.1, 0.1])  # P(r=0 | tau, a=R, x=1) for M_L, M_R, M_C
    expected = p / p.sum()
    expected = np.maximum(expected, 0.01)
    expected /= expected.sum()
    assert np.allclose(a.posterior, expected)
    for _ in range(50):
        a.update(1, Hand.R, Hand.L, 0.0)
    assert a.posterior.min() >= 0.01 / (1 + 0.01 * 3) - 1e-12


def test_probability_of_improvement_equals_best_response_to_predictive():
    rng = np.random.default_rng(1)
    for _ in range(200):
        a = agent()
        a._belief = rng.dirichlet(np.ones(3))
        pred = a._belief @ np.array([[m.theta0, m.theta1] for m in LIB])
        for x in (0, 1):
            if abs(pred[x] - 0.5) > 1e-9:
                assert a.greedy_action(x) == (Hand.R if pred[x] > 0.5 else Hand.L)


def test_ends_best_responding_after_within_library_switch():
    # Right after the switch the 7-round window straddles it, so no known
    # model fits and BPR+ (as specified) may relearn a copy of M_R; either
    # way it must end up best-responding to M_R in both contexts.
    spec = inst.condition_by_label("within_library_switch")
    for seed in range(5):
        a = BPRPlusAgent(LIB, np.random.default_rng(seed), rho=0.67, epsilon=0.0)
        run_episode(a, spec.condition, seed)
        assert a.greedy_action(0) == Hand.R and a.greedy_action(1) == Hand.R


def test_new_model_detection_learning_phase_and_library_growth():
    a = agent(rho=0.95, epsilon=0.0, learn_rounds=4)
    seq = [(0, Hand.R), (1, Hand.R), (0, Hand.R), (1, Hand.L), (0, Hand.R), (1, Hand.L), (0, Hand.R)]
    for x, h in seq:
        act = a.act(x)
        a.update(x, act, h, float(act == h))
    assert a.detection_times == [6] and a.learning
    learn = [(0, Hand.R), (0, Hand.R), (1, Hand.L), (1, Hand.R)]
    for x, h in learn:
        act = a.act(x)
        a.update(x, act, h, float(act == h))
    assert not a.learning
    t, m = a.learned_models[0]
    assert t == 10
    assert m == BernoulliHandModel((2 + 1) / (2 + 2), (1 + 1) / (2 + 2))
    assert len(a.hypotheses) == 4
    assert a.posterior[-1] == pytest.approx(1 - 0.01 * 3)


def test_no_detection_without_full_window_and_epsilon_explores():
    a = agent(rho=1.0)
    for _ in range(6):
        a.update(0, Hand.L, Hand.L, 1.0)
    assert a.detection_times == []
    a.update(0, Hand.L, Hand.L, 1.0)
    assert a.detection_times == [6]  # rho = 1 always fires once the window is full

    explore = BPRPlusAgent(LIB, np.random.default_rng(0), rho=0.0, epsilon=1.0)
    acts = [explore.act(0) for _ in range(400)]
    assert 0.4 < np.mean([x == Hand.R for x in acts]) < 0.6


def test_registered_agent_is_deterministic_and_instrumentable():
    spec = inst.condition_by_label("outside_library_switch_m2")
    r1 = [l.reward for l in run_episode(make_bpr_agent(3), spec.condition, 3)]
    r2 = [l.reward for l in run_episode(make_bpr_agent(3), spec.condition, 3)]
    assert r1 == r2
    ag = METHODS["BPR+"].factory(3, spec)
    res = inst.run_instrumented(ag, spec, 3, "BPR+", log_level="full")
    assert [float(x) for x in res.per_step["ego_reward"]] == r1
    assert res.summary["n_detections"] == len(ag.detection_times)
    assert np.isfinite(res.summary["kld_final"])


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        BPRPlusAgent(LIB, np.random.default_rng(0), rho=0.5, belief_floor=0.5)
    with pytest.raises(ValueError):
        BPRPlusAgent((), np.random.default_rng(0), rho=0.5)
    assert best_response(BernoulliHandModel(0.5, 0.51), 0) == Hand.L
