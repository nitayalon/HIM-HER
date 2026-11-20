import numpy as np
import pytest

from himher.baselines.continuous_bayes import ContinuousBayesAgent
from himher.baselines.discounted_bayes import DiscountedBayesAgent
from himher.baselines.windowed_bayes import WindowedBayesAgent
from himher.domains.handoff.env import Hand
from himher.experiment import conditions as cond
from himher.experiment.runner import run_episode


def _stream(seed, n=300):
    rng = np.random.default_rng(seed)
    return [(int(rng.random() < 0.5), Hand(int(rng.random() < 0.7))) for _ in range(n)]


def _assert_tracks_cb(agent, seed):
    cb = ContinuousBayesAgent(alpha=1.0)
    for x, a in _stream(seed):
        for c in (0, 1):
            assert agent.act(c) == cb.act(c)
        assert agent.current_model() == cb.current_model()  # exact float equality
        agent.update(x, Hand.L, a, 0.0)
        cb.update(x, Hand.L, a, 0.0)
    assert agent.current_model() == cb.current_model()


@pytest.mark.parametrize("seed", range(5))
def test_wb_with_window_longer_than_episode_equals_cb(seed):
    _assert_tracks_cb(WindowedBayesAgent(alpha=1.0, window=10_000), seed)


@pytest.mark.parametrize("seed", range(5))
def test_db_with_gamma_one_equals_cb(seed):
    _assert_tracks_cb(DiscountedBayesAgent(alpha=1.0, gamma=1.0), seed)


@pytest.mark.parametrize("label_factory", [cond.within_library_switch, cond.recurring])
def test_wb_and_db_limits_reproduce_cb_episodes(label_factory):
    condition = label_factory()
    for seed in range(5):
        cb = [l.reward for l in run_episode(ContinuousBayesAgent(1.0), condition, seed)]
        wb = [l.reward for l in run_episode(WindowedBayesAgent(1.0, 10_000), condition, seed)]
        db = [l.reward for l in run_episode(DiscountedBayesAgent(1.0, 1.0), condition, seed)]
        assert cb == wb == db


def test_wb_converges_to_last_k_empirical_mean():
    k = 20
    agent = WindowedBayesAgent(alpha=1.0, window=k)
    for _ in range(100):  # long stale run of L in context 0
        agent.update(0, Hand.L, Hand.L, 0.0)
    for _ in range(200):  # constant R stream in context 0
        agent.update(0, Hand.L, Hand.R, 0.0)
    # posterior mean = (k + 1) / (k + 2): only the last k (all R) count
    assert agent.current_model().theta0 == pytest.approx((k + 1) / (k + 2))
    # the window is per context: context 1 is untouched (prior mean)
    assert agent.current_model().theta1 == 0.5


def test_wb_window_is_per_context():
    agent = WindowedBayesAgent(alpha=1.0, window=3)
    for a in (Hand.R, Hand.R, Hand.R):
        agent.update(0, Hand.L, a, 0.0)
    for _ in range(10):
        agent.update(1, Hand.L, Hand.L, 0.0)  # must not evict context-0 data
    assert agent.current_model().theta0 == pytest.approx(4 / 5)


def test_db_discounts_evidence_not_prior():
    g = 0.9
    agent = DiscountedBayesAgent(alpha=1.0, gamma=g)
    for _ in range(500):
        agent.update(1, Hand.L, Hand.R, 0.0)
    # fixed point: a - 1 = g (a - 1) + 1 -> a = 1 + 1 / (1 - g); b stays at prior 1
    a_star = 1 + 1 / (1 - g)
    assert agent.current_model().theta1 == pytest.approx(a_star / (a_star + 1.0))
    assert agent.current_model().theta0 == 0.5  # no observation in context 0: no discount


def test_db_only_discounts_on_observations_in_that_context():
    agent = DiscountedBayesAgent(alpha=1.0, gamma=0.5)
    agent.update(0, Hand.L, Hand.R, 0.0)  # a0 = 2, b0 = 1
    before = agent.current_model().theta0
    for _ in range(50):
        agent.update(1, Hand.L, Hand.L, 0.0)
    assert agent.current_model().theta0 == before
