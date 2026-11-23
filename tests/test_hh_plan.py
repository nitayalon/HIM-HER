"""Task P (Mon 11/23): HH-plan = H+H's cascade + WB-k20's decision rule."""

from __future__ import annotations

import numpy as np
import pytest

from himher.baselines.hh_plan import EpsilonGreedy, HHPlanAgent, wb_decision
from himher.baselines.windowed_bayes import WindowedBayesAgent
from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Hand, SwitchingHandoffGame
from himher.experiment import instrumentation as inst
from himher.experiment import rebuttal_methods as rm

LABELS = [s.label for s in inst.condition_specs()]


def _recorded_observations(label: str, seed: int) -> list[tuple[int, Hand]]:
    spec = inst.condition_by_label(label)
    game = SwitchingHandoffGame(spec.condition.schedule, np.random.default_rng(seed))
    obs = []
    for _ in range(spec.condition.episode_length):
        x = game.sample_context()
        obs.append((x, game.step(x, Hand.L).counterpart_action))
    return obs


def _trajectory(agent) -> list[tuple]:
    return [(s.detected, s.revised, s.regime, s.incumbent_after.theta, len(s.attempts)) for s in agent.step_logs]


@pytest.mark.parametrize("label", LABELS)
def test_model_trajectory_equals_hhr_on_same_observations(label):
    for seed in range(3):
        obs = _recorded_observations(label, seed)
        hhr = rm.make_hh_agent(seed, replay=True)
        plan = rm.make_hh_plan_agent(seed)
        rng = np.random.default_rng(99)
        for x, a in obs:
            ego = Hand(int(rng.integers(0, 2)))  # ego actions are irrelevant to the cascade
            hhr.act(x)
            plan.act(x)
            hhr.update(x, ego, a, float(ego == a))
            plan.update(x, ego, a, float(ego == a))
            assert plan.cascade.incumbent == hhr.cascade.incumbent
        assert _trajectory(plan) == _trajectory(hhr)
        assert tuple(plan.cascade.library) == tuple(hhr.cascade.library)


def test_model_trajectory_equals_hhr_in_full_episodes():
    """In the game the cascade sees only (context, counterpart action), which
    do not depend on ego actions, so whole episodes match too."""
    for label in LABELS:
        spec = inst.condition_by_label(label)
        for seed in range(2):
            a = inst.run_instrumented(rm.make_hh_agent(seed, True), spec, seed, "HH-R", log_level="events")
            b = inst.run_instrumented(rm.make_hh_plan_agent(seed), spec, seed, "HH-plan", log_level="events")
            strip = lambda ev: [{k: v for k, v in e.items() if k != "run_id"} for e in ev  # noqa: E731
                                if e["event"] != "eps_reset"]
            ea = [{k: v for k, v in e.items() if k not in ("n_replayed", "replay_start", "replay_end")}
                  for e in strip(a.events)]
            assert ea == strip(b.events)
            for k in ("n_detections", "n_false_alarms", "n_accept_reuse", "n_accept_merge",
                      "n_accept_construct", "n_reject", "kld_final"):
                assert a.summary[k] == b.summary[k] or (np.isnan(a.summary[k]) and np.isnan(b.summary[k]))


def test_actions_equal_wb_when_model_is_wb_posterior():
    wb = WindowedBayesAgent(alpha=rm.ALPHA, window=rm.WB_WINDOW)
    plan = rm.make_hh_plan_agent(0)
    for x, a in _recorded_observations("within_library_switch", 3):
        plan.cascade.incumbent = wb.current_model()  # H+H's model := WB's posterior mean
        assert plan.act(x) == wb.act(x)
        wb.update(x, Hand.L, a, 0.0)


def test_wb_decision_rule_and_ties():
    assert wb_decision(BernoulliHandModel(0.2, 0.8), 0) == Hand.L
    assert wb_decision(BernoulliHandModel(0.2, 0.8), 1) == Hand.R
    assert wb_decision(BernoulliHandModel(0.5, 0.5), 1) == Hand.L  # ties -> L, as WB


def test_eps_variants_share_exploration_draws():
    plan = rm.make_hh_plan_agent(7, rm.EPSILON)
    wb = rm.make_wb_eps_agent(7)
    hhr_policy_rng = np.random.default_rng(7)
    n_explore = 0
    for x, a in _recorded_observations("stationary", 7):
        explore = hhr_policy_rng.random() < rm.EPSILON
        if explore:
            hhr_policy_rng.integers(0, 2)
            n_explore += 1
            assert plan.act(x) == wb.act(x)  # same random hand
        else:
            assert plan.act(x) == plan.greedy_action(x)
            assert wb.act(x) == wb.greedy_action(x)
        plan.update(x, Hand.L, a, 0.0)
        wb.update(x, Hand.L, a, 0.0)
    assert 20 < n_explore < 80


def test_no_exploration_by_default():
    plan = rm.make_hh_plan_agent(0)
    assert plan.policy.epsilon == 0 and plan.policy.total_updates == 0
    with pytest.raises(ValueError):
        HHPlanAgent(rm.make_hh_cascade(), None, epsilon=0.1)
    assert isinstance(rm.make_wb_eps_agent(0), EpsilonGreedy)
