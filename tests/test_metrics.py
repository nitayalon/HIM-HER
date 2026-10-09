from __future__ import annotations

import pytest

from himher.core.cascade import CascadeStepLog
from himher.core.him import HIMResult
from himher.core.regime import Attempt, Regime
from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.experiment import metrics as m
from himher.experiment.runner import RoundLog

M_L = BernoulliHandModel(0.1, 0.1)
M_R = BernoulliHandModel(0.9, 0.9)


def test_steady_state_return_uses_the_better_of_theta_and_its_complement():
    assert m.steady_state_return(BernoulliHandModel(0.9, 0.9)) == pytest.approx(0.9)
    assert m.steady_state_return(BernoulliHandModel(0.1, 0.1)) == pytest.approx(0.9)
    assert m.steady_state_return(BernoulliHandModel(0.5, 0.5)) == pytest.approx(0.5)
    assert m.steady_state_return(BernoulliHandModel(0.2, 0.8)) == pytest.approx(0.8)


def test_optimal_reward_at_picks_the_dominant_action():
    model = BernoulliHandModel(theta0=0.2, theta1=0.8)
    assert m.optimal_reward_at(0, model) == pytest.approx(0.8)  # L dominates (0.8 prob)
    assert m.optimal_reward_at(1, model) == pytest.approx(0.8)  # R dominates (0.8 prob)


def _round(t, context, ego, counterpart, reward) -> RoundLog:
    return RoundLog(t, context, ego, counterpart, reward)


def test_cumulative_return_sums_rewards():
    from himher.domains.handoff.env import Hand

    logs = [
        _round(0, 0, Hand.R, Hand.R, 1.0),
        _round(1, 0, Hand.L, Hand.R, 0.0),
        _round(2, 1, Hand.R, Hand.R, 1.0),
    ]
    assert m.cumulative_return(logs) == pytest.approx(2.0)


def test_post_switch_regret_ignores_rounds_before_the_switch():
    from himher.domains.handoff.env import Hand

    logs = [
        _round(0, 0, Hand.L, Hand.L, 1.0),  # pre-switch, ignored regardless of fit
        _round(1, 0, Hand.L, Hand.R, 0.0),  # post-switch (switch_round=1), true model m_R
        _round(2, 0, Hand.R, Hand.R, 1.0),  # post-switch, optimal
    ]

    def true_model_at(t: int) -> BernoulliHandModel:
        return M_R

    regret = m.post_switch_regret(logs, true_model_at, switch_round=1)
    # round 1: optimal 0.9 - realized 0.0 = 0.9; round 2: optimal 0.9 - realized 1.0 = -0.1
    assert regret == pytest.approx(0.9 - 0.1)


def test_adaptation_delay_finds_the_first_full_window_meeting_the_target():
    from himher.domains.handoff.env import Hand

    # switch at round 5; rewards are 0 until round 10, then 1 forever.
    # With window=3 and fraction=0.9 against m_R (steady_state=0.9), the
    # trailing average first reaches 0.9 once 3 consecutive 1.0-reward
    # rounds have occurred: rounds 10, 11, 12 -> delay = 12 - 5 = 7.
    logs = []
    for t in range(20):
        reward = 1.0 if t >= 10 else 0.0
        logs.append(_round(t, 0, Hand.R, Hand.R if reward else Hand.L, reward))

    def true_model_at(t: int) -> BernoulliHandModel:
        return M_R

    delay = m.adaptation_delay(logs, true_model_at, switch_round=5, window=3, fraction=0.9)
    assert delay == 7


def test_adaptation_delay_returns_none_if_never_reached():
    from himher.domains.handoff.env import Hand

    logs = [_round(t, 0, Hand.L, Hand.R, 0.0) for t in range(20)]

    def true_model_at(t: int) -> BernoulliHandModel:
        return M_R

    assert m.adaptation_delay(logs, true_model_at, switch_round=0, window=5, fraction=0.9) is None


def _step_log(detected: bool, revised: bool, regime, attempts=()) -> CascadeStepLog:
    return CascadeStepLog(
        detected=detected,
        statistic=0.0,
        kappa=0.3,
        revised=revised,
        regime=regime,
        attempts=attempts,
        incumbent_before=M_L,
        incumbent_after=M_R if revised else M_L,
    )


def test_false_alarm_rate_counts_detected_fraction():
    logs = [_step_log(True, False, None), _step_log(False, False, None), _step_log(False, False, None)]
    assert m.false_alarm_rate(logs) == pytest.approx(1 / 3)


def test_false_alarm_rate_of_empty_logs_is_zero():
    assert m.false_alarm_rate([]) == 0.0


def test_detection_delay_finds_first_detection_at_or_after_switch():
    logs = [
        _step_log(False, False, None),
        _step_log(False, False, None),
        _step_log(True, False, None),
        _step_log(True, True, Regime.REUSE),
    ]
    assert m.detection_delay(logs, switch_round=2) == 0
    assert m.detection_delay(logs, switch_round=1) == 1


def test_detection_delay_is_none_if_never_detected_after_the_switch():
    logs = [_step_log(False, False, None) for _ in range(5)]
    assert m.detection_delay(logs, switch_round=0) is None


def _attempt(regime: Regime, accepted: bool) -> Attempt:
    result = HIMResult(
        accepted=accepted, score=0.0, min_improvement=0.0, min_distinctness=0.0
    )
    return Attempt(regime, M_R, result)


def test_computational_economy_counts_each_category():
    logs = [
        _step_log(True, False, None, attempts=(_attempt(Regime.REUSE, False),)),
        _step_log(
            True,
            True,
            Regime.REVISE,
            attempts=(_attempt(Regime.REUSE, False), _attempt(Regime.REVISE, True)),
        ),
        _step_log(
            True,
            True,
            Regime.CONSTRUCT,
            attempts=(
                _attempt(Regime.REUSE, False),
                _attempt(Regime.REVISE, False),
                _attempt(Regime.CONSTRUCT, True),
            ),
        ),
        _step_log(False, False, None),
    ]
    economy = m.computational_economy(logs)
    assert economy == {
        "detections": 3,
        "constructions": 1,
        "revise_attempts": 2,
        "reuse_accepted": 0,
        "revise_accepted": 1,
        "construct_accepted": 1,
        "revisions": 2,
    }


def test_model_accuracy_measures_distance_for_the_first_accepted_construct():
    true_model = BernoulliHandModel(0.85, 0.95)
    logs = [
        _step_log(True, False, None),
        CascadeStepLog(
            detected=True,
            statistic=1.0,
            kappa=0.3,
            revised=True,
            regime=Regime.CONSTRUCT,
            attempts=(),
            incumbent_before=M_L,
            incumbent_after=M_R,
        ),
    ]
    import numpy as np

    expected = float(np.linalg.norm(np.array(M_R.theta) - np.array(true_model.theta)))
    assert m.model_accuracy(logs, true_model) == pytest.approx(expected)


def test_model_accuracy_also_measures_distance_for_an_accepted_revise():
    true_model = BernoulliHandModel(0.1, 0.5)
    revised_model = BernoulliHandModel(0.1, 0.5)  # e.g. the M_L/M_C midpoint
    logs = [
        CascadeStepLog(
            detected=True,
            statistic=1.0,
            kappa=0.3,
            revised=True,
            regime=Regime.REVISE,
            attempts=(),
            incumbent_before=M_L,
            incumbent_after=revised_model,
        ),
    ]
    assert m.model_accuracy(logs, true_model) == pytest.approx(0.0)


def test_model_accuracy_is_none_when_neither_revise_nor_construct_accepted():
    logs = [_step_log(True, True, Regime.REUSE)]
    assert m.model_accuracy(logs, M_R) is None


def test_replay_benefit_is_positive_when_replay_is_faster():
    assert m.replay_benefit(delay_with_replay=3, delay_without_replay=10) == 7


def test_replay_benefit_is_none_if_either_run_never_adapted():
    assert m.replay_benefit(None, 10) is None
    assert m.replay_benefit(3, None) is None
