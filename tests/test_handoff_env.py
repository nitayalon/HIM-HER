from __future__ import annotations

import numpy as np
import pytest

from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Hand, SwitchingHandoffGame, SwitchSchedule


def _stationary_schedule(model: BernoulliHandModel) -> SwitchSchedule:
    return SwitchSchedule(segments=[(0, model)])


def test_reward_is_one_iff_actions_match():
    always_right = BernoulliHandModel(theta0=1.0, theta1=1.0)
    game = SwitchingHandoffGame(_stationary_schedule(always_right), rng=np.random.default_rng(0))

    matching = game.step(context=0, ego_action=Hand.R)
    mismatching = game.step(context=0, ego_action=Hand.L)

    assert matching.reward == 1.0
    assert matching.counterpart_action == Hand.R
    assert mismatching.reward == 0.0
    assert mismatching.counterpart_action == Hand.R


def test_sample_context_is_roughly_balanced():
    game = SwitchingHandoffGame(
        _stationary_schedule(BernoulliHandModel(0.5, 0.5)), rng=np.random.default_rng(1)
    )
    contexts = [game.sample_context() for _ in range(20_000)]
    assert set(contexts) <= {0, 1}
    assert np.mean(contexts) == pytest.approx(0.5, abs=0.02)


def test_step_advances_the_round_counter():
    game = SwitchingHandoffGame(
        _stationary_schedule(BernoulliHandModel(0.5, 0.5)), rng=np.random.default_rng(2)
    )
    assert game.t == 0
    game.step(context=0, ego_action=Hand.L)
    assert game.t == 1
    game.step(context=1, ego_action=Hand.R)
    assert game.t == 2


def test_switch_schedule_rejects_empty_segments():
    with pytest.raises(ValueError):
        SwitchSchedule(segments=[])


def test_switch_schedule_rejects_segments_not_starting_at_zero():
    model = BernoulliHandModel(0.1, 0.9)
    with pytest.raises(ValueError):
        SwitchSchedule(segments=[(10, model)])


def test_switch_schedule_rejects_unsorted_segments():
    model = BernoulliHandModel(0.1, 0.9)
    with pytest.raises(ValueError):
        SwitchSchedule(segments=[(0, model), (50, model), (20, model)])


def test_switch_schedule_selects_correct_segment_by_round():
    m_left = BernoulliHandModel(0.1, 0.1)
    m_right = BernoulliHandModel(0.9, 0.9)
    m_new = BernoulliHandModel(0.2, 0.8)
    schedule = SwitchSchedule(segments=[(0, m_left), (100, m_right), (200, m_new)])

    assert schedule.policy_at(0) is m_left
    assert schedule.policy_at(99) is m_left
    assert schedule.policy_at(100) is m_right
    assert schedule.policy_at(150) is m_right
    assert schedule.policy_at(200) is m_new
    assert schedule.policy_at(1000) is m_new


def test_within_library_switch_condition_changes_the_true_generator_mid_episode():
    # Mirrors Section 5.3's "Within library switch" condition: m_L for the first
    # half of the episode, m_R for the second half. The environment must
    # actually draw from the new generator after the switch round.
    m_left = BernoulliHandModel(0.0, 0.0)  # always L, deterministic for the test
    m_right = BernoulliHandModel(1.0, 1.0)  # always R, deterministic for the test
    switch_round = 50
    schedule = SwitchSchedule(segments=[(0, m_left), (switch_round, m_right)])
    game = SwitchingHandoffGame(schedule, rng=np.random.default_rng(3))

    actions_before = [game.step(context=0, ego_action=Hand.L).counterpart_action for _ in range(switch_round)]
    actions_after = [game.step(context=0, ego_action=Hand.R).counterpart_action for _ in range(switch_round)]

    assert all(a == Hand.L for a in actions_before)
    assert all(a == Hand.R for a in actions_after)
