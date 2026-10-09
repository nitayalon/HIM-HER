from __future__ import annotations

from himher.experiment import conditions as c


def test_stationary_has_a_single_segment():
    condition = c.stationary()
    assert condition.episode_length == c.EPISODE_LENGTH
    assert condition.schedule.policy_at(0) == c.M_L
    assert condition.schedule.policy_at(c.EPISODE_LENGTH - 1) == c.M_L


def test_within_library_switch_switches_at_the_midpoint_between_library_models():
    condition = c.within_library_switch(episode_length=100)
    assert condition.schedule.policy_at(49) == c.M_L
    assert condition.schedule.policy_at(50) == c.M_R
    assert condition.true_models_in_order == (c.M_L, c.M_R)


def test_outside_library_switch_targets_a_model_outside_the_library():
    condition = c.outside_library_switch(c.M_NEW_1, episode_length=100)
    assert c.M_NEW_1 not in c.LIBRARY_0
    assert condition.schedule.policy_at(49) == c.M_L
    assert condition.schedule.policy_at(50) == c.M_NEW_1


def test_recurring_revisits_m_new_at_the_final_quarter():
    condition = c.recurring(c.M_NEW_1, episode_length=100)
    assert condition.schedule.policy_at(0) == c.M_L
    assert condition.schedule.policy_at(25) == c.M_NEW_1
    assert condition.schedule.policy_at(50) == c.M_R
    assert condition.schedule.policy_at(75) == c.M_NEW_1
    assert condition.true_models_in_order == (c.M_L, c.M_NEW_1, c.M_R, c.M_NEW_1)
