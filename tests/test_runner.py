from __future__ import annotations

from himher.domains.handoff.env import Context, Hand
from himher.experiment.conditions import stationary, within_library_switch
from himher.experiment.runner import run_episode, run_seeds


class _AlwaysRightAgent:
    def act(self, context: Context) -> Hand:
        return Hand.R

    def update(self, context, ego_action, counterpart_action, reward) -> None:
        pass


def test_run_episode_produces_one_log_per_round():
    condition = stationary(episode_length=50)
    logs = run_episode(_AlwaysRightAgent(), condition, seed=0)
    assert len(logs) == 50
    assert [log.t for log in logs] == list(range(50))


def test_run_episode_rewards_match_action_comparison():
    condition = stationary(episode_length=50)
    logs = run_episode(_AlwaysRightAgent(), condition, seed=0)
    for log in logs:
        assert log.ego_action == Hand.R
        expected_reward = 1.0 if log.ego_action == log.counterpart_action else 0.0
        assert log.reward == expected_reward


def test_run_episode_is_reproducible_given_the_same_seed():
    condition = within_library_switch(episode_length=50)
    logs_a = run_episode(_AlwaysRightAgent(), condition, seed=42)
    logs_b = run_episode(_AlwaysRightAgent(), condition, seed=42)
    assert logs_a == logs_b


def test_run_seeds_runs_one_fresh_agent_per_seed():
    condition = stationary(episode_length=20)
    episodes = run_seeds(_AlwaysRightAgent, condition, seeds=[0, 1, 2])
    assert len(episodes) == 3
    assert all(len(episode) == 20 for episode in episodes)
