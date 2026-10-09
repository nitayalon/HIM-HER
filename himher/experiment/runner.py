"""Runs a single agent against a single condition for one seeded episode.
Both the Agent and the Bayesian baselines (Section 5.4) implement the same
act/update interface (himher.baselines.agent_base.Agent), so this runner
does not need to know which kind of agent it is driving.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from himher.baselines.agent_base import Agent
from himher.domains.handoff.env import Context, Hand, SwitchingHandoffGame
from himher.experiment.conditions import Condition


@dataclass(frozen=True)
class RoundLog:
    t: int
    context: Context
    ego_action: Hand
    counterpart_action: Hand
    reward: float


def run_episode(agent: Agent, condition: Condition, seed: int) -> list[RoundLog]:
    rng = np.random.default_rng(seed)
    game = SwitchingHandoffGame(condition.schedule, rng)
    logs = []
    for t in range(condition.episode_length):
        context = game.sample_context()
        ego_action = agent.act(context)
        result = game.step(context, ego_action)
        agent.update(context, ego_action, result.counterpart_action, result.reward)
        logs.append(RoundLog(t, context, ego_action, result.counterpart_action, result.reward))
    return logs


def run_seeds(
    agent_factory: Callable[[], Agent], condition: Condition, seeds: Sequence[int]
) -> list[list[RoundLog]]:
    """``agent_factory`` must return a fresh agent instance per seed,
    since agents are stateful.
    """
    return [run_episode(agent_factory(), condition, seed) for seed in seeds]
