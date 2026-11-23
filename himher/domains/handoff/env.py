"""Switching Handoff Game (paper Section 5.1).

A contextual repeated coordination game: each round has a binary context
x_t (the handoff type); the counterpart j picks a hand a^j_t; the ego agent
i picks the side a^i_t on which to present the instrument; the ego is
rewarded iff the two match. The counterpart's *true* generating policy can
change over the course of an episode -- that switching is what the H+H
cascade is meant to detect and adapt to. The ego never sees which true
policy is active; it only sees (context, counterpart_action, reward) each
round.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol, Sequence

import numpy as np

from himher.core.schedule import SwitchSchedule  # re-exported for existing imports


class Hand(IntEnum):
    L = 0
    R = 1


Context = int  # x_t in {0, 1}


class CounterpartPolicy(Protocol):
    """Anything that can play the counterpart role in a round."""

    def sample(self, context: Context, rng: np.random.Generator) -> Hand: ...


@dataclass(frozen=True)
class StepResult:
    context: Context
    ego_action: Hand
    counterpart_action: Hand
    reward: float


class SwitchingHandoffGame:
    """One round at a time: sample a context, accept the ego's action,
    sample the (schedule-determined) counterpart's action, and return the
    reward. The game owns the RNG so that a fixed seed reproduces an entire
    episode's contexts and counterpart draws regardless of which agent is
    playing the ego role.
    """

    def __init__(self, schedule: SwitchSchedule, rng: np.random.Generator) -> None:
        self._schedule = schedule
        self._rng = rng
        self._t = 0

    @property
    def t(self) -> int:
        return self._t

    def sample_context(self) -> Context:
        return int(self._rng.random() < 0.5)

    def step(self, context: Context, ego_action: Hand) -> StepResult:
        true_policy = self._schedule.policy_at(self._t)
        counterpart_action = true_policy.sample(context, self._rng)
        reward = 1.0 if ego_action == counterpart_action else 0.0
        result = StepResult(context, ego_action, counterpart_action, reward)
        self._t += 1
        return result
