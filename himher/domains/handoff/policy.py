"""Ego response policy for the Switching Handoff Game (Section 5.2):
Q(x, m, a^i), learned by tabular Q-learning and updated via hindsight
replay (Section 4.5) once a new counterpart model is accepted.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Context, Hand


@dataclass(frozen=True)
class Transition:
    context: Context
    ego_action: Hand
    counterpart_action: Hand
    reward: float


class TabularQPolicy:
    """Q(x, m, a) indexed by (context, model, ego_action); ``model`` is
    used directly as a dict key since BernoulliHandModel is a frozen,
    hashable dataclass, so no separate model-id indirection is needed.
    """

    def __init__(self, learning_rate: float = 0.1, epsilon: float = 0.05) -> None:
        self._learning_rate = learning_rate
        self._epsilon = epsilon
        self._q: dict[tuple[Context, BernoulliHandModel, Hand], float] = defaultdict(float)
        # Section 5.5 measure 5 ("...and policy updates"): how many times
        # each model's Q-entries were touched, online or via replay --
        # per model, not just a single aggregate count, so e.g. a
        # correctly-identified incumbent's attention can be compared
        # against a quickly-discarded wrong one.
        self._update_counts: dict[BernoulliHandModel, int] = defaultdict(int)

    def q_value(self, context: Context, model: BernoulliHandModel, action: Hand) -> float:
        return self._q[(context, model, action)]

    def update_count(self, model: BernoulliHandModel) -> int:
        return self._update_counts[model]

    @property
    def update_counts(self) -> dict[BernoulliHandModel, int]:
        return dict(self._update_counts)

    @property
    def total_updates(self) -> int:
        return sum(self._update_counts.values())

    def greedy_action(self, context: Context, model: BernoulliHandModel) -> Hand:
        q_left = self.q_value(context, model, Hand.L)
        q_right = self.q_value(context, model, Hand.R)
        return Hand.R if q_right >= q_left else Hand.L

    def act(self, context: Context, model: BernoulliHandModel, rng: np.random.Generator) -> Hand:
        if rng.random() < self._epsilon:
            return Hand(int(rng.integers(0, 2)))
        return self.greedy_action(context, model)

    def update(self, transition: Transition, model: BernoulliHandModel) -> None:
        # One-shot matching game: there is no next state to bootstrap
        # from, so the Q-learning update reduces to a running average
        # toward the observed reward (the discount/next-state term is
        # identically zero).
        key = (transition.context, model, transition.ego_action)
        self._q[key] += self._learning_rate * (transition.reward - self._q[key])
        self._update_counts[model] += 1

    @staticmethod
    def analytic_best_response(context: Context, model: BernoulliHandModel) -> Hand:
        """The known-optimal response when ``model`` is exactly correct:
        a^{i*} = R iff theta_x > 0.5. Used only as a correctness oracle in
        tests -- Section 5.2's game has an analytic solution given the
        model, so Q-learning is not strictly required for optimality
        here, only for demonstrating the general hindsight-replay
        mechanism.
        """
        theta_x = model.theta1 if context else model.theta0
        return Hand.R if theta_x > 0.5 else Hand.L
