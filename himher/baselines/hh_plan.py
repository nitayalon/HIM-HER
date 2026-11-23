"""HH-plan (rebuttal Mon 11/23, Task P): H+H's model revision with
closed-form action selection.

Detection, VOC, reuse/merge/construct and HIM acceptance are the HHCascade,
called exactly as HHAgent calls it (``cascade.step(context,
counterpart_action)`` once per round). Action selection is **WB-k20's own
decision function**, ``WindowedBayesAgent.act``, applied to a view whose
``current_model()`` is H+H's incumbent instead of WB's windowed posterior
(R iff theta_x > 0.5, ties to L). There is no policy learning and no HER
relabeling/replay (nothing learned to act on them).

Exploration is WB-k20's, i.e. none, unless ``epsilon > 0`` (HH-plan-eps):
then each round, with probability epsilon, a uniformly random hand, drawn
with exactly TabularQPolicy.act's RNG calls (``rng.random()`` then
``rng.integers(0, 2)``). ``EpsilonGreedy`` applies the same rule to any
agent (WB-k20-eps), so HH-plan-eps and WB-k20-eps given the same seed make
identical exploration draws in every round.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from himher.baselines.windowed_bayes import WindowedBayesAgent
from himher.core.cascade import CascadeStepLog, HHCascade
from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Context, Hand


class _ModelView:
    """Duck-types the one attribute WindowedBayesAgent.act reads."""

    def __init__(self, model: BernoulliHandModel) -> None:
        self._model = model

    def current_model(self) -> BernoulliHandModel:
        return self._model


def wb_decision(model: BernoulliHandModel, context: Context) -> Hand:
    """WB-k20's decision function, unmodified, on an arbitrary model."""
    return WindowedBayesAgent.act(_ModelView(model), context)  # type: ignore[arg-type]


def _explore(rng: np.random.Generator, epsilon: float) -> Hand | None:
    if rng.random() < epsilon:  # same call sequence as TabularQPolicy.act
        return Hand(int(rng.integers(0, 2)))
    return None


class _NoPolicy:
    """Read by the instrumentation (n_policy_updates, logged epsilon)."""

    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon
        self.total_updates = 0


class HHPlanAgent:
    def __init__(self, cascade: HHCascade, rng: np.random.Generator | None = None,
                 epsilon: float = 0.0) -> None:
        if epsilon < 0 or epsilon > 1:
            raise ValueError("epsilon must be in [0, 1]")
        if epsilon > 0 and rng is None:
            raise ValueError("epsilon > 0 needs an rng")
        self._cascade = cascade
        self._rng = rng
        self._epsilon = epsilon
        self._policy = _NoPolicy(epsilon)
        self.last_step_log: CascadeStepLog | None = None
        self.step_logs: list[CascadeStepLog] = []
        self.last_epsilon = epsilon

    @property
    def cascade(self) -> HHCascade:
        return self._cascade

    @property
    def policy(self) -> _NoPolicy:
        return self._policy

    def greedy_action(self, context: Context) -> Hand:
        return wb_decision(self._cascade.incumbent, context)

    def act(self, context: Context) -> Hand:
        if self._epsilon > 0:
            explored = _explore(self._rng, self._epsilon)
            if explored is not None:
                return explored
        return self.greedy_action(context)

    def update(self, context: Context, ego_action: Hand, counterpart_action: Hand, reward: float) -> None:
        step_log = self._cascade.step(context, counterpart_action)
        self.last_step_log = step_log
        self.step_logs.append(step_log)


class EpsilonGreedy:
    """Adds constant-epsilon uniform exploration to an agent's act()."""

    def __init__(self, base: Any, epsilon: float, rng: np.random.Generator) -> None:
        self._base = base
        self._epsilon = epsilon
        self._rng = rng
        self.last_epsilon = epsilon

    def act(self, context: Context) -> Hand:
        explored = _explore(self._rng, self._epsilon)
        return explored if explored is not None else self._base.act(context)

    def greedy_action(self, context: Context) -> Hand:
        return self._base.act(context)

    def update(self, context: Context, ego_action: Hand, counterpart_action: Hand, reward: float) -> None:
        self._base.update(context, ego_action, counterpart_action, reward)

    def current_model(self) -> BernoulliHandModel:
        return self._base.current_model()
