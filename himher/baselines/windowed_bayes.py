"""Windowed Beta-Bayes baseline (rebuttal, method ``WB-k<window>``).

Identical to ContinuousBayesAgent (CB) -- same per-context Beta(alpha,
alpha) prior, same greedy posterior-mean decision rule (R iff
theta_x > 0.5, ties to L), no exploration -- except that each context's
posterior uses only the last ``window`` observations *in that context*
(CB already conditions per context, via construct_bernoulli_model's
per-context counts). With window >= the episode length it reproduces CB
exactly (tests/test_rebuttal_baselines.py).

A fixed-length deque per context is the ring buffer; the "validity mask"
of a static-shape implementation is implicit in the deque's length.
"""

from __future__ import annotations

from collections import deque

from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Context, Hand


class WindowedBayesAgent:
    def __init__(self, alpha: float = 1.0, window: int = 20) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self._alpha = alpha
        self._window = window
        self._obs: dict[Context, deque[int]] = {0: deque(maxlen=window), 1: deque(maxlen=window)}

    def _theta(self, context: Context) -> float:
        # Same expression as construct_bernoulli_model.theta_for, so a
        # window that never evicts gives bitwise-identical estimates to CB.
        obs = self._obs[context]
        n_x = len(obs)
        n_x_right = sum(obs)
        return (n_x_right + self._alpha) / (n_x + 2 * self._alpha)

    def current_model(self) -> BernoulliHandModel:
        return BernoulliHandModel(theta0=self._theta(0), theta1=self._theta(1))

    def act(self, context: Context) -> Hand:
        model = self.current_model()
        theta_x = model.theta1 if context else model.theta0
        return Hand.R if theta_x > 0.5 else Hand.L

    def update(
        self, context: Context, ego_action: Hand, counterpart_action: Hand, reward: float
    ) -> None:
        self._obs[context].append(1 if counterpart_action == Hand.R else 0)
