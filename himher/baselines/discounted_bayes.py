"""Discounted Beta-Bayes baseline (rebuttal, method ``DB-<gamma>``).

Identical to ContinuousBayesAgent (CB) -- same per-context Beta(alpha0,
alpha0) prior, same greedy posterior-mean decision rule (R iff
theta_x > 0.5, ties to L), no exploration -- except for the belief
update. On each observation y in {0, 1} (1 = R) in context x:

    a_x <- alpha0 + gamma * (a_x - alpha0) + y
    b_x <- alpha0 + gamma * (b_x - alpha0) + (1 - y)

i.e. the *evidence* is discounted, never the prior. CB decays nothing per
step, so discounting happens only on observations in that context, not on
every global step. gamma = 1 reproduces CB exactly.
"""

from __future__ import annotations

from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Context, Hand


class DiscountedBayesAgent:
    def __init__(self, alpha: float = 1.0, gamma: float = 0.95) -> None:
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        self._alpha0 = alpha
        self._gamma = gamma
        self._a = {0: alpha, 1: alpha}
        self._b = {0: alpha, 1: alpha}

    def _theta(self, context: Context) -> float:
        a, b = self._a[context], self._b[context]
        return a / (a + b)

    def current_model(self) -> BernoulliHandModel:
        return BernoulliHandModel(theta0=self._theta(0), theta1=self._theta(1))

    def act(self, context: Context) -> Hand:
        model = self.current_model()
        theta_x = model.theta1 if context else model.theta0
        return Hand.R if theta_x > 0.5 else Hand.L

    def update(
        self, context: Context, ego_action: Hand, counterpart_action: Hand, reward: float
    ) -> None:
        y = 1.0 if counterpart_action == Hand.R else 0.0
        a0, g = self._alpha0, self._gamma
        self._a[context] = a0 + g * (self._a[context] - a0) + y
        self._b[context] = a0 + g * (self._b[context] - a0) + (1.0 - y)
