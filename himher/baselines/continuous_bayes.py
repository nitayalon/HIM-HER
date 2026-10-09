"""Continuous Bayesian adaptation baseline (Section 5.4): independent
Beta(alpha, alpha) posteriors over theta_0 and theta_1, updated after
every interaction regardless of detection -- the "always adapt" extreme
against which H+H's selective revision is compared.

Reuses the domain's Beta-Bernoulli construction formula (Section 5.2)
applied to the entire history rather than a fixed window; the only
difference from H+H's Construct regime is that this baseline always
re-estimates and always acts on the result, with no detection or
acceptance gate.
"""

from __future__ import annotations

from himher.domains.handoff.construct import construct_bernoulli_model
from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Context, Hand


class ContinuousBayesAgent:
    def __init__(self, alpha: float = 1.0) -> None:
        self._alpha = alpha
        self._history: list[tuple[Context, Hand]] = []

    def current_model(self) -> BernoulliHandModel:
        return construct_bernoulli_model(self._history, self._alpha)

    def act(self, context: Context) -> Hand:
        model = self.current_model()
        theta_x = model.theta1 if context else model.theta0
        return Hand.R if theta_x > 0.5 else Hand.L

    def update(
        self, context: Context, ego_action: Hand, counterpart_action: Hand, reward: float
    ) -> None:
        self._history.append((context, counterpart_action))
