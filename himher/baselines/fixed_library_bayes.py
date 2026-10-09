"""Fixed-library Bayesian baseline (Section 5.4): a categorical posterior
over a fixed, never-expanded hypothesis space of counterpart models,
updated exactly via Bayes' rule each round, acting via the posterior-
predictive best response.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Context, Hand


class FixedLibraryBayesAgent:
    def __init__(self, hypotheses: Sequence[BernoulliHandModel]) -> None:
        if not hypotheses:
            raise ValueError("hypotheses must be non-empty")
        self._hypotheses = tuple(hypotheses)
        self._log_posterior = np.zeros(len(self._hypotheses))  # uniform prior

    @property
    def posterior(self) -> np.ndarray:
        shifted = self._log_posterior - self._log_posterior.max()
        unnormalized = np.exp(shifted)
        return unnormalized / unnormalized.sum()

    @property
    def hypotheses(self) -> tuple[BernoulliHandModel, ...]:
        return self._hypotheses

    def act(self, context: Context) -> Hand:
        posterior = self.posterior
        p_right = sum(
            p * h.predict_proba(context)[Hand.R] for p, h in zip(posterior, self._hypotheses)
        )
        return Hand.R if p_right > 0.5 else Hand.L

    def update(
        self, context: Context, ego_action: Hand, counterpart_action: Hand, reward: float
    ) -> None:
        log_likelihoods = np.array(
            [h.log_prob(context, counterpart_action) for h in self._hypotheses]
        )
        self._log_posterior = self._log_posterior + log_likelihoods
