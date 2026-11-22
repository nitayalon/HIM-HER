"""Bayes-adaptive baseline (Section 5.4): Bayesian Online Changepoint
Detection \\citep{adams2007bocpd} over a fixed hypothesis space, jointly
inferring the counterpart's type *and* whether a changepoint has
occurred -- unlike the other Bayesian baselines, which never represent
"a changepoint happened" at all and so accumulate evidence forever
(Section 5's "non-windowed posterior" finding).

At each round t, BOCD maintains a posterior over the run length r_t --
how many rounds since the last changepoint -- jointly with, for each
live run-length hypothesis, a posterior over the counterpart's type
given only the observations since that run started. A constant hazard
rate ``hazard`` gives a geometric prior on run length (expected run
length 1/hazard): each round, every live run either grows by one round
(probability 1-hazard) or is replaced by a brand-new run of length 0
(probability hazard, pooled across all prior run lengths). This is the
standard BOCD recursion (Adams & MacKay 2007), generalized from a single
conjugate-exponential-family model to a finite mixture over
``hypotheses``: the predictive likelihood used in the recursion is the
*within-run* posterior-predictive mixture over hypotheses, not a single
model's likelihood.

Acting uses the type posterior marginalized over run length -- so unlike
fixed_bayes, old evidence is automatically discounted once the run-length
posterior shifts toward "a changepoint recently occurred," without any
explicit detect-then-revise cascade.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Context, Hand


def _logsumexp(log_values: np.ndarray, axis: int | None = None) -> np.ndarray:
    maximum = np.max(log_values, axis=axis, keepdims=True)
    summed = np.log(np.sum(np.exp(log_values - maximum), axis=axis, keepdims=True))
    result = maximum + summed
    return result.squeeze(axis=axis) if axis is not None else float(result.item())


class BOCDBayesAgent:
    def __init__(self, hypotheses: Sequence[BernoulliHandModel], hazard: float) -> None:
        if not hypotheses:
            raise ValueError("hypotheses must be non-empty")
        if not 0.0 < hazard < 1.0:
            raise ValueError("hazard must be in (0, 1)")
        self._hypotheses = tuple(hypotheses)
        self._hazard = hazard
        n = len(self._hypotheses)
        # One row per live run-length hypothesis r = 0, 1, 2, ...; row r
        # holds log P(x_{t-r:t}; h) for each hypothesis h, accumulated
        # over only that run's observations (uniform/empty prior: all
        # zeros before any data, i.e. a flat predictive mixture).
        self._run_log_likelihoods = np.zeros((1, n))
        # log P(r_t = r | x_{1:t}), normalized (logsumexp = 0).
        self._run_log_probs = np.zeros(1)

    @property
    def hypotheses(self) -> tuple[BernoulliHandModel, ...]:
        return self._hypotheses

    @property
    def run_length_posterior(self) -> np.ndarray:
        """P(r_t = r | data) for r = 0, 1, 2, ..., most recent run first."""
        shifted = self._run_log_probs - self._run_log_probs.max()
        unnormalized = np.exp(shifted)
        return unnormalized / unnormalized.sum()

    @property
    def posterior(self) -> np.ndarray:
        """Type posterior marginalized over run length: P(m | data)."""
        run_weights = self.run_length_posterior
        per_run_type_posterior = self._softmax_rows(self._run_log_likelihoods)
        return run_weights @ per_run_type_posterior

    @staticmethod
    def _softmax_rows(log_values: np.ndarray) -> np.ndarray:
        shifted = log_values - log_values.max(axis=1, keepdims=True)
        unnormalized = np.exp(shifted)
        return unnormalized / unnormalized.sum(axis=1, keepdims=True)

    def act(self, context: Context) -> Hand:
        posterior = self.posterior
        p_right = sum(
            p * h.predict_proba(context)[Hand.R] for p, h in zip(posterior, self._hypotheses)
        )
        return Hand.R if p_right > 0.5 else Hand.L

    def update(
        self, context: Context, ego_action: Hand, counterpart_action: Hand, reward: float
    ) -> None:
        log_lik_x = np.array(
            [h.log_prob(context, counterpart_action) for h in self._hypotheses]
        )

        # Within-run posterior-predictive probability of this round's
        # observation, for every currently live run -- used both to grow
        # each run and to pool evidence for a fresh changepoint.
        per_run_type_posterior = self._softmax_rows(self._run_log_likelihoods)
        log_pred = _logsumexp(
            np.log(per_run_type_posterior) + log_lik_x[None, :], axis=1
        )

        log_grow = self._run_log_probs + log_pred + np.log1p(-self._hazard)
        log_changepoint = _logsumexp(self._run_log_probs + log_pred + np.log(self._hazard))

        new_run_log_probs = np.concatenate([[log_changepoint], log_grow])
        new_run_log_probs -= _logsumexp(new_run_log_probs)

        grown_log_likelihoods = self._run_log_likelihoods + log_lik_x[None, :]
        new_run_log_likelihoods = np.vstack([log_lik_x[None, :], grown_log_likelihoods])

        self._run_log_probs = new_run_log_probs
        self._run_log_likelihoods = new_run_log_likelihoods
