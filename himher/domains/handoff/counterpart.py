"""Counterpart model for the Switching Handoff Game (paper Section 5.1).

m_theta = (theta_0, theta_1), theta_x = P(a^j = R | x_t = x). The same class
plays two roles, as it does in the paper: it is both a belief the ego agent
can hold (via log_prob/predict_proba, satisfying CounterpartModel) and,
when wired into a SwitchSchedule, the ground-truth generator the
environment samples from (via sample).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from himher.domains.handoff.env import Context, Hand

# Clamp predicted probabilities away from the simplex boundary so that
# log_prob stays finite even for library/held-out models with theta exactly
# 0 or 1 (e.g. a model that has only ever observed one action in a context).
_PROBA_EPS = 1e-12


@dataclass(frozen=True)
class BernoulliHandModel:
    """theta0 = P(a^j = R | x = 0), theta1 = P(a^j = R | x = 1)."""

    theta0: float
    theta1: float

    def __post_init__(self) -> None:
        for name, value in (("theta0", self.theta0), ("theta1", self.theta1)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    @property
    def theta(self) -> tuple[float, float]:
        return (self.theta0, self.theta1)

    def _p_right(self, context: Context) -> float:
        return self.theta1 if context else self.theta0

    def predict_proba(self, context: Context) -> np.ndarray:
        p_right = self._p_right(context)
        return np.array([1.0 - p_right, p_right])  # indexed by Hand.L, Hand.R

    def log_prob(self, context: Context, action: Hand) -> float:
        proba = self.predict_proba(context)[int(action)]
        proba = float(np.clip(proba, _PROBA_EPS, 1.0 - _PROBA_EPS))
        return float(np.log(proba))

    def sample(self, context: Context, rng: np.random.Generator) -> Hand:
        return Hand.R if rng.random() < self._p_right(context) else Hand.L
