"""Model construction for the Switching Handoff Game (Section 5.2): a
Beta(alpha, alpha) posterior-mean estimate of theta_x from the counts
observed in a window of (context, action) pairs.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Context, Hand


def construct_bernoulli_model(
    window: Sequence[tuple[Context, Hand]], alpha: float
) -> BernoulliHandModel:
    """hat_theta_x = (n_{x,R} + alpha) / (n_x + 2*alpha) (Section 5.2).

    If a context never appears in the window, its count is zero and the
    estimate falls back to the prior mean (0.5 for a symmetric
    Beta(alpha, alpha)), since (0 + alpha) / (0 + 2*alpha) = 0.5
    regardless of alpha.
    """
    right_counts: Counter[Context] = Counter()
    total_counts: Counter[Context] = Counter()
    for context, action in window:
        total_counts[context] += 1
        if action == Hand.R:
            right_counts[context] += 1

    def theta_for(context: Context) -> float:
        n_x = total_counts[context]
        n_x_right = right_counts[context]
        return (n_x_right + alpha) / (n_x + 2 * alpha)

    return BernoulliHandModel(theta0=theta_for(0), theta1=theta_for(1))
