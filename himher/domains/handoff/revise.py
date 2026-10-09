"""Revise regime (Section 4.4): interpolate between two existing library
models rather than constructing an entirely new one from scratch -- the
middle-cost regime between Reuse and Construct ("Create a... mixture of
two existing types, or revise the closest model").

For the Switching Handoff Game, Section 5.3's m_2 = (0.1, 0.5) is
literally the theta-space midpoint of m_L = (0.1, 0.1) and
m_C = (0.1, 0.9) (p=0.5); m_1 = (0.2, 0.8) does not lie on the segment
between any two of the three library models, so no mixture can represent
it and only Construct can -- this module lets the cascade discover that
distinction empirically rather than it being hardcoded: revise_mixture_model
always returns its best attempt, and HIM accepts or rejects it on the
merits of how well that attempt explains the window.
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np

from himher.core.library import ModelLibrary
from himher.core.model import history_log_likelihood
from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Context, Hand

DEFAULT_P_GRID: tuple[float, ...] = tuple(np.linspace(0.0, 1.0, 21))


def interpolate(a: BernoulliHandModel, b: BernoulliHandModel, p: float) -> BernoulliHandModel:
    """theta-space mixture: (1-p) * a + p * b, componentwise."""
    return BernoulliHandModel(
        theta0=(1 - p) * a.theta0 + p * b.theta0,
        theta1=(1 - p) * a.theta1 + p * b.theta1,
    )


def revise_mixture_model(
    library: ModelLibrary,
    window: Sequence[tuple[Context, Hand]],
    p_grid: Sequence[float] = DEFAULT_P_GRID,
) -> BernoulliHandModel | None:
    """The best-fitting theta-space interpolation between any two distinct
    library models, with the mixing weight p chosen by grid search over
    p_grid to maximize the window's log-likelihood. None if the library
    has fewer than two models to mix.
    """
    models = list(library)
    if len(models) < 2:
        return None

    best_model: BernoulliHandModel | None = None
    best_log_likelihood = -np.inf
    for a, b in combinations(models, 2):
        for p in p_grid:
            candidate = interpolate(a, b, p)
            log_likelihood = history_log_likelihood(candidate, window)
            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_model = candidate
    return best_model
