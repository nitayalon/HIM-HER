"""Merge regime for the Predator-Prey game: the handoff interpolation
(himher.domains.handoff.revise) with theta replaced by w.

Blending w adds the two models' move scores, so a blend weighs both goals
in every round (e.g. the Den B / Evade midpoint is Cautious-B). It is not a
mixture of the two models' move probabilities.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

from himher.domains.predator_prey.counterpart import WeightedPreyModel, log_likelihoods
from himher.domains.predator_prey.env import Context

DEFAULT_P_GRID: tuple[float, ...] = tuple(np.linspace(0.0, 1.0, 21))
ROUND_DECIMALS = 3  # so equal blends are equal by value (library dedup)


def revise_weight_mixture(
    pool: Iterable[WeightedPreyModel],
    window: Sequence[tuple[Context, int]],
    p_grid: Sequence[float] = DEFAULT_P_GRID,
) -> WeightedPreyModel | None:
    """Best-fitting blend (1 - p) * a + p * b over every pair (a, b) in
    ``pool`` and every p in ``p_grid``; None if the pool has < 2 models."""
    models = list(dict.fromkeys(pool))
    if len(models) < 2:
        return None
    ps = np.asarray(p_grid)[:, None]
    candidates = np.concatenate(
        [(1 - ps) * np.asarray(a.w) + ps * np.asarray(b.w) for a, b in combinations(models, 2)]
    )
    candidates = np.round(candidates, ROUND_DECIMALS) + 0.0
    best = candidates[int(np.argmax(log_likelihoods(candidates, window)))]
    return WeightedPreyModel(*map(float, best))
