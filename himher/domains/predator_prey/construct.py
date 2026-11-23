"""Construct regime for the Predator-Prey game: a grid search over w.

The handoff constructor counts hands per context; that fails here because
positions rarely repeat within a window. Instead every w in
{0, 0.5, ..., 3}^3 is scored on the window and the best is returned. The
result is always an exact grid value, so a model rebuilt from similar data
equals the stored one (needed for Reuse in the Recurring condition).
"""

from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np

from himher.domains.predator_prey.counterpart import WeightedPreyModel, log_likelihoods
from himher.domains.predator_prey.env import Context

WEIGHT_VALUES: tuple[float, ...] = tuple(np.arange(0.0, 3.01, 0.5))
WEIGHT_GRID: np.ndarray = np.array(list(product(WEIGHT_VALUES, repeat=3)))  # (343, 3)


def construct_weight_model(
    window: Sequence[tuple[Context, int]], grid: np.ndarray = WEIGHT_GRID
) -> WeightedPreyModel:
    best = grid[int(np.argmax(log_likelihoods(grid, window)))]
    return WeightedPreyModel(*map(float, best))
