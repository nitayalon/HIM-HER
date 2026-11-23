"""Prey model for the Predator-Prey game: w = (w_A, w_B, w_pred).

Each of the prey's 4 moves gets three features in {-1, 0, +1}:
  f_A    closer to / no change / further from den A (shortest path, the
         predator's cell counts as a wall)
  f_B    the same for den B
  f_pred further from / no change / closer to the predator's current cell (l1)
and P(move | context) is proportional to exp(w . f).

A move into the predator's cell is treated as blocked for the den features
(f_A = f_B = 0) but is closer to the predator (f_pred = -1).

As in the handoff game, one class is both a belief the ego can hold and the
ground-truth generator the environment samples from.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

from himher.domains.predator_prey.env import (
    DEN_DIST,
    L1,
    N_CELLS,
    N_PREY_ACTIONS,
    NEXT,
    Context,
)

_PROBA_EPS = 1e-12
N_FEATURES = 3


def _build_features() -> np.ndarray:
    """FEATURES[p, q, a, k]: feature k of prey move a, predator at p, prey at q."""
    p = np.arange(N_CELLS)[:, None, None]
    q = np.arange(N_CELLS)[None, :, None]
    a = np.arange(N_PREY_ACTIONS)[None, None, :]
    nq = NEXT[q, a]  # (1, 49, 4)
    into_predator = nq == p  # (49, 49, 4)
    feats = np.zeros((N_CELLS, N_CELLS, N_PREY_ACTIONS, N_FEATURES))
    for k in (0, 1):
        before = DEN_DIST[k][p, q]  # (49, 49, 1)
        after = DEN_DIST[k][p, nq]  # (49, 49, 4)
        with np.errstate(invalid="ignore"):
            f = np.sign(before - after)
        f = np.where(into_predator | ~np.isfinite(f), 0.0, f)
        feats[..., k] = f
    feats[..., 2] = np.sign(L1[nq, p] - L1[q, p])
    return feats


FEATURES = _build_features()


@lru_cache(maxsize=512)
def _proba_table(w: tuple[float, ...]) -> np.ndarray:
    scores = FEATURES @ np.asarray(w, dtype=float)
    scores -= scores.max(axis=-1, keepdims=True)
    e = np.exp(scores)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass(frozen=True)
class WeightedPreyModel:
    w_A: float
    w_B: float
    w_pred: float

    @property
    def w(self) -> tuple[float, float, float]:
        return (self.w_A, self.w_B, self.w_pred)

    def proba_table(self) -> np.ndarray:
        """P[p, q, a] for every context; shape (49, 49, 4)."""
        return _proba_table(self.w)

    def predict_proba(self, context: Context) -> np.ndarray:
        p, q = context
        return self.proba_table()[p, q]

    def log_prob(self, context: Context, action: int) -> float:
        p, q = context
        proba = float(np.clip(self.proba_table()[p, q, action], _PROBA_EPS, 1.0 - _PROBA_EPS))
        return float(np.log(proba))

    def sample(self, context: Context, rng: np.random.Generator) -> int:
        return int(rng.choice(N_PREY_ACTIONS, p=self.predict_proba(context)))

    def __repr__(self) -> str:
        return f"w({self.w_A:g}, {self.w_B:g}, {self.w_pred:g})"


DEN_A_MODEL = WeightedPreyModel(1.0, 0.0, 0.0)
DEN_B_MODEL = WeightedPreyModel(0.0, 1.0, 0.0)
# Evade is half strength: a pure evader that ignores the dens but is noisier
# than the den-runners. At full strength (0, 0, 1) no learned linear response
# could chase it (best-fit greedy play earned 0.52 vs the oracle's 0.69).
EVADE_MODEL = WeightedPreyModel(0.0, 0.0, 0.5)
# Midpoint of Den B and Evade: reachable by Merge, not on Construct's grid.
CAUTIOUS_B_MODEL = WeightedPreyModel(0.0, 0.5, 0.25)


def window_features(window: Sequence[tuple[Context, int]]) -> tuple[np.ndarray, np.ndarray]:
    """(features of every move at each observed context, observed actions):
    shapes (n, 4, 3) and (n,)."""
    ps = np.fromiter((c[0] for c, _ in window), dtype=np.int64, count=len(window))
    qs = np.fromiter((c[1] for c, _ in window), dtype=np.int64, count=len(window))
    actions = np.fromiter((a for _, a in window), dtype=np.int64, count=len(window))
    return FEATURES[ps, qs], actions


def log_likelihoods(weights: np.ndarray, window: Sequence[tuple[Context, int]]) -> np.ndarray:
    """Window log-likelihood under each row of ``weights`` (shape (K, 3)),
    vectorised so Merge and Construct can score many candidates at once."""
    if not window:
        return np.zeros(len(weights))
    feats, actions = window_features(window)
    scores = np.einsum("nak,mk->mna", feats, weights)  # (K, n, 4)
    scores -= scores.max(axis=-1, keepdims=True)
    log_z = np.log(np.exp(scores).sum(axis=-1))
    chosen = scores[:, np.arange(len(actions)), actions]
    log_p = np.clip(chosen - log_z, np.log(_PROBA_EPS), None)
    return log_p.sum(axis=1)
