"""Finite-horizon value iteration for the predator against a known prey model.

Q[t, p, q, a] is the predator's expected return from round t (0-based) of a
game, predator at p, prey at q, taking action a, if the prey follows the
given model for the rest of the game. Solved exactly over all 2,401
positions and T rounds. Used by every method to act on whatever model it
currently believes, and with the true model as the oracle.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from himher.domains.predator_prey.counterpart import WeightedPreyModel, _proba_table
from himher.domains.predator_prey.env import (
    CATCH_REWARD,
    DEN_REWARD,
    DENS,
    L1,
    N_CELLS,
    N_PRED_ACTIONS,
    N_PREY_ACTIONS,
    NEXT,
    STEP_COST,
    T,
)

_P = np.arange(N_CELLS)[:, None, None, None]
_Q = np.arange(N_CELLS)[None, :, None, None]
_NP = NEXT[:, :N_PRED_ACTIONS][:, None, :, None]  # (49, 1, 5, 1)
_NQ = NEXT[:, :N_PREY_ACTIONS][None, :, None, :]  # (1, 49, 1, 4)
_NP_FULL = np.broadcast_to(_NP, (N_CELLS, N_CELLS, N_PRED_ACTIONS, N_PREY_ACTIONS))
_NQ_FULL = np.broadcast_to(_NQ, (N_CELLS, N_CELLS, N_PRED_ACTIONS, N_PREY_ACTIONS))
_CATCH = (_NP_FULL == _NQ_FULL) | ((_NP == _Q) & (_NQ == _P))
_DEN = np.isin(_NQ_FULL, DENS) & ~_CATCH
_CONTINUE = ~(_CATCH | _DEN)
_IMMEDIATE = np.where(_CATCH, CATCH_REWARD, np.where(_DEN, DEN_REWARD, 0.0)) - STEP_COST
_TIMEOUT = 1.0 / (1.0 + L1[_NP_FULL, _NQ_FULL])


def plan_from_table(prey_proba: np.ndarray) -> np.ndarray:
    """Q values, shape (T, 49, 49, 5), against prey probabilities P[p, q, a]."""
    prey = prey_proba[:, :, None, :]  # (49, 49, 1, 4)
    q_values = np.empty((T, N_CELLS, N_CELLS, N_PRED_ACTIONS))
    v_next = None
    for t in range(T - 1, -1, -1):
        future = _TIMEOUT if t == T - 1 else v_next[_NP_FULL, _NQ_FULL]
        outcome = _IMMEDIATE + np.where(_CONTINUE, future, 0.0)
        q_values[t] = (prey * outcome).sum(axis=-1)
        v_next = q_values[t].max(axis=-1)
    return q_values


@lru_cache(maxsize=128)
def _plan_cached(w: tuple[float, ...]) -> np.ndarray:
    return plan_from_table(_proba_table(w))


def plan(model: WeightedPreyModel) -> np.ndarray:
    return _plan_cached(model.w)


def greedy_action(q_row: np.ndarray, rng: np.random.Generator) -> int:
    """argmax with uniform random tie-breaking."""
    best = np.flatnonzero(q_row >= q_row.max() - 1e-12)
    return int(best[0] if len(best) == 1 else rng.choice(best))


def plan_worst_case(dens_are_refuges: bool = True) -> np.ndarray:
    """Q values against a worst-case prey that sees the predator's move and picks
    the reply that is worst for the predator (a model-free pursuit value). With
    ``dens_are_refuges=False`` entering a den is treated as an ordinary move."""
    immediate = _IMMEDIATE if dens_are_refuges else np.where(_CATCH, CATCH_REWARD, 0.0) - STEP_COST
    cont_mask = _CONTINUE if dens_are_refuges else ~_CATCH
    q_values = np.empty((T, N_CELLS, N_CELLS, N_PRED_ACTIONS))
    v_next = None
    for t in range(T - 1, -1, -1):
        future = _TIMEOUT if t == T - 1 else v_next[_NP_FULL, _NQ_FULL]
        outcome = immediate + np.where(cont_mask, future, 0.0)
        q_values[t] = outcome.min(axis=-1)
        v_next = q_values[t].max(axis=-1)
    return q_values
