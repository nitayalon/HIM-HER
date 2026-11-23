"""Predator-Prey game (see REPORT.md).

A 7x7 grid with dens at the top-left (A) and bottom-right (B). Each round
the predator (ego) and prey (counterpart) move at the same time. A game
lasts up to ``T`` rounds and ends on a catch (same cell or swapped cells),
the prey entering a den, or timeout. Cells are integers ``r * SIZE + c``.

The context the ego sees each round is ``(predator_cell, prey_cell)``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

SIZE = 7
N_CELLS = SIZE * SIZE
T = 20
MIN_START_DISTANCE = 3
STEP_COST = 0.01
CATCH_REWARD = 1.0
DEN_REWARD = 0.0

DEN_A = 0  # (0, 0)
DEN_B = N_CELLS - 1  # (6, 6)
DENS = (DEN_A, DEN_B)

# Prey actions are indices 0-3; the predator additionally has 4 = stay.
MOVES = ((-1, 0), (1, 0), (0, 1), (0, -1), (0, 0))  # N, S, E, W, stay
MOVE_NAMES = ("N", "S", "E", "W", "stay")
N_PREY_ACTIONS = 4
N_PRED_ACTIONS = 5

Context = tuple[int, int]  # (predator_cell, prey_cell)


def cell(r: int, c: int) -> int:
    return r * SIZE + c


def coords(x: int) -> tuple[int, int]:
    return divmod(x, SIZE)


def _build_next() -> np.ndarray:
    """NEXT[cell, action] -> cell after the move; a wall leaves it in place."""
    nxt = np.empty((N_CELLS, len(MOVES)), dtype=np.int64)
    for x in range(N_CELLS):
        r, c = coords(x)
        for a, (dr, dc) in enumerate(MOVES):
            nr, nc = r + dr, c + dc
            nxt[x, a] = cell(nr, nc) if 0 <= nr < SIZE and 0 <= nc < SIZE else x
    return nxt


def _build_l1() -> np.ndarray:
    rc = np.array([coords(x) for x in range(N_CELLS)])
    return np.abs(rc[:, None, :] - rc[None, :, :]).sum(-1)


NEXT = _build_next()
L1 = _build_l1()


def _bfs_from(target: int, blocked: int | None) -> np.ndarray:
    dist = np.full(N_CELLS, np.inf)
    dist[target] = 0
    queue = deque([target])
    while queue:
        x = queue.popleft()
        for a in range(N_PREY_ACTIONS):
            y = NEXT[x, a]
            if y == blocked or np.isfinite(dist[y]):
                continue
            dist[y] = dist[x] + 1
            queue.append(y)
    return dist


def _build_den_distance() -> np.ndarray:
    """DEN_DIST[den_index, predator_cell, cell]: shortest-path distance from
    ``cell`` to the den, with the predator's cell counted as a wall (unless
    the predator stands on the den itself, where no wall is applied)."""
    out = np.empty((len(DENS), N_CELLS, N_CELLS))
    for i, den in enumerate(DENS):
        for p in range(N_CELLS):
            out[i, p] = _bfs_from(den, None if p == den else p)
    return out


DEN_DIST = _build_den_distance()


@dataclass(frozen=True)
class RoundResult:
    next_context: Context
    reward: float
    done: bool
    outcome: str | None  # "catch", "den", "timeout", or None if the game continues


def resolve_round(context: Context, pred_action: int, prey_action: int, t: int) -> RoundResult:
    """Apply one simultaneous move at round ``t`` (0-based) of a game."""
    p, q = context
    np_, nq = int(NEXT[p, pred_action]), int(NEXT[q, prey_action])
    caught = np_ == nq or (np_ == q and nq == p)
    if caught:
        return RoundResult((np_, nq), CATCH_REWARD - STEP_COST, True, "catch")
    if nq in DENS:
        return RoundResult((np_, nq), DEN_REWARD - STEP_COST, True, "den")
    if t == T - 1:
        return RoundResult((np_, nq), 1.0 / (1.0 + L1[np_, nq]) - STEP_COST, True, "timeout")
    return RoundResult((np_, nq), -STEP_COST, False, None)


CENTRE = tuple(cell(r, c) for r in (2, 3, 4) for c in (2, 3, 4))


def sample_start(rng: np.random.Generator) -> Context:
    """Prey uniformly in the central 3x3; predator uniformly among cells at
    least MIN_START_DISTANCE (l1) from the prey and not on a den."""
    q = int(rng.choice(CENTRE))
    options = [x for x in range(N_CELLS) if L1[x, q] >= MIN_START_DISTANCE and x not in DENS]
    p = int(rng.choice(options))
    return (p, q)
