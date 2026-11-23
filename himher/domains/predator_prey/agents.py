"""Predators for the Predator-Prey game: H+H and the Section 5.4 baselines.

Every method acts by planning (finite-horizon value iteration) against the
prey model it currently believes, so methods differ only in how they model
the prey:

- HHPredator: the H+H cascade's incumbent model.
- BayesPredator: a posterior over a fixed hypothesis set; acts on the
  posterior-weighted Q values. Fixed-library Bayes uses {Den A, Den B};
  Oracle Bayes adds the held-out true types.
- ContinuousBayesPredator: the best w on the 343-point Construct grid,
  re-fit on the entire history every round (no detection or HIM gate).
- OraclePredator: plans against the true prey model.

Each predator also reports its belief for the metrics: ``predictive`` (its
predicted prey move probabilities) and ``map_model`` (its most likely model).
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from himher.core.cascade import CascadeStepLog, HHCascade
from himher.core.him import HIMVerifier
from himher.core.library import ModelLibrary
from himher.domains.predator_prey.construct import WEIGHT_GRID, construct_weight_model
from himher.domains.predator_prey.counterpart import (
    DEN_A_MODEL,
    DEN_B_MODEL,
    WeightedPreyModel,
    log_likelihoods,
)
from himher.domains.predator_prey.env import Context
from himher.domains.predator_prey.planner import greedy_action, plan
from himher.domains.predator_prey.revise import revise_weight_mixture

LIBRARY_0 = (DEN_A_MODEL, DEN_B_MODEL)


class HHPredator:
    def __init__(self, cascade: HHCascade, rng: np.random.Generator) -> None:
        self.cascade = cascade
        self._rng = rng
        self.step_logs: list[CascadeStepLog] = []

    def act(self, context: Context, t: int) -> int:
        p, q = context
        return greedy_action(plan(self.cascade.incumbent)[t, p, q], self._rng)

    def observe(self, context: Context, prey_action: int) -> None:
        self.step_logs.append(self.cascade.step(context, prey_action))

    def predictive(self, context: Context) -> np.ndarray:
        return self.cascade.incumbent.predict_proba(context)

    def map_model(self) -> WeightedPreyModel:
        return self.cascade.incumbent


def make_hh_predator(
    rng: np.random.Generator,
    kappa: float,
    tau: float = 1.0,
    delta: float = 0.1,
    window: int = 20,
    max_learned: int = 20,
) -> HHPredator:
    cascade = HHCascade(
        incumbent=DEN_A_MODEL,
        library=ModelLibrary(LIBRARY_0, max_learned=max_learned),
        kappa=kappa,
        window=window,
        verifier=HIMVerifier(tau=tau, delta=delta, contexts=None),
        revise=revise_weight_mixture,
        construct=construct_weight_model,
        merge_includes_incumbent=True,
    )
    return HHPredator(cascade, rng)


class BayesPredator:
    def __init__(self, hypotheses: Sequence[WeightedPreyModel], rng: np.random.Generator) -> None:
        self.hypotheses = tuple(dict.fromkeys(hypotheses))
        self._log_post = np.zeros(len(self.hypotheses))
        self._rng = rng

    @property
    def posterior(self) -> np.ndarray:
        x = np.exp(self._log_post - self._log_post.max())
        return x / x.sum()

    def act(self, context: Context, t: int) -> int:
        p, q = context
        post = self.posterior
        q_row = sum(b * plan(h)[t, p, q] for b, h in zip(post, self.hypotheses))
        return greedy_action(q_row, self._rng)

    def observe(self, context: Context, prey_action: int) -> None:
        self._log_post += np.array([h.log_prob(context, prey_action) for h in self.hypotheses])

    def predictive(self, context: Context) -> np.ndarray:
        return sum(b * h.predict_proba(context) for b, h in zip(self.posterior, self.hypotheses))

    def map_model(self) -> WeightedPreyModel:
        return self.hypotheses[int(np.argmax(self._log_post))]


class ContinuousBayesPredator:
    """Construct on the entire history, every round. The grid log-likelihoods
    are accumulated incrementally, which is equivalent to re-scoring the
    whole history each round."""

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._scores = np.zeros(len(WEIGHT_GRID))
        self._model = construct_weight_model([])

    def act(self, context: Context, t: int) -> int:
        p, q = context
        return greedy_action(plan(self._model)[t, p, q], self._rng)

    def observe(self, context: Context, prey_action: int) -> None:
        self._scores += log_likelihoods(WEIGHT_GRID, [(context, prey_action)])
        self._model = WeightedPreyModel(*map(float, WEIGHT_GRID[int(np.argmax(self._scores))]))

    def predictive(self, context: Context) -> np.ndarray:
        return self._model.predict_proba(context)

    def map_model(self) -> WeightedPreyModel:
        return self._model


class OraclePredator:
    """Knows the true prey model of every game (set by the runner)."""

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self.true_model: WeightedPreyModel | None = None

    def act(self, context: Context, t: int) -> int:
        p, q = context
        return greedy_action(plan(self.true_model)[t, p, q], self._rng)

    def observe(self, context: Context, prey_action: int) -> None:
        pass

    def predictive(self, context: Context) -> np.ndarray:
        return self.true_model.predict_proba(context)

    def map_model(self) -> WeightedPreyModel:
        return self.true_model


PredatorFactory = Callable[[np.random.Generator], object]
