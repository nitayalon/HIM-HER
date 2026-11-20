"""Method registry for the rebuttal runs. Every constant mirrors
scripts/compute_phase_returns.py / scripts/run_experiments.py exactly (the
paper's frozen calibration, brief D6); nothing here is tuned.

Keys are "<method>" or "<method>@<variant>"; the summary rows carry the
method and variant separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from himher.baselines.continuous_bayes import ContinuousBayesAgent
from himher.baselines.fixed_library_bayes import FixedLibraryBayesAgent
from himher.baselines.hh_agent import HHAgent
from himher.baselines.oracle_bayes import make_oracle_bayes_agent
from himher.core.cascade import HHCascade
from himher.core.him import HIMVerifier
from himher.core.library import ModelLibrary
from himher.domains.handoff.construct import construct_bernoulli_model
from himher.domains.handoff.policy import TabularQPolicy
from himher.domains.handoff.revise import revise_mixture_model
from himher.experiment import conditions as cond
from himher.experiment.instrumentation import ConditionSpec

CONTEXTS = (0, 1)
WINDOW = 20
MAX_LEARNED = 20
TAU = 1.0
DELTA = 0.1
ALPHA = 1.0
LEARNING_RATE = 0.3
EPSILON = 0.15
KAPPA = 0.6705027574263254


def make_hh_agent(seed: int, replay: bool, **hh_kwargs: Any) -> HHAgent:
    """Identical to compute_phase_returns.make_hh_agent when hh_kwargs is
    empty; hh_kwargs are the new, default-off HHAgent flags."""
    cascade = HHCascade(
        incumbent=cond.LIBRARY_0[0],
        library=ModelLibrary(cond.LIBRARY_0, max_learned=MAX_LEARNED),
        kappa=KAPPA,
        window=WINDOW,
        verifier=HIMVerifier(tau=TAU, delta=DELTA, contexts=CONTEXTS),
        revise=revise_mixture_model,
        construct=lambda w: construct_bernoulli_model(w, ALPHA),
    )
    policy = TabularQPolicy(learning_rate=LEARNING_RATE, epsilon=EPSILON)
    return HHAgent(cascade, policy, np.random.default_rng(seed), replay=replay, **hh_kwargs)


@dataclass(frozen=True)
class MethodSpec:
    method: str
    variant: str
    factory: Callable[[int, ConditionSpec], Any]


METHODS: dict[str, MethodSpec] = {}


def register(key: str, method: str, variant: str, factory: Callable[[int, ConditionSpec], Any]) -> None:
    METHODS[key] = MethodSpec(method, variant, factory)


register("HH-R", "HH-R", "default", lambda seed, spec: make_hh_agent(seed, replay=True))
register("HH-A", "HH-A", "default", lambda seed, spec: make_hh_agent(seed, replay=False))
register("FB", "FB", "default", lambda seed, spec: FixedLibraryBayesAgent(cond.LIBRARY_0))
register("OB", "OB", "default", lambda seed, spec: make_oracle_bayes_agent(
    cond.LIBRARY_0, spec.condition.true_models_in_order[-1]))
register("CB", "CB", "default", lambda seed, spec: ContinuousBayesAgent(alpha=ALPHA))

ORIGINAL_METHODS = ("HH-R", "HH-A", "FB", "OB", "CB")

# Mapping to the method names used in the stored paper results.
PAPER_NAME = {"HH-R": "hh_replay", "HH-A": "hh_ablation", "FB": "fixed_bayes",
              "OB": "oracle_bayes", "CB": "continuous_bayes"}
