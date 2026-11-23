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
from himher.baselines.discounted_bayes import DiscountedBayesAgent
from himher.baselines.windowed_bayes import WindowedBayesAgent
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


def make_hh_cascade() -> HHCascade:
    """The paper's H+H cascade (detection, VOC, reuse/merge/construct, HIM)."""
    return HHCascade(
        incumbent=cond.LIBRARY_0[0],
        library=ModelLibrary(cond.LIBRARY_0, max_learned=MAX_LEARNED),
        kappa=KAPPA,
        window=WINDOW,
        verifier=HIMVerifier(tau=TAU, delta=DELTA, contexts=CONTEXTS),
        revise=revise_mixture_model,
        construct=lambda w: construct_bernoulli_model(w, ALPHA),
    )


def make_hh_agent(seed: int, replay: bool, **hh_kwargs: Any) -> HHAgent:
    """Identical to compute_phase_returns.make_hh_agent when hh_kwargs is
    empty; hh_kwargs are the new, default-off HHAgent flags."""
    cascade = make_hh_cascade()
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

# Task B: windowed / discounted Beta-Bayes (same prior and decision rule as CB).
WB_WINDOW = 20  # brief D1: identical to H+H's k
DB_GAMMAS = (0.90, 0.95, 0.98)  # brief D2: report all three
register("WB-k20", "WB-k20", "default",
         lambda seed, spec: WindowedBayesAgent(alpha=ALPHA, window=WB_WINDOW))
for _g in DB_GAMMAS:
    register(f"DB-{_g:.2f}", f"DB-{_g:.2f}", "default",
             lambda seed, spec, g=_g: DiscountedBayesAgent(alpha=ALPHA, gamma=g))
B_METHODS = ("WB-k20",) + tuple(f"DB-{g:.2f}" for g in DB_GAMMAS)

# Task A: replay-scope ablation (HH-R only; "full" is the registered "HH-R").
for _scope in ("him_window", "post_detection"):
    register(f"HH-R@{_scope}", "HH-R", _scope,
             lambda seed, spec, sc=_scope: make_hh_agent(seed, replay=True, replay_scope=sc))
A_METHODS = ("HH-R", "HH-R@him_window", "HH-R@post_detection", "HH-A")

# Task D: epsilon schedule (brief D3: eps0 = 0.15, eps_min = 0.02, h = 50).
EPS0, EPS_MIN, EPS_HALF_LIFE = 0.15, 0.02, 50.0
register("HH-R@decay_reset", "HH-R", "decay_reset",
         lambda seed, spec: make_hh_agent(seed, replay=True, eps_schedule="decay_reset", eps0=EPS0,
                                          eps_min=EPS_MIN, eps_half_life=EPS_HALF_LIFE))
register("HH-R@him_window+decay_reset", "HH-R", "him_window+decay_reset",
         lambda seed, spec: make_hh_agent(seed, replay=True, replay_scope="him_window",
                                          eps_schedule="decay_reset", eps0=EPS0, eps_min=EPS_MIN,
                                          eps_half_life=EPS_HALF_LIFE))
D_METHODS = ("HH-R@decay_reset", "HH-R@him_window+decay_reset")

# BPR+ (Hernandez-Leal et al., AAAI-16 workshop). Exploration matched to
# H+H (author's choice): drift epsilon = H+H's epsilon, learning phase =
# H+H's window k. n = 7 is the paper's value. rho is calibrated on
# Stationary by scripts/rebuttal_bpr_calibrate.py (the paper's 0.8 was
# tuned for the iPD). The belief floor is not given in the paper.
BPR_N = 7
BPR_LEARN_ROUNDS = WINDOW
BPR_BELIEF_FLOOR = 0.01
# Calibrated (rebuttal_bpr_calibrate.py): the largest rho on a 0.01 grid
# whose mean Stationary false alarms (1.805/episode) do not exceed H+H's
# (2.185). Any rho in [0.56, 0.67] gives the same rule here: p takes
# values in {0.1, 0.9} under M_L, so p_hat < rho <=> >= 3 of the last 7
# hands are surprising under every known model.
BPR_BERN_RHO = 0.67


def make_bpr_bern_agent(seed: int, rho: float | None = None, library=None) -> Any:
    """First BPR+ implementation (11/22 a.m.): exact Bernoulli performance
    models, closed-form best responses. Kept for reproducibility as
    "BPR+bern"; superseded by the MDP implementation below."""
    from himher.baselines.bpr_plus import BPRPlusAgent
    return BPRPlusAgent(cond.LIBRARY_0 if library is None else library, np.random.default_rng(seed),
                        rho=BPR_BERN_RHO if rho is None else rho, n=BPR_N, epsilon=EPSILON,
                        learn_rounds=BPR_LEARN_ROUNDS, belief_floor=BPR_BELIEF_FLOOR, alpha=ALPHA)


# BPR+ as in the paper (author's request, 11/22): opponent strategies are
# MDPs (learned by counts), policies by value iteration, Gaussian
# performance models, probability-of-improvement selection
# (himher/baselines/bpr_plus_mdp.py). Same n, epsilon, learning-phase
# length and belief floor as above. rho lives on a density scale here and
# is recalibrated by the same procedure (rebuttal_bpr_calibrate.py --tag mdp).
# Calibrated (rebuttal_bpr_calibrate.py --tag mdp, 0.01 grid on 0.05-2.00):
# largest rho with mean Stationary false alarms (1.98/episode) <= H+H's (2.185).
BPR_RHO = 0.77


def make_bpr_agent(seed: int, rho: float | None = None, library=None) -> Any:
    from himher.baselines.bpr_plus_mdp import BPRPlusMDPAgent
    return BPRPlusMDPAgent(cond.LIBRARY_0 if library is None else library, np.random.default_rng(seed),
                           rho=BPR_RHO if rho is None else rho, n=BPR_N, epsilon=EPSILON,
                           learn_rounds=BPR_LEARN_ROUNDS, belief_floor=BPR_BELIEF_FLOOR)


register("BPR+", "BPR+", "default", lambda seed, spec: make_bpr_agent(seed))
register("BPR+bern", "BPR+bern", "default", lambda seed, spec: make_bpr_bern_agent(seed))

# BOCPD (author's bocd_bayes.py, unchanged): Adams & MacKay recursion over
# the L0 hypotheses, no run-length pruning. hazard = 0.01 is an untuned
# generic value chosen by the author for the cost study (11/22); it was
# not calibrated (it never ran in the Friday/Saturday results).
BOCPD_HAZARD = 0.01


def make_bocpd_agent(seed: int, hypotheses=None) -> Any:
    from himher.baselines.bocd_bayes import BOCDBayesAgent
    return BOCDBayesAgent(cond.LIBRARY_0 if hypotheses is None else hypotheses, hazard=BOCPD_HAZARD)


register("BOCPD", "BOCPD", "default", lambda seed, spec: make_bocpd_agent(seed))

# Monday Task P (11/23): H+H's cascade with WB-k20's closed-form action
# selection (himher/baselines/hh_plan.py). No exploration (as WB-k20), and
# matched-exploration variants with constant epsilon = EPSILON (0.15), using
# the same seeded ego RNG as HH-R so both -eps variants draw identically.
def make_hh_plan_agent(seed: int, epsilon: float = 0.0) -> Any:
    from himher.baselines.hh_plan import HHPlanAgent
    return HHPlanAgent(make_hh_cascade(), np.random.default_rng(seed), epsilon=epsilon)


def make_wb_eps_agent(seed: int) -> Any:
    from himher.baselines.hh_plan import EpsilonGreedy
    return EpsilonGreedy(WindowedBayesAgent(alpha=ALPHA, window=WB_WINDOW), EPSILON, np.random.default_rng(seed))


register("HH-plan", "HH-plan", "default", lambda seed, spec: make_hh_plan_agent(seed))
register("HH-plan-eps", "HH-plan-eps", "default", lambda seed, spec: make_hh_plan_agent(seed, EPSILON))
register("WB-k20-eps", "WB-k20-eps", "default", lambda seed, spec: make_wb_eps_agent(seed))
P_METHODS = ("HH-plan", "HH-plan-eps", "WB-k20-eps")

# Mapping to the method names used in the stored paper results.
PAPER_NAME = {"HH-R": "hh_replay", "HH-A": "hh_ablation", "FB": "fixed_bayes",
              "OB": "oracle_bayes", "CB": "continuous_bayes"}
