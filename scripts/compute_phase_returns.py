"""Computes mean per-round return before and after the first switch in
each condition, for every method, at the production calibration
(scripts/run_experiments.py). Writes results/phase_return_raw.csv and
results/phase_return_summary.csv.

Separate from run_experiments.py because this is a lighter-weight
re-simulation (reward trace only, none of the detection/economy
bookkeeping) added after the fact to support a before/after-change
return plot; re-running the full pipeline for one new measure was not
warranted.

Usage: python scripts/compute_phase_returns.py [--seeds N] [--out DIR]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from himher.baselines.agent_base import Agent
from himher.baselines.continuous_bayes import ContinuousBayesAgent
from himher.baselines.fixed_library_bayes import FixedLibraryBayesAgent
from himher.baselines.hh_agent import HHAgent
from himher.baselines.oracle_bayes import make_oracle_bayes_agent
from himher.core.cascade import HHCascade
from himher.core.him import HIMVerifier
from himher.core.library import ModelLibrary
from himher.domains.handoff.construct import construct_bernoulli_model
from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.policy import TabularQPolicy
from himher.domains.handoff.revise import revise_mixture_model
from himher.experiment import conditions as cond
from himher.experiment import metrics as m
from himher.experiment.aggregate import summarize
from himher.experiment.runner import run_episode

# Mirrors scripts/run_experiments.py's calibrated defaults exactly.
CONTEXTS = (0, 1)
WINDOW = 20
MAX_LEARNED = 20
TAU = 1.0
DELTA = 0.1
ALPHA = 1.0
LEARNING_RATE = 0.3
EPSILON = 0.15
KAPPA = 0.6705027574263254


def make_hh_agent(seed: int, replay: bool) -> HHAgent:
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
    return HHAgent(cascade, policy, np.random.default_rng(seed), replay=replay)


def make_methods(true_model: BernoulliHandModel) -> dict[str, Callable[[int], Agent]]:
    return {
        "hh_replay": lambda seed: make_hh_agent(seed, replay=True),
        "hh_ablation": lambda seed: make_hh_agent(seed, replay=False),
        "fixed_bayes": lambda seed: FixedLibraryBayesAgent(cond.LIBRARY_0),
        "oracle_bayes": lambda seed: make_oracle_bayes_agent(cond.LIBRARY_0, true_model),
        "continuous_bayes": lambda seed: ContinuousBayesAgent(alpha=ALPHA),
    }


def _condition_specs() -> list[tuple[str, cond.Condition, int]]:
    """(label, condition, switch_round). Stationary has no switch, so its
    whole episode is treated as the "before" phase (switch_round =
    episode_length) and it is absent from the "after" phase, matching
    how it is already omitted from the paper's adaptation-delay figure.
    """
    episode_length = cond.EPISODE_LENGTH
    return [
        ("stationary", cond.stationary(), episode_length),
        ("within_library_switch", cond.within_library_switch(), episode_length // 2),
        ("outside_library_switch_m1", cond.outside_library_switch(cond.M_NEW_1), episode_length // 2),
        ("outside_library_switch_m2", cond.outside_library_switch(cond.M_NEW_2), episode_length // 2),
        ("recurring", cond.recurring(cond.M_NEW_2), episode_length // 4),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for label, condition, switch_round in _condition_specs():
        true_model_for_oracle = condition.true_models_in_order[-1]
        methods = make_methods(true_model_for_oracle)

        for method_name, factory in methods.items():
            for seed in range(args.seeds):
                agent = factory(seed)
                logs = run_episode(agent, condition, seed)
                before, after = m.phase_average_return(logs, switch_round)
                records.append(
                    {
                        "condition": label,
                        "method": method_name,
                        "seed": seed,
                        "return_before": before,
                        "return_after": after,
                    }
                )

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "phase_return_raw.csv", index=False)

    summary = summarize(df, group_by=["condition", "method"], value_columns=["return_before", "return_after"])
    summary.to_csv(out_dir / "phase_return_summary.csv", index=False)
    print(f"Wrote phase_return_raw.csv and phase_return_summary.csv to {out_dir}/")


if __name__ == "__main__":
    main()
