"""Logs, for every round of every (condition, method, seed), the
estimated counterpart model: H+H's and Continuous Bayes's point estimate
(theta0, theta1), or Fixed-/Oracle-Bayes's full posterior over their
hypothesis set. At the production calibration
(scripts/run_experiments.py).

Writes two tables, since the two method families report fundamentally
different objects (a point estimate vs. a distribution), not one padded
into the other:

  - results/model_trace_raw.csv: one row per (condition, method, seed, t)
    for hh_replay, hh_ablation, continuous_bayes. Columns: theta0, theta1
    (the point estimate in force that round) and true_theta0, true_theta1
    (the actual counterpart that round, for reference).

  - results/belief_trace_raw.csv: one row per (condition, method, seed, t,
    hypothesis_index) for fixed_bayes, oracle_bayes. Columns:
    hypothesis_theta0, hypothesis_theta1, posterior (that hypothesis's
    posterior probability that round), plus true_theta0/true_theta1.

Both are re-simulations (not reading from the existing run), since the
main pipeline never persisted per-round state -- only the final scalar
metrics derived from it.

Usage: python scripts/compute_model_traces.py [--seeds N] [--out DIR]
"""

from __future__ import annotations

import argparse
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
from himher.domains.handoff.env import SwitchingHandoffGame
from himher.domains.handoff.policy import TabularQPolicy
from himher.domains.handoff.revise import revise_mixture_model
from himher.experiment import conditions as cond

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

POINT_ESTIMATE_METHODS = {"hh_replay", "hh_ablation", "continuous_bayes"}
BELIEF_METHODS = {"fixed_bayes", "oracle_bayes"}


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


def _condition_specs() -> list[tuple[str, cond.Condition]]:
    return [
        ("stationary", cond.stationary()),
        ("within_library_switch", cond.within_library_switch()),
        ("outside_library_switch_m1", cond.outside_library_switch(cond.M_NEW_1)),
        ("outside_library_switch_m2", cond.outside_library_switch(cond.M_NEW_2)),
        ("recurring", cond.recurring(cond.M_NEW_2)),
    ]


def _current_point_estimate(method_name: str, agent: Agent) -> BernoulliHandModel:
    if method_name in ("hh_replay", "hh_ablation"):
        return agent.cascade.incumbent
    return agent.current_model()  # continuous_bayes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_rows = []
    belief_rows = []

    for label, condition in _condition_specs():
        true_model_for_oracle = condition.true_models_in_order[-1]
        methods = make_methods(true_model_for_oracle)

        for method_name, factory in methods.items():
            for seed in range(args.seeds):
                agent = factory(seed)
                rng = np.random.default_rng(seed)
                game = SwitchingHandoffGame(condition.schedule, rng)

                for t in range(condition.episode_length):
                    context = game.sample_context()
                    ego_action = agent.act(context)
                    result = game.step(context, ego_action)
                    agent.update(context, ego_action, result.counterpart_action, result.reward)

                    true_model = condition.schedule.policy_at(t)

                    if method_name in POINT_ESTIMATE_METHODS:
                        estimate = _current_point_estimate(method_name, agent)
                        model_rows.append(
                            {
                                "condition": label,
                                "method": method_name,
                                "seed": seed,
                                "t": t,
                                "theta0": estimate.theta0,
                                "theta1": estimate.theta1,
                                "true_theta0": true_model.theta0,
                                "true_theta1": true_model.theta1,
                            }
                        )
                    else:  # fixed_bayes, oracle_bayes
                        posterior = agent.posterior
                        for h_idx, (prob, hypothesis) in enumerate(zip(posterior, agent.hypotheses)):
                            belief_rows.append(
                                {
                                    "condition": label,
                                    "method": method_name,
                                    "seed": seed,
                                    "t": t,
                                    "hypothesis_index": h_idx,
                                    "hypothesis_theta0": hypothesis.theta0,
                                    "hypothesis_theta1": hypothesis.theta1,
                                    "posterior": prob,
                                    "true_theta0": true_model.theta0,
                                    "true_theta1": true_model.theta1,
                                }
                            )

        print(f"[{label}] done")

    model_df = pd.DataFrame(model_rows)
    model_df.to_csv(out_dir / "model_trace_raw.csv", index=False)

    belief_df = pd.DataFrame(belief_rows)
    belief_df.to_csv(out_dir / "belief_trace_raw.csv", index=False)

    print(
        f"Wrote model_trace_raw.csv ({len(model_df)} rows) and "
        f"belief_trace_raw.csv ({len(belief_df)} rows) to {out_dir}/"
    )


if __name__ == "__main__":
    main()
