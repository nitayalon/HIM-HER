"""Grid search over HIM's calibration parameters (tau, delta), evaluated
on all five conditions:
  - stationary: nothing should ever be accepted; counts wasted revisions
    and the resulting cost to cumulative return.
  - within_library_switch, outside_library_switch_m1/m2, recurring:
    final-window accuracy (L2 distance of the held model to the true
    target over the last 50 rounds), not exact parameter equality --
    Revise's grid search can approach the true model in small steps that
    each clear tau without ever landing on it exactly (a genuine,
    reportable property of the cascade, not a tuning artifact: the
    *remaining* improvement margin shrinks as the incumbent gets closer
    to the truth, so eventually no candidate clears a fixed tau).

kappa is held fixed at the value already chosen by run_experiments.py's
kappa sweep -- this script is about HIM's acceptance strictness, not
detection sensitivity.

Usage: python scripts/tune_him.py [--seeds N] [--out DIR] [--kappa K]
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from himher.core.cascade import HHCascade
from himher.core.him import HIMVerifier
from himher.core.library import ModelLibrary
from himher.domains.handoff.construct import construct_bernoulli_model
from himher.domains.handoff.policy import TabularQPolicy
from himher.domains.handoff.revise import revise_mixture_model
from himher.baselines.hh_agent import HHAgent
from himher.experiment import conditions as cond
from himher.experiment import metrics as m
from himher.experiment.runner import run_episode

CONTEXTS = (0, 1)
DEFAULT_KAPPA = 0.6705027574263254
WINDOW = 20
MAX_LEARNED = 20  # see ModelLibrary's docstring / run_experiments.py
ALPHA = 1.0
LEARNING_RATE = 0.3
EPSILON = 0.15  # see run_experiments.py's EPSILON comment

TAU_GRID = [0.0, 0.5, 1.0, 2.0, 3.0]
DELTA_GRID = [0.05, 0.1]
LATE_WINDOW = (250, 300)


def make_agent(tau: float, delta: float, kappa: float, seed: int) -> HHAgent:
    cascade = HHCascade(
        incumbent=cond.LIBRARY_0[0],
        library=ModelLibrary(cond.LIBRARY_0, max_learned=MAX_LEARNED),
        kappa=kappa,
        window=WINDOW,
        verifier=HIMVerifier(tau=tau, delta=delta, contexts=CONTEXTS),
        revise=revise_mixture_model,
        construct=lambda w: construct_bernoulli_model(w, ALPHA),
    )
    policy = TabularQPolicy(learning_rate=LEARNING_RATE, epsilon=EPSILON)
    return HHAgent(cascade, policy, np.random.default_rng(seed), replay=True)


def _late_l2(agent: HHAgent, condition: cond.Condition) -> float:
    true_theta = np.array(
        [condition.schedule.policy_at(t).theta for t in range(condition.episode_length)]
    )
    held_theta = np.array([s.incumbent_after.theta for s in agent.step_logs])
    distance = np.linalg.norm(held_theta - true_theta, axis=1)
    return float(distance[LATE_WINDOW[0] : LATE_WINDOW[1]].mean())


def _switch_specs() -> list[tuple[str, cond.Condition, int]]:
    return [
        ("within_library_switch", cond.within_library_switch(), 150),
        ("outside_library_switch_m1", cond.outside_library_switch(cond.M_NEW_1), 150),
        ("outside_library_switch_m2", cond.outside_library_switch(cond.M_NEW_2), 150),
        ("recurring", cond.recurring(cond.M_NEW_2), 75),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--kappa", type=float, default=DEFAULT_KAPPA)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stationary = cond.stationary()
    switch_specs = _switch_specs()

    records = []
    for tau, delta in itertools.product(TAU_GRID, DELTA_GRID):
        for seed in range(args.seeds):
            agent = make_agent(tau, delta, args.kappa, seed)
            logs = run_episode(agent, stationary, seed)
            economy = m.computational_economy(agent.step_logs)
            records.append(
                {
                    "tau": tau,
                    "delta": delta,
                    "seed": seed,
                    "condition": "stationary",
                    "cumulative_return": m.cumulative_return(logs),
                    "revisions": economy["revisions"],
                    "late_l2": _late_l2(agent, stationary),
                }
            )

            for name, condition, switch_round in switch_specs:
                agent = make_agent(tau, delta, args.kappa, seed)
                logs = run_episode(agent, condition, seed)
                records.append(
                    {
                        "tau": tau,
                        "delta": delta,
                        "seed": seed,
                        "condition": name,
                        "cumulative_return": m.cumulative_return(logs),
                        "detection_delay": m.detection_delay(agent.step_logs, switch_round),
                        "late_l2": _late_l2(agent, condition),
                    }
                )

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "tuning_raw.csv", index=False)

    value_columns = ["cumulative_return", "revisions", "detection_delay", "late_l2"]
    summary = (
        df.groupby(["tau", "delta", "condition"])[value_columns]
        .mean()
        .reset_index()
        .sort_values(["condition", "tau", "delta"])
    )
    summary.to_csv(out_dir / "tuning_summary.csv", index=False)

    wide = summary.pivot_table(index=["tau", "delta"], columns="condition", values="late_l2")
    wide = wide.join(
        df[df.condition == "stationary"].groupby(["tau", "delta"])[["revisions"]].mean()
    )
    wide = wide.rename(columns={"revisions": "stationary_revisions"})
    print(wide.round(3).to_string())


if __name__ == "__main__":
    main()
