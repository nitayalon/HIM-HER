"""Runs the Section 5 illustration end to end and writes Section 5.5's
measures to results/, as raw per-seed CSVs, aggregated summary CSVs, a
few plots, and LaTeX tables ready to drop into the paper's Results
section.

Usage:
    python scripts/run_experiments.py [--seeds N] [--out DIR]

Two passes:
  1. A kappa sweep on the Stationary condition (Section 5.3), to choose a
     detection threshold with a low false-alarm rate (Section 5.2: "We
     sweep kappa from permissive to conservative settings").
  2. The four experimental conditions, each run with every method
     (H+H with replay, H+H ablation, and the three Section 5.4
     baselines), at the kappa chosen in pass 1.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from himher.experiment.aggregate import summarize, to_latex_table
from himher.experiment.runner import run_episode

CONTEXTS = (0, 1)

# H+H calibration parameters (phi), fixed across conditions once kappa is
# chosen by the Pass 1 sweep below. window=k=20, alpha=1.0 (Laplace),
# tau=0.0/delta=0.1/ not a tuned final answer (see
# the "library pollution" caveat in the implementation notes).
WINDOW = 20
# A reasonable bound on the library's learned (non-core) entries; not
# load-bearing for correctness now that HIMVerifier.evaluate scopes both
# conditions to the incumbent only (see himher/core/him.py's docstring
# for why that scoping was necessary -- checking against the full
# library created a permanent recovery-blocking gridlock).
MAX_LEARNED = 20
# tau=1.0 chosen by a grid search (tau in {0, 0.5, 1, 2, 3} x delta in
# {0.05, 0.1}) over 40 seeds each, measuring late-window (rounds
# 250-300) L2 accuracy to the true model in all five conditions
# (scripts/tune_him.py) -- not exact parameter equality, since Revise's
# grid search can approach the true model in small steps that each clear
# tau without ever landing on it exactly (the remaining improvement
# margin shrinks as the incumbent nears the truth, so eventually no
# candidate clears a fixed tau; a real, reportable property of the
# cascade, not a tuning artifact). tau=1.0 beat tau=0.5 on every single
# condition, and was the best-balanced choice overall: higher tau
# (2-3) improves accuracy further in Stationary/Within-library but
# actively hurts Recurring (0.277 late L2 at tau=3 vs. 0.216 at tau=1),
# since Recurring's switches are only 75 rounds apart and a stricter bar
# doesn't leave enough time to clear before the next switch.
# delta showed no detectable effect in {0.05, 0.1}, kept at 0.1.
TAU = 1.0
DELTA = 0.1
ALPHA = 1.0
LEARNING_RATE = 0.3
# 0.05 was the dev-time default, but it's too low for this domain: Q is
# indexed by (context, model, action), so exploration budget is already
# split across contexts, and at 0.05 a non-trivial fraction of seeds
# never explore the correct action at some context before locking onto
# the wrong one via the greedy tie-break (confirmed empirically: mean
# stationary return 185 with std 46 and min 86 at epsilon=0.05, vs mean
# 234 with std 15 and min 202 at epsilon=0.15, with zero cascade
# revisions in both cases -- i.e. this is a pure Q-learning convergence
# artifact, not anything to do with detection or HIM).
EPSILON = 0.15

KAPPA_GRID = np.geomspace(-math.log(0.9), -math.log(0.1), 6).tolist()


def make_hh_agent(kappa: float, seed: int, replay: bool) -> HHAgent:
    cascade = HHCascade(
        incumbent=cond.LIBRARY_0[0],
        library=ModelLibrary(cond.LIBRARY_0, max_learned=MAX_LEARNED),
        kappa=kappa,
        window=WINDOW,
        verifier=HIMVerifier(tau=TAU, delta=DELTA, contexts=CONTEXTS),
        revise=revise_mixture_model,
        construct=lambda w: construct_bernoulli_model(w, ALPHA),
    )
    policy = TabularQPolicy(learning_rate=LEARNING_RATE, epsilon=EPSILON)
    return HHAgent(cascade, policy, np.random.default_rng(seed), replay=replay)


def make_methods(kappa: float, true_model: BernoulliHandModel) -> dict[str, Callable[[int], Agent]]:
    return {
        "hh_replay": lambda seed: make_hh_agent(kappa, seed, replay=True),
        "hh_ablation": lambda seed: make_hh_agent(kappa, seed, replay=False),
        "fixed_bayes": lambda seed: FixedLibraryBayesAgent(cond.LIBRARY_0),
        "oracle_bayes": lambda seed: make_oracle_bayes_agent(cond.LIBRARY_0, true_model),
        "continuous_bayes": lambda seed: ContinuousBayesAgent(alpha=ALPHA),
    }


# ---------------------------------------------------------------------
# Pass 1: kappa calibration on the Stationary condition
# ---------------------------------------------------------------------
def run_kappa_sweep(seeds: int, out_dir: Path) -> float:
    condition = cond.stationary()
    records = []
    for kappa in KAPPA_GRID:
        for seed in range(seeds):
            agent = make_hh_agent(kappa, seed, replay=True)
            run_episode(agent, condition, seed)
            records.append(
                {"kappa": kappa, "seed": seed, "false_alarm_rate": m.false_alarm_rate(agent.step_logs)}
            )
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "kappa_sweep_raw.csv", index=False)

    summary = summarize(df, group_by=["kappa"], value_columns=["false_alarm_rate"])
    summary.to_csv(out_dir / "kappa_sweep_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.errorbar(summary["kappa"], summary["false_alarm_rate"], yerr=summary["false_alarm_rate_sem"], marker="o")
    ax.set_xlabel(r"Detection threshold $\kappa$")
    ax.set_ylabel("False-alarm rate (stationary counterpart)")
    ax.set_title("Kappa calibration sweep")
    fig.tight_layout()
    fig.savefig(out_dir / "kappa_sweep.png", dpi=150)
    plt.close(fig)

    target_false_alarm_rate = 0.05
    feasible = summary[summary["false_alarm_rate"] <= target_false_alarm_rate]
    chosen_kappa = feasible["kappa"].min() if not feasible.empty else summary["kappa"].max()
    print(f"[kappa sweep] chosen kappa = {chosen_kappa:.4f} "
          f"(false-alarm rate {summary.loc[summary['kappa'] == chosen_kappa, 'false_alarm_rate'].iloc[0]:.3f})")
    return float(chosen_kappa)


# ---------------------------------------------------------------------
# Pass 2: the four experimental conditions
# ---------------------------------------------------------------------
def _condition_specs() -> list[tuple[str, cond.Condition, int]]:
    """(label, condition, switch_round) for every condition we measure
    adaptation-speed/regret on. switch_round is the first behavioral
    switch in the condition's schedule.
    """
    episode_length = cond.EPISODE_LENGTH
    return [
        ("stationary", cond.stationary(), 0),
        ("within_library_switch", cond.within_library_switch(), episode_length // 2),
        ("outside_library_switch_m1", cond.outside_library_switch(cond.M_NEW_1), episode_length // 2),
        ("outside_library_switch_m2", cond.outside_library_switch(cond.M_NEW_2), episode_length // 2),
        ("recurring", cond.recurring(cond.M_NEW_2), episode_length // 4),
    ]


def run_main_experiments(kappa: float, seeds: int, out_dir: Path) -> pd.DataFrame:
    records = []
    for label, condition, switch_round in _condition_specs():
        true_model_for_oracle = condition.true_models_in_order[-1]
        methods = make_methods(kappa, true_model_for_oracle)

        for method_name, factory in methods.items():
            for seed in range(seeds):
                agent = factory(seed)
                logs = run_episode(agent, condition, seed)

                record = {
                    "condition": label,
                    "method": method_name,
                    "seed": seed,
                    "cumulative_return": m.cumulative_return(logs),
                }
                if label != "stationary":
                    record["post_switch_regret"] = m.post_switch_regret(
                        logs, condition.schedule.policy_at, switch_round
                    )
                    record["adaptation_delay"] = m.adaptation_delay(
                        logs, condition.schedule.policy_at, switch_round
                    )

                step_logs = getattr(agent, "step_logs", None)
                if step_logs is not None:
                    record["false_alarm_rate"] = m.false_alarm_rate(step_logs)
                    if label != "stationary":
                        record["detection_delay"] = m.detection_delay(step_logs, switch_round)
                        record["model_accuracy"] = m.model_accuracy(step_logs, true_model_for_oracle)
                    economy = m.computational_economy(step_logs)
                    record.update({f"economy_{k}": v for k, v in economy.items()})

                policy = getattr(agent, "policy", None)
                if policy is not None:
                    record["policy_updates_total"] = m.policy_updates(policy)["total"]

                records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "main_raw.csv", index=False)
    return df


def add_replay_benefit(df: pd.DataFrame) -> pd.DataFrame:
    """Section 5.5 item 6: per (condition, seed), replay's adaptation-
    delay advantage over the post-detection-only ablation.
    """
    pivot = df[df["method"].isin(["hh_replay", "hh_ablation"])].pivot_table(
        index=["condition", "seed"], columns="method", values="adaptation_delay"
    )
    if pivot.empty or "hh_replay" not in pivot or "hh_ablation" not in pivot:
        return df
    pivot["replay_benefit"] = pivot.apply(
        lambda row: m.replay_benefit(row.get("hh_replay"), row.get("hh_ablation")), axis=1
    )
    benefit = pivot["replay_benefit"].reset_index()
    benefit["method"] = "hh_replay"
    return df.merge(benefit, on=["condition", "seed", "method"], how="left")


# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running with {args.seeds} seeds per condition/method; output -> {out_dir}/")

    kappa = run_kappa_sweep(args.seeds, out_dir)
    df = run_main_experiments(kappa, args.seeds, out_dir)
    df = add_replay_benefit(df)
    df.to_csv(out_dir / "main_raw.csv", index=False)

    value_columns = [
        c
        for c in df.columns
        if c not in ("condition", "method", "seed") and pd.api.types.is_numeric_dtype(df[c])
    ]
    summary = summarize(df, group_by=["condition", "method"], value_columns=value_columns)
    summary.to_csv(out_dir / "main_summary.csv", index=False)

    tables = {
        "cumulative_return": "Cumulative return by condition and method.",
        "post_switch_regret": "Post-switch regret by condition and method.",
        "adaptation_delay": "Rounds to recover 90\\% of steady-state return after a switch.",
        "false_alarm_rate": "False-alarm rate (H+H methods only).",
        "model_accuracy": r"$\lVert\hat\theta-\theta_{\mathrm{new}}\rVert$ for the first accepted construction.",
        "replay_benefit": "Replay benefit: ablation delay minus replay delay (positive favors replay).",
    }
    latex_chunks = []
    for value_column, caption in tables.items():
        if value_column not in summary.columns:
            continue
        latex_chunks.append(
            to_latex_table(
                summary,
                row_key="condition",
                column_key="method",
                value_column=value_column,
                caption=caption,
                label=f"tab:{value_column}",
            )
        )
    (out_dir / "tables.tex").write_text("\n\n".join(latex_chunks), encoding="utf-8")

    print(f"Done. kappa={kappa:.4f}. Wrote raw/summary CSVs, plots, and tables.tex under {out_dir}/")


if __name__ == "__main__":
    main()
