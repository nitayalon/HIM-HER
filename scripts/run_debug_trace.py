"""Produces a human-inspectable, round-by-round trace of H+H's behavior
on one seeded episode, to check "is the framework operating as planned."

For each condition (default: outside_library_switch_m1 and
outside_library_switch_m2, the two cases the Revise regime is meant to
distinguish), writes three
CSVs under <out>/<condition>/:

  - rounds.csv:      one row per round -- context/actions/reward,
                      incumbent before/after, detection statistic,
                      accepted regime, replay bookkeeping, and the ego
                      policy's Q(L)/Q(R) before and after the round.
  - attempts.csv:     one row per *attempted* candidate model that round,
                      accepted or not -- this is the explicit record of
                      rejected/failed models (regime, theta, HIM's score,
                      improvement-over-incumbent, distinctness-from-
                      incumbent) for evaluating why a candidate failed,
                      not just whether one was accepted.
  - likelihoods.csv:  one row per (round, reference model) -- average
                      per-step log-likelihood of the current verification
                      window under each of the three library models, the
                      known true target (ground truth, for debugging
                      only -- the agent itself never sees this label),
                      and the current incumbent, so "which model best
                      explains recent behavior" is directly inspectable
                      over time.

Also runs the Fixed-library Bayes baseline on the same episode and writes
<out>/<condition>/fixed_bayes_posterior.csv (one row per round: posterior
over each of the 3 library hypotheses, chosen action, reward), to make
its failure mode explicit on Outside library switch (m2): since m_2 is in
none of its hypotheses, its posterior can only ever concentrate on the
best *available* (but wrong) one.

Usage: python scripts/run_debug_trace.py [--seed N] [--out DIR]
                                          [--conditions outside_library_switch_m1,outside_library_switch_m2,...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from himher.baselines.fixed_library_bayes import FixedLibraryBayesAgent
from himher.baselines.hh_agent import HHAgent
from himher.core.cascade import HHCascade
from himher.core.diagnostics import likelihood_table
from himher.core.him import HIMVerifier
from himher.core.library import ModelLibrary
from himher.domains.handoff.construct import construct_bernoulli_model
from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Hand, SwitchingHandoffGame
from himher.domains.handoff.policy import TabularQPolicy
from himher.domains.handoff.revise import revise_mixture_model
from himher.experiment import conditions as cond

CONTEXTS = (0, 1)

# Mirrors scripts/run_experiments.py's calibrated defaults; see that
# file's comments for how each value was chosen.
WINDOW = 20
MAX_LEARNED = 20
TAU = 1.0
DELTA = 0.1
ALPHA = 1.0
LEARNING_RATE = 0.3
EPSILON = 0.15
KAPPA = 0.6705027574263254

REFERENCE_MODELS = {
    "m_L": cond.M_L,
    "m_R": cond.M_R,
    "m_C": cond.M_C,
}


def _make_hh_agent(seed: int) -> HHAgent:
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
    return HHAgent(cascade, policy, np.random.default_rng(seed), replay=True)


def trace_hh_episode(
    condition: cond.Condition, seed: int, true_target: BernoulliHandModel
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    agent = _make_hh_agent(seed)
    rng = np.random.default_rng(seed)
    game = SwitchingHandoffGame(condition.schedule, rng)

    reference_models = {**REFERENCE_MODELS, "m_true_target": true_target}

    round_rows, attempt_rows, likelihood_rows = [], [], []

    for t in range(condition.episode_length):
        context = game.sample_context()
        incumbent_before = agent.cascade.incumbent
        q_before_l = agent.policy.q_value(context, incumbent_before, Hand.L)
        q_before_r = agent.policy.q_value(context, incumbent_before, Hand.R)
        buffer_size_before = len(agent.buffer)

        ego_action = agent.act(context)
        result = game.step(context, ego_action)
        agent.update(context, ego_action, result.counterpart_action, result.reward)

        step_log = agent.last_step_log
        incumbent_after = agent.cascade.incumbent
        replayed = step_log.revised and agent.replay_enabled

        round_rows.append(
            {
                "t": t,
                "context": context,
                "ego_action": ego_action.name,
                "counterpart_action": result.counterpart_action.name,
                "reward": result.reward,
                "incumbent_before_theta0": incumbent_before.theta0,
                "incumbent_before_theta1": incumbent_before.theta1,
                "incumbent_after_theta0": incumbent_after.theta0,
                "incumbent_after_theta1": incumbent_after.theta1,
                "true_target_theta0": true_target.theta0,
                "true_target_theta1": true_target.theta1,
                "detected": step_log.detected,
                "statistic": step_log.statistic,
                "kappa": step_log.kappa,
                "revised": step_log.revised,
                "regime": step_log.regime.value if step_log.regime else None,
                "replayed": replayed,
                "buffer_size_before": buffer_size_before,
                "buffer_size_after": len(agent.buffer),
                "q_before_L": q_before_l,
                "q_before_R": q_before_r,
                "q_after_L": agent.policy.q_value(context, incumbent_after, Hand.L),
                "q_after_R": agent.policy.q_value(context, incumbent_after, Hand.R),
            }
        )

        # Every attempted candidate this round, accepted or not -- the
        # explicit record of rejected/failed models and why (HIM's
        # components), not just the winner.
        for i, attempt in enumerate(step_log.attempts):
            attempt_rows.append(
                {
                    "t": t,
                    "attempt_index": i,
                    "regime": attempt.regime.value,
                    "theta0": attempt.model.theta0,
                    "theta1": attempt.model.theta1,
                    "accepted": attempt.result.accepted,
                    "score": attempt.result.score,
                    "min_improvement": attempt.result.min_improvement,
                    "min_distinctness": attempt.result.min_distinctness,
                    "tau": TAU,
                    "delta": DELTA,
                }
            )

        window = agent.cascade.history.as_tuple()
        for name, avg_ll in likelihood_table({**reference_models, "incumbent": incumbent_after}, window).items():
            likelihood_rows.append({"t": t, "model_name": name, "avg_log_likelihood": avg_ll})

    # Per-model Q-update attention (Section 5.5 measure 5's "...and policy
    # updates"): how many times each distinct model's Q-entries were
    # touched, online or via replay, over the whole episode.
    policy_update_rows = [
        {"theta0": model.theta0, "theta1": model.theta1, "update_count": count}
        for model, count in agent.policy.update_counts.items()
    ]

    return (
        pd.DataFrame(round_rows),
        pd.DataFrame(attempt_rows),
        pd.DataFrame(likelihood_rows),
        pd.DataFrame(policy_update_rows),
    )


def trace_fixed_bayes_episode(condition: cond.Condition, seed: int) -> pd.DataFrame:
    agent = FixedLibraryBayesAgent(cond.LIBRARY_0)
    rng = np.random.default_rng(seed)
    game = SwitchingHandoffGame(condition.schedule, rng)

    rows = []
    for t in range(condition.episode_length):
        context = game.sample_context()
        ego_action = agent.act(context)
        result = game.step(context, ego_action)
        agent.update(context, ego_action, result.counterpart_action, result.reward)

        posterior = agent.posterior
        rows.append(
            {
                "t": t,
                "context": context,
                "ego_action": ego_action.name,
                "counterpart_action": result.counterpart_action.name,
                "reward": result.reward,
                "posterior_m_L": posterior[0],
                "posterior_m_R": posterior[1],
                "posterior_m_C": posterior[2],
            }
        )
    return pd.DataFrame(rows)


def plot_likelihoods(likelihoods: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {
        "m_L": "#eb6834",
        "m_R": "#1baf7a",
        "m_C": "#eda100",
        "m_true_target": "#2a78d6",
        "incumbent": "#0b0b0b",
    }
    for name, group in likelihoods.groupby("model_name"):
        style = "-" if name != "incumbent" else "--"
        ax.plot(group["t"], group["avg_log_likelihood"], style, label=name, color=colors.get(name), linewidth=1.4)
    ax.set_xlabel("round")
    ax.set_ylabel("avg. per-step log-likelihood of recent window")
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(color="#e1e0d9", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results/debug_trace")
    parser.add_argument(
        "--conditions",
        type=str,
        default="outside_library_switch_m1,outside_library_switch_m2",
        help=(
            "comma-separated: outside_library_switch_m1, outside_library_switch_m2, "
            "within_library_switch, recurring, stationary"
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    specs = {
        "outside_library_switch_m1": (cond.outside_library_switch(cond.M_NEW_1), cond.M_NEW_1),
        "outside_library_switch_m2": (cond.outside_library_switch(cond.M_NEW_2), cond.M_NEW_2),
        "within_library_switch": (cond.within_library_switch(), cond.M_R),
        "recurring": (cond.recurring(cond.M_NEW_2), cond.M_NEW_2),
        "stationary": (cond.stationary(), cond.M_L),
    }

    for name in args.conditions.split(","):
        name = name.strip()
        condition, true_target = specs[name]
        condition_dir = out_dir / name
        condition_dir.mkdir(parents=True, exist_ok=True)

        rounds, attempts, likelihoods, policy_updates = trace_hh_episode(condition, args.seed, true_target)
        rounds.to_csv(condition_dir / "rounds.csv", index=False)
        attempts.to_csv(condition_dir / "attempts.csv", index=False)
        likelihoods.to_csv(condition_dir / "likelihoods.csv", index=False)
        policy_updates.to_csv(condition_dir / "policy_updates.csv", index=False)
        plot_likelihoods(likelihoods, condition_dir / "likelihoods.png", f"{name} (seed={args.seed})")

        bayes = trace_fixed_bayes_episode(condition, args.seed)
        bayes.to_csv(condition_dir / "fixed_bayes_posterior.csv", index=False)

        n_revisions = int(rounds["revised"].sum())
        regimes_used = sorted(rounds.loc[rounds["revised"], "regime"].unique().tolist())
        n_rejected_attempts = int((~attempts["accepted"]).sum())
        final_incumbent = (rounds.iloc[-1]["incumbent_after_theta0"], rounds.iloc[-1]["incumbent_after_theta1"])
        print(
            f"[{name}] true_target={true_target.theta} final_incumbent={final_incumbent} "
            f"revisions={n_revisions} regimes_used={regimes_used} rejected_attempts={n_rejected_attempts} "
            f"-> {condition_dir}/"
        )


if __name__ == "__main__":
    main()
