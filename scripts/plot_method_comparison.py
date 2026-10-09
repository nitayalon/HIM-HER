"""Runs H+H and the three Bayesian baselines on the same seeded counterpart
sequence and plots, for each method:

  1. predictive log-likelihood of the observed counterpart action (time x,
     log-likelihood y; rolling mean over 20 rounds), with vertical lines at
     regime changes;
  2. the theta_0 of the model the method holds (time x, theta_0 y), with
     horizontal lines at the true theta_0 of each regime and vertical lines
     at regime changes.

Conventions:
  - loglik at round t: log P(observed action | context) under the model the
    method held BEFORE round t (for the Bayesian methods, the posterior
    predictive; for H+H, the incumbent).
  - theta_0 at round t: the model the method holds AFTER round t (for the
    Bayesian methods, the posterior-mean theta_0).

Usage: python scripts/plot_method_comparison.py [--seed N] [--conditions ...] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from himher.baselines.hh_agent import HHAgent
from himher.core.cascade import HHCascade
from himher.core.him import HIMVerifier
from himher.core.library import ModelLibrary
from himher.domains.handoff.construct import construct_bernoulli_model
from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Hand, SwitchingHandoffGame
from himher.domains.handoff.policy import TabularQPolicy
from himher.domains.handoff.revise import revise_mixture_model
from himher.experiment import conditions as cond

CONTEXTS = (0, 1)
# Same calibrated H+H defaults as scripts/run_experiments.py.
WINDOW, MAX_LEARNED, TAU, DELTA, ALPHA = 20, 20, 1.0, 0.1, 1.0
LEARNING_RATE, EPSILON, KAPPA = 0.3, 0.15, 0.6705027574263254

METHOD_COLORS = {
    "H+H": "#2a78d6",
    "Fixed library": "#1baf7a",
    "Oracle": "#eda100",
    "Continuous Bayes": "#e87ba4",
}
SWITCH_LINE = "#898781"
TRUE_LINE = "#0b0b0b"
ROLLING = 20

plt.rcParams.update({"font.family": "sans-serif", "axes.grid": True, "grid.color": "#e1e0d9"})


def _loglik(p_right: float, action: Hand) -> float:
    p = float(np.clip(p_right if action == Hand.R else 1.0 - p_right, 1e-12, 1.0))
    return float(np.log(p))


def _hh_run(condition: cond.Condition, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cascade = HHCascade(
        incumbent=cond.LIBRARY_0[0],
        library=ModelLibrary(cond.LIBRARY_0, max_learned=MAX_LEARNED),
        kappa=KAPPA,
        window=WINDOW,
        verifier=HIMVerifier(tau=TAU, delta=DELTA, contexts=CONTEXTS),
        revise=revise_mixture_model,
        construct=lambda w: construct_bernoulli_model(w, ALPHA),
    )
    agent = HHAgent(
        cascade,
        TabularQPolicy(learning_rate=LEARNING_RATE, epsilon=EPSILON),
        np.random.default_rng(seed),
        replay=True,
    )
    game = SwitchingHandoffGame(condition.schedule, np.random.default_rng(seed))
    theta0 = np.empty(condition.episode_length)
    theta1 = np.empty(condition.episode_length)
    loglik = np.empty(condition.episode_length)
    for t in range(condition.episode_length):
        context = game.sample_context()
        before = agent.cascade.incumbent
        ego = agent.act(context)
        result = game.step(context, ego)
        agent.update(context, ego, result.counterpart_action, result.reward)
        loglik[t] = _loglik(before.predict_proba(context)[Hand.R], result.counterpart_action)
        theta0[t] = agent.cascade.incumbent.theta0
        theta1[t] = agent.cascade.incumbent.theta1
    return theta0, theta1, loglik


def _bayes_run(
    condition: cond.Condition, seed: int, hypotheses: tuple[BernoulliHandModel, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_post = np.zeros(len(hypotheses))
    game = SwitchingHandoffGame(condition.schedule, np.random.default_rng(seed))
    theta0 = np.empty(condition.episode_length)
    theta1 = np.empty(condition.episode_length)
    loglik = np.empty(condition.episode_length)
    for t in range(condition.episode_length):
        context = game.sample_context()
        post = np.exp(log_post - log_post.max())
        post /= post.sum()
        p_right = float(sum(w * h.predict_proba(context)[Hand.R] for w, h in zip(post, hypotheses)))
        result = game.step(context, Hand.L)  # ego action does not affect the counterpart draw
        loglik[t] = _loglik(p_right, result.counterpart_action)
        log_post += np.array([h.log_prob(context, result.counterpart_action) for h in hypotheses])
        post = np.exp(log_post - log_post.max())
        post /= post.sum()
        theta0[t] = float(sum(w * h.theta0 for w, h in zip(post, hypotheses)))
        theta1[t] = float(sum(w * h.theta1 for w, h in zip(post, hypotheses)))
    return theta0, theta1, loglik


def _continuous_run(condition: cond.Condition, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history: list[tuple[int, Hand]] = []
    game = SwitchingHandoffGame(condition.schedule, np.random.default_rng(seed))
    theta0 = np.empty(condition.episode_length)
    theta1 = np.empty(condition.episode_length)
    loglik = np.empty(condition.episode_length)
    for t in range(condition.episode_length):
        context = game.sample_context()
        before = construct_bernoulli_model(history, ALPHA)
        result = game.step(context, Hand.L)
        loglik[t] = _loglik(before.predict_proba(context)[Hand.R], result.counterpart_action)
        history.append((context, result.counterpart_action))
        model_now = construct_bernoulli_model(history, ALPHA)
        theta0[t] = model_now.theta0
        theta1[t] = model_now.theta1
    return theta0, theta1, loglik


def run_all(condition: cond.Condition, true_target: BernoulliHandModel, seed: int) -> dict:
    library = cond.LIBRARY_0
    oracle_hyps = library if true_target in library else (*library, true_target)
    return {
        "H+H": _hh_run(condition, seed),
        "Fixed library": _bayes_run(condition, seed, library),
        "Oracle": _bayes_run(condition, seed, oracle_hyps),
        "Continuous Bayes": _continuous_run(condition, seed),
    }


def _switch_times(condition: cond.Condition) -> list[int]:
    return [start for start, _ in condition.schedule.segments if start > 0]


def _true_segments(condition: cond.Condition, param: str) -> list[tuple[int, int, float]]:
    starts = [start for start, _ in condition.schedule.segments] + [condition.episode_length]
    return [
        (starts[i], starts[i + 1], getattr(condition.schedule.segments[i][1], param))
        for i in range(len(condition.schedule.segments))
    ]


def plot_loglik(results: dict, condition: cond.Condition, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, (_, _, loglik) in results.items():
        smoothed = pd.Series(loglik).rolling(ROLLING, min_periods=1).mean().to_numpy()
        ax.plot(np.arange(len(smoothed)), smoothed, color=METHOD_COLORS[name], linewidth=1.6, label=name)
    for s in _switch_times(condition):
        ax.axvline(s, color=SWITCH_LINE, linestyle=":", linewidth=1.0)
    ax.set_xlabel("round")
    ax.set_ylabel(f"predictive log-likelihood (rolling mean, {ROLLING} rounds)")
    ax.set_title(title)
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_l2_distance(results: dict, condition: cond.Condition, title: str, out: Path) -> None:
    """L2 distance in (theta0, theta1) between the model each method holds and
    the true model active at that round, with vertical lines at regime changes."""
    fig, ax = plt.subplots(figsize=(9, 4))
    true_models = [condition.schedule.policy_at(t) for t in range(condition.episode_length)]
    true_theta = np.array([m.theta for m in true_models])
    for name, (theta0, theta1, _) in results.items():
        held = np.column_stack([theta0, theta1])
        distance = np.linalg.norm(held - true_theta, axis=1)
        ax.plot(np.arange(len(distance)), distance, color=METHOD_COLORS[name], linewidth=1.5, label=name)
    for s in _switch_times(condition):
        ax.axvline(s, color=SWITCH_LINE, linestyle=":", linewidth=1.0)
    ax.set_xlabel("round")
    ax.set_ylabel(r"$\|\hat{\theta}_t - \theta_{\mathrm{true},t}\|_2$")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _hh_q(condition: cond.Condition, seed: int) -> np.ndarray:
    cascade = HHCascade(
        incumbent=cond.LIBRARY_0[0],
        library=ModelLibrary(cond.LIBRARY_0, max_learned=MAX_LEARNED),
        kappa=KAPPA,
        window=WINDOW,
        verifier=HIMVerifier(tau=TAU, delta=DELTA, contexts=CONTEXTS),
        revise=revise_mixture_model,
        construct=lambda w: construct_bernoulli_model(w, ALPHA),
    )
    agent = HHAgent(
        cascade,
        TabularQPolicy(learning_rate=LEARNING_RATE, epsilon=EPSILON),
        np.random.default_rng(seed),
        replay=True,
    )
    game = SwitchingHandoffGame(condition.schedule, np.random.default_rng(seed))
    q = np.empty((condition.episode_length, 2))
    for t in range(condition.episode_length):
        context = game.sample_context()
        ego = agent.act(context)
        result = game.step(context, ego)
        agent.update(context, ego, result.counterpart_action, result.reward)
        for x in CONTEXTS:
            q[t, x] = agent.policy.q_value(x, agent.cascade.incumbent, Hand.R)
    return q


def _bayes_q(condition: cond.Condition, seed: int, hypotheses: tuple[BernoulliHandModel, ...]) -> np.ndarray:
    log_post = np.zeros(len(hypotheses))
    game = SwitchingHandoffGame(condition.schedule, np.random.default_rng(seed))
    q = np.empty((condition.episode_length, 2))
    for t in range(condition.episode_length):
        context = game.sample_context()
        result = game.step(context, Hand.L)
        log_post += np.array([h.log_prob(context, result.counterpart_action) for h in hypotheses])
        post = np.exp(log_post - log_post.max())
        post /= post.sum()
        for x in CONTEXTS:
            q[t, x] = float(sum(w * h.predict_proba(x)[Hand.R] for w, h in zip(post, hypotheses)))
    return q


def _continuous_q(condition: cond.Condition, seed: int) -> np.ndarray:
    history: list[tuple[int, Hand]] = []
    game = SwitchingHandoffGame(condition.schedule, np.random.default_rng(seed))
    q = np.empty((condition.episode_length, 2))
    for t in range(condition.episode_length):
        context = game.sample_context()
        result = game.step(context, Hand.L)
        history.append((context, result.counterpart_action))
        model_now = construct_bernoulli_model(history, ALPHA)
        for x in CONTEXTS:
            q[t, x] = model_now.predict_proba(x)[Hand.R]
    return q


def run_q_values(condition: cond.Condition, true_target: BernoulliHandModel, seed: int) -> dict:
    library = cond.LIBRARY_0
    oracle_hyps = library if true_target in library else (*library, true_target)
    return {
        "H+H": _hh_q(condition, seed),
        "Fixed library": _bayes_q(condition, seed, library),
        "Oracle": _bayes_q(condition, seed, oracle_hyps),
        "Continuous Bayes": _continuous_q(condition, seed),
    }


def plot_q_l2(q_results: dict, condition: cond.Condition, title: str, out: Path) -> None:
    """L2 distance between each method's Q(x, a=R) vector (x=0,1) and the true
    Q(x, a=R) = P(counterpart plays R | x) of the model active at that round."""
    fig, ax = plt.subplots(figsize=(9, 4))
    true_q = np.array(
        [[condition.schedule.policy_at(t).predict_proba(x)[Hand.R] for x in CONTEXTS]
         for t in range(condition.episode_length)]
    )
    t_axis = np.arange(condition.episode_length)
    for name, q in q_results.items():
        distance = np.linalg.norm(q - true_q, axis=1)
        ax.plot(t_axis, distance, color=METHOD_COLORS[name], linewidth=1.5, label=name)
    for s in _switch_times(condition):
        ax.axvline(s, color=SWITCH_LINE, linestyle=":", linewidth=1.0)
    ax.set_xlabel("round")
    ax.set_ylabel(r"$\|\hat{Q}(\cdot, R) - Q_{\mathrm{true}}(\cdot, R)\|_2$")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="results/method_comparison")
    parser.add_argument(
        "--conditions", type=str, default="outside_library_switch_m1,outside_library_switch_m2"
    )
    args = parser.parse_args()

    specs = {
        "outside_library_switch_m1": (cond.outside_library_switch(cond.M_NEW_1), cond.M_NEW_1),
        "outside_library_switch_m2": (cond.outside_library_switch(cond.M_NEW_2), cond.M_NEW_2),
        "within_library_switch": (cond.within_library_switch(), cond.M_R),
        "recurring": (cond.recurring(cond.M_NEW_2), cond.M_NEW_2),
    }
    for name in args.conditions.split(","):
        condition, true_target = specs[name.strip()]
        results = run_all(condition, true_target, args.seed)
        out_dir = Path(args.out) / name.strip()
        out_dir.mkdir(parents=True, exist_ok=True)
        title = f"{name.strip()}, seed {args.seed}"
        plot_loglik(results, condition, title, out_dir / f"seed{args.seed}_loglik.png")
        plot_l2_distance(results, condition, title, out_dir / f"seed{args.seed}_l2_distance.png")
        q_results = run_q_values(condition, true_target, args.seed)
        plot_q_l2(q_results, condition, title, out_dir / f"seed{args.seed}_q_l2.png")
        print(f"[{name.strip()}] wrote {out_dir}/seed{args.seed}_loglik.png, _l2_distance.png, _q_l2.png")


if __name__ == "__main__":
    main()
