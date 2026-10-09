"""Tests whether H+H recovers the true Q-values better than the baselines.

For every condition and seed, runs H+H and the three Bayesian baselines on
the same counterpart sequence (plot_method_comparison.run_q_values) and
computes the per-round L2 error between each method's Q(x, a=R) vector and
the true Q(x, a=R) = P(counterpart plays R | x) of the active model.

Per seed, the error is averaged over three windows:
  pre   rounds 20..s-1       (before the first switch, burn-in dropped)
  post  rounds s..s+49        (immediately after the first switch)
  late  rounds 250..299       (steady state)
where s is the first switch round of the condition.

Tests (paired by seed; H+H vs each baseline; one-sided, H+H error < baseline):
  - Wilcoxon signed-rank on the pre, post and late errors, and on the
    change post - pre;
  - Holm correction across all tests in the file;
  - median paired difference (H+H - baseline) with a bootstrap 95% CI.

Outputs under <out>/: q_errors_per_seed.csv and q_recovery_tests.csv.

Usage: python scripts/q_recovery_tests.py [--seeds N] [--out DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_method_comparison import run_q_values  # noqa: E402

from himher.domains.handoff.env import Hand  # noqa: E402
from himher.experiment import conditions as cond  # noqa: E402

BASELINES = ["Fixed library", "Oracle", "Continuous Bayes"]
BURN_IN = 20
LATE = (250, 300)
N_BOOT = 10_000


def _specs() -> dict:
    return {
        "stationary": (cond.stationary(), cond.M_L, 150),
        "within_library_switch": (cond.within_library_switch(), cond.M_R, 150),
        "outside_library_switch_m1": (cond.outside_library_switch(cond.M_NEW_1), cond.M_NEW_1, 150),
        "outside_library_switch_m2": (cond.outside_library_switch(cond.M_NEW_2), cond.M_NEW_2, 150),
        "recurring": (cond.recurring(cond.M_NEW_2), cond.M_NEW_2, 75),
    }


def _errors(q: np.ndarray, condition: cond.Condition) -> np.ndarray:
    true_q = np.array(
        [[condition.schedule.policy_at(t).predict_proba(x)[Hand.R] for x in (0, 1)]
         for t in range(condition.episode_length)]
    )
    return np.linalg.norm(q - true_q, axis=1)


def _window_means(err: np.ndarray, switch: int) -> dict[str, float]:
    return {
        "pre": float(err[BURN_IN:switch].mean()),
        "post": float(err[switch : switch + 50].mean()),
        "late": float(err[LATE[0] : LATE[1]].mean()),
    }


def _holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def _bootstrap_ci(diff: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
    medians = np.median(diff[idx], axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--out", type=str, default="results/q_recovery")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, (condition, true_target, switch) in _specs().items():
        for seed in range(args.seeds):
            q_results = run_q_values(condition, true_target, seed)
            for method, q in q_results.items():
                err = _errors(q, condition)
                rows.append({"condition": name, "method": method, "seed": seed, **_window_means(err, switch)})
            print(f"[{name}] seed {seed} done", flush=True)
    per_seed = pd.DataFrame(rows)
    per_seed["change"] = per_seed["post"] - per_seed["pre"]
    per_seed.to_csv(out_dir / "q_errors_per_seed.csv", index=False)

    rng = np.random.default_rng(0)
    tests = []
    for name in _specs():
        hh = per_seed[(per_seed.condition == name) & (per_seed.method == "H+H")].sort_values("seed")
        for baseline in BASELINES:
            bl = per_seed[(per_seed.condition == name) & (per_seed.method == baseline)].sort_values("seed")
            for window in ["pre", "post", "late", "change"]:
                diff = hh[window].to_numpy() - bl[window].to_numpy()
                if np.allclose(diff, 0):
                    p = 1.0
                else:
                    p = float(wilcoxon(diff, alternative="less").pvalue)
                lo, hi = _bootstrap_ci(diff, rng)
                tests.append({
                    "condition": name, "baseline": baseline, "window": window,
                    "median_diff_hh_minus_baseline": float(np.median(diff)),
                    "ci95_low": lo, "ci95_high": hi,
                    "mean_hh": float(hh[window].mean()), "mean_baseline": float(bl[window].mean()),
                    "p_one_sided": p,
                })
    tests_df = pd.DataFrame(tests)
    tests_df["p_holm"] = _holm(tests_df["p_one_sided"].tolist())
    tests_df.to_csv(out_dir / "q_recovery_tests.csv", index=False)
    print(tests_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
