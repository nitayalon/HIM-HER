"""Runs the Predator-Prey experiments (see REPORT.md) and writes results.

Usage:
    python scripts/run_pp_experiments.py [--seeds 200] [--games 40] [--out results/predator_prey]

Two passes, as for the handoff game:
  1. A kappa sweep on the Stationary condition; the smallest kappa whose
     false-alarm rate is <= 5% is chosen.
  2. Every condition x method, one episode (40 games) per seed.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from himher.domains.predator_prey.experiment import METHODS, conditions, run_episode
from himher.experiment.aggregate import summarize

KAPPA_GRID = tuple(np.round(np.arange(1.0, 2.01, 0.1), 2))
TARGET_FALSE_ALARM_RATE = 0.05


def _run(args: tuple[str, str, int, float, int]) -> dict:
    method, condition_name, seed, kappa, games = args
    condition = next(c for c in conditions() if c.name == condition_name)
    record = run_episode(method, condition, seed, kappa, n_games=games)
    record["kappa"] = kappa
    return record


def _run_all(tasks: list, workers: int) -> pd.DataFrame:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return pd.DataFrame(list(pool.map(_run, tasks, chunksize=20)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--out", type=str, default="results/predator_prey")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sweep = _run_all(
        [("hh_replay", "stationary", s, k, args.games) for k in KAPPA_GRID for s in range(args.seeds)],
        args.workers,
    )
    sweep.to_csv(out / "kappa_sweep_raw.csv", index=False)
    sweep_summary = summarize(sweep, ["kappa"], ["false_alarm_rate", "revisions", "kl_mean"])
    sweep_summary.to_csv(out / "kappa_sweep_summary.csv", index=False)
    feasible = sweep_summary[sweep_summary["false_alarm_rate"] <= TARGET_FALSE_ALARM_RATE]
    kappa = float(feasible["kappa"].min() if not feasible.empty else sweep_summary["kappa"].max())
    print(sweep_summary[["kappa", "false_alarm_rate", "revisions", "kl_mean"]].to_string(index=False))
    print(f"[kappa sweep] chosen kappa = {kappa}")

    tasks = [
        (method, c.name, s, kappa, args.games)
        for c in conditions()
        for method in METHODS
        for s in range(args.seeds)
    ]
    df = _run_all(tasks, args.workers)
    df.to_csv(out / "main_raw.csv", index=False)

    value_columns = [
        c for c in df.columns
        if c not in ("condition", "method", "seed", "kappa") and pd.api.types.is_numeric_dtype(df[c])
    ]
    summary = summarize(df, ["condition", "method"], value_columns)
    summary.to_csv(out / "main_summary.csv", index=False)
    print(f"Done: kappa={kappa}, wrote {out}/")


if __name__ == "__main__":
    main()
