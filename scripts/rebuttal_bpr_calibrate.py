"""Calibrate BPR+'s new-model threshold rho the way H+H's kappa was
calibrated: a sweep on the Stationary condition (seeds 0-199), choosing
the most sensitive setting whose false-alarm level does not exceed a
target. The paper's rho = 0.8 was tuned for the deterministic iPD; here
even the correct model with its best response gives a mean reward
likelihood of ~0.82 against M_L, so it is not transferable.

Target: H+H's realised false alarms per Stationary episode at the frozen
kappa (mean `n_false_alarms`, HH-R, results/rebuttal_fri/orig), i.e. BPR+
may declare at most as many spurious new strategies as H+H flags
spurious mismatches. n = 7 (paper), and the other hyper-parameters are
those registered for "BPR+" in rebuttal_methods.

Usage: python scripts/rebuttal_bpr_calibrate.py [--workers 14]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from himher.experiment import instrumentation as inst  # noqa: E402
from himher.experiment.rebuttal_methods import make_bpr_agent  # noqa: E402
from himher.experiment.runner import run_episode  # noqa: E402

RHO_GRID = tuple(round(x, 2) for x in np.arange(0.30, 0.801, 0.01))
OUT_RUNS = ROOT / "AAMAS 2027" / "results" / "rebuttal_fri" / "bpr_calibration"
OUT_TABLES = ROOT / "AAMAS 2027" / "reports" / "rebuttal_fri_tables"


def run_one(args: tuple[float, int]) -> dict:
    rho, seed = args
    spec = inst.condition_by_label("stationary")
    agent = make_bpr_agent(seed, rho=rho)
    logs = run_episode(agent, spec.condition, seed)
    return {"rho": rho, "seed": seed, "n_false_alarms": len(agent.detection_times),
            "return_total": float(np.mean([l.reward for l in logs]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    orig = pd.read_csv(ROOT / "AAMAS 2027/results/rebuttal_fri/orig/summary.csv")
    target = float(orig[(orig.method == "HH-R") & (orig.condition == "stationary")].n_false_alarms.mean())

    tasks = [(rho, s) for rho in RHO_GRID for s in range(200)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = pool.map(run_one, tasks, chunksize=50)
    df = pd.DataFrame(rows)
    OUT_RUNS.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_RUNS / "rho_sweep_raw.csv", index=False)
    summ = df.groupby("rho").agg(n_false_alarms=("n_false_alarms", "mean"),
                                 return_total=("return_total", "mean")).reset_index()
    summ.to_csv(OUT_RUNS / "rho_sweep_summary.csv", index=False)
    feasible = summ[summ.n_false_alarms <= target]
    chosen = float(feasible.rho.max()) if not feasible.empty else float(summ.rho.min())

    lines = [f"BPR+ rho calibration on Stationary (seeds 0-199, n = 7). Target = H+H's mean false alarms "
             f"per Stationary episode at the frozen kappa = {target:.3f}. Chosen rho = **{chosen:.2f}** "
             "(largest rho, i.e. most sensitive, with mean false alarms <= target).", "",
             "| rho | mean false alarms / episode | mean return |", "|---|---|---|"]
    for _, r in summ.iterrows():
        mark = " ←" if abs(r.rho - chosen) < 1e-9 else ""
        lines.append(f"| {r.rho:.2f}{mark} | {r.n_false_alarms:.3f} | {r.return_total:.3f} |")
    (OUT_TABLES / "bpr_calibration.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"target={target:.3f} chosen rho={chosen:.2f}")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
