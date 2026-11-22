"""Task C4/T2 input: H+H counter totals split by phase (pre/post first
switch; 50 steps after each switch vs the rest). A separate deterministic
pass (parallel; counts do not depend on timing), checked against the C3
totals in cost_runs.csv.

Usage: python scripts/rebuttal_cost_phases.py [--seeds 200] [--workers 14]
Writes AAMAS 2027/results/rebuttal_cost/hh_phase_split.csv
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from himher.experiment import cost_counters as cc  # noqa: E402
from himher.experiment import instrumentation as inst  # noqa: E402
from himher.experiment.cost_runner import make_agent, run_cost  # noqa: E402

OUT = ROOT / "AAMAS 2027" / "results" / "rebuttal_cost"
HH = ("HH-R", "HH-A")


def run_one(task: tuple[str, str, int]) -> dict:
    key, label, seed = task
    spec = inst.condition_by_label(label)
    row = run_cost(make_agent(key, seed, spec), spec, seed, phase_split=True)["row"]
    return {"seed": seed, "condition": label, "method": key, **row}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    tasks = [(k, s.label, seed) for k in HH for s in inst.condition_specs() for seed in range(args.seeds)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = pool.map(run_one, tasks, chunksize=10)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "hh_phase_split.csv", index=False)
    c3 = pd.read_csv(OUT / "cost_runs.csv")
    j = df.merge(c3, on=["method", "condition", "seed"], suffixes=("", "_c3"))
    ok = all((j[f"ev_{c}"] == j[f"ev_{c}_c3"]).all() and
             (j[f"ev_{c}_pre"] + j[f"ev_{c}_post"] == j[f"ev_{c}"]).all() and
             (j[f"ev_{c}_w50"] + j[f"ev_{c}_rest"] == j[f"ev_{c}"]).all() for c in cc.EV_CATS)
    ok &= bool((j.pol_updates == j.pol_updates_c3).all())
    print(f"{len(df)} runs; matched C3 rows {len(j)}; totals and phase sums consistent: {ok}")


if __name__ == "__main__":
    main()
