"""Monday brief rule 1.4: with new flags at defaults, rerun HH-R and WB-k20
on 10 seeds in every condition and require per-run outputs to be bitwise
identical to the stored Friday runs (summary: every column except
wall-clock; events: every row and column, HH-R).

Usage: python scripts/rebuttal_mon_regression.py TASK [--seeds 10]
Writes AAMAS 2027/results/rebuttal_mon/<TASK>/regression.md; exit code 1 on mismatch.
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
from himher.experiment.rebuttal_methods import METHODS  # noqa: E402

FRI = ROOT / "AAMAS 2027" / "results" / "rebuttal_fri"
# HH-R: A_replay_scope (its default-variant HH-R runs equal `orig` and also carry the
# replay columns added to the event log by Task A, which `orig` predates).
STORED = {"HH-R": "A_replay_scope", "WB-k20": "B_baselines"}
SKIP = {"wall_clock_s"}


def _init() -> None:
    inst.install_loglik_counter()  # n_loglik_evals is part of the stored summary


def _run(task):
    key, label, seed = task
    spec = inst.condition_by_label(label)
    ms = METHODS[key]
    r = inst.run_instrumented(ms.factory(seed, spec), spec, seed, ms.method, ms.variant, "events")
    return r.summary, r.events


def _roundtrip(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    df.to_csv(path, index=False)
    out = pd.read_csv(path, float_precision="round_trip")
    path.unlink()
    return out


def _same(a: pd.DataFrame, b: pd.DataFrame) -> tuple[bool, list[str]]:
    bad = []
    for c in a.columns:
        if c in SKIP:
            continue
        if c not in b.columns:
            bad.append(f"{c} (missing in stored)")
            continue
        x, y = a[c].to_numpy(), b[c].to_numpy()
        eq = (x == y) | (pd.isna(x) & pd.isna(y))
        if not np.all(eq):
            bad.append(f"{c} ({int((~eq).sum())} rows)")
    return not bad, bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    out = ROOT / "AAMAS 2027" / "results" / "rebuttal_mon" / args.task
    out.mkdir(parents=True, exist_ok=True)
    labels = [s.label for s in inst.condition_specs()]
    tasks = [(k, c, s) for k in STORED for c in labels for s in range(args.seeds)]
    with mp.get_context("spawn").Pool(args.workers, initializer=_init) as pool:
        res = pool.map(_run, tasks)
    lines = [f"# Regression check ({args.task})", "",
             f"HH-R and WB-k20, {args.seeds} seeds × {len(labels)} conditions, new flags at defaults, "
             "against the stored Friday runs (`rebuttal_fri/A_replay_scope` for HH-R, `rebuttal_fri/B_baselines` for WB-k20). Bitwise after a "
             "CSV round trip (`float_precision=round_trip`); `wall_clock_s` excluded.", "",
             "| method | what | rows | identical | mismatching columns |", "|---|---|---|---|---|"]
    ok_all = True
    for key, exp in STORED.items():
        new = pd.DataFrame([s for (k, _, _), (s, _) in zip(tasks, res) if k == key])
        new = _roundtrip(new, out / "_new.csv").sort_values(["condition", "seed"]).reset_index(drop=True)
        old = pd.read_csv(FRI / exp / "summary.csv", float_precision="round_trip")
        old = old[(old.method == key) & (old.variant == "default") & (old.seed < args.seeds)]
        old = old.sort_values(["condition", "seed"]).reset_index(drop=True)[new.columns.intersection(old.columns)]
        ok = len(new) == len(old)
        same, bad = _same(new, old) if ok else (False, ["row count"])
        ok_all &= same
        lines.append(f"| {key} | summary | {len(new)} | {'yes' if same else 'NO'} | {', '.join(bad) or '—'} |")
        evs = [e for (k, _, _), (_, ev) in zip(tasks, res) if k == key for e in ev]
        if evs:
            ne = _roundtrip(pd.DataFrame(evs), out / "_ev.csv")
            ne = ne.sort_values(["run_id", "t"], kind="stable").reset_index(drop=True)
            oe = pd.read_csv(FRI / exp / "events.csv", float_precision="round_trip")
            keep = set(new.run_id)
            oe = oe[oe.run_id.isin(keep)].sort_values(["run_id", "t"], kind="stable").reset_index(drop=True)
            oe = oe[[c for c in ne.columns if c in oe.columns]]
            same_e, bad_e = _same(ne, oe) if len(ne) == len(oe) else (False, [f"row count {len(ne)} vs {len(oe)}"])
            ok_all &= same_e
            lines.append(f"| {key} | events | {len(ne)} | {'yes' if same_e else 'NO'} | {', '.join(bad_e) or '—'} |")
    lines += ["", f"**{'PASS' if ok_all else 'FAIL'}**"]
    (out / "regression.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"regression {args.task}: {'PASS' if ok_all else 'FAIL'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
