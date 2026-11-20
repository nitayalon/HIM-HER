"""Rebuttal (Fri 11/20) experiment runner: runs registered methods x
conditions x seeds through himher.experiment.instrumentation and writes

    AAMAS 2027/results/rebuttal_fri/<experiment>/summary.csv   one row per run
    AAMAS 2027/results/rebuttal_fri/<experiment>/events.csv    log_level >= events
    AAMAS 2027/results/rebuttal_fri/<experiment>/per_step.npz  log_level == full
    AAMAS 2027/results/rebuttal_fri/<experiment>/timing.csv    per method
    AAMAS 2027/results/rebuttal_fri/<experiment>/meta.json     command, hardware, versions

Seeds are 0..N-1 (the paper's seed list is range(200) for every
condition x method), so every method sees identical contexts, counterpart
draws and switch times for a given seed (paired comparisons).

Usage:
    python scripts/rebuttal_fri.py --experiment NAME --methods HH-R CB \
        [--seeds 200] [--conditions ...] [--log-level events] [--workers 14]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from himher.experiment import instrumentation as inst  # noqa: E402
from himher.experiment.rebuttal_methods import METHODS  # noqa: E402

OUT_ROOT = ROOT / "AAMAS 2027" / "results" / "rebuttal_fri"


def _init_worker() -> None:
    inst.install_loglik_counter()


def _run_one(task: tuple[str, str, int, str]):
    key, label, seed, log_level = task
    spec = inst.condition_by_label(label)
    mspec = METHODS[key]
    agent = mspec.factory(seed, spec)
    result = inst.run_instrumented(agent, spec, seed, mspec.method, mspec.variant, log_level)
    return key, result


def _hardware() -> dict:
    cpu = platform.processor()
    try:
        out = subprocess.run(["wmic", "cpu", "get", "name"], capture_output=True, text=True, timeout=10)
        names = [l.strip() for l in out.stdout.splitlines() if l.strip() and l.strip() != "Name"]
        if names:
            cpu = names[0]
    except Exception:
        pass
    return {"cpu": cpu, "logical_cores": os.cpu_count(), "platform": platform.platform(),
            "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "jax": "not used (codebase is NumPy-only)"}


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return ""


def run_experiment(experiment: str, method_keys: list[str], conditions: list[str], seeds: list[int],
                   log_level: str, workers: int) -> Path:
    out = OUT_ROOT / experiment
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(k, c, s, log_level) for k in method_keys for c in conditions for s in seeds]
    t0 = time.perf_counter()
    summaries, events, steps = [], [], []
    step_times: dict[str, list[np.ndarray]] = {k: [] for k in method_keys}
    if workers <= 1:
        _init_worker()
        results = map(_run_one, tasks)
    else:
        pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker)
        results = pool.imap_unordered(_run_one, tasks, chunksize=4)
    for i, (key, r) in enumerate(results, 1):
        summaries.append(r.summary)
        events.extend(r.events)
        if r.per_step is not None:
            steps.append(r.per_step)
        step_times[key].append(r.step_times)
        if i % 500 == 0:
            print(f"  [{experiment}] {i}/{len(tasks)} runs", flush=True)
    if workers > 1:
        pool.close()
        pool.join()
    elapsed = time.perf_counter() - t0

    sort_cols = ["condition", "method", "variant", "seed"]
    summary = pd.DataFrame(summaries).sort_values(sort_cols).reset_index(drop=True)
    summary.to_csv(out / "summary.csv", index=False)

    if log_level != "none":
        ev = pd.DataFrame(events)
        if not ev.empty:
            ev = ev.sort_values(["run_id", "t"], kind="stable").reset_index(drop=True)
        ev.to_csv(out / "events.csv", index=False)
    if steps:
        merged = {k: np.concatenate([s[k] for s in steps]) for k in steps[0]}
        np.savez_compressed(out / "per_step.npz", **merged)

    timing_rows = []
    for key in method_keys:
        sub = summary[(summary.method == METHODS[key].method) & (summary.variant == METHODS[key].variant)]
        per_step = np.concatenate(step_times[key]) if step_times[key] else np.array([np.nan])
        timing_rows.append({
            "method": METHODS[key].method, "variant": METHODS[key].variant,
            "compile_time_s": "n/a (no JIT; NumPy)",
            "n_runs": len(sub),
            "wall_per_step_mean_s": float(per_step.mean()),
            "wall_per_step_std_s": float(per_step.std()),
            "wall_per_run_mean_s": float(sub.wall_clock_s.mean()),
            "loglik_evals_per_run_mean": float(sub.n_loglik_evals.mean()),
            "revisions_per_run_mean": float(sub.n_revisions.mean()) if sub.n_revisions.notna().any() else np.nan,
            "workers": workers,
        })
    pd.DataFrame(timing_rows).to_csv(out / "timing.csv", index=False)

    meta = {"experiment": experiment, "methods": method_keys, "conditions": conditions,
            "seeds": [min(seeds), max(seeds), len(seeds)], "log_level": log_level,
            "workers": workers, "elapsed_s": elapsed, "git_head": _git_head(),
            "argv": sys.argv, "hardware": _hardware()}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[{experiment}] {len(tasks)} runs in {elapsed:.1f}s -> AAMAS 2027/results/rebuttal_fri/{experiment}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--conditions", nargs="+", default=[s.label for s in inst.condition_specs()])
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--log-level", default="events", choices=inst.LOG_LEVELS)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = p.parse_args()
    unknown = [k for k in a.methods if k not in METHODS]
    if unknown:
        raise SystemExit(f"unknown methods {unknown}; known: {sorted(METHODS)}")
    run_experiment(a.experiment, a.methods, a.conditions, list(range(a.seeds)), a.log_level, a.workers)


if __name__ == "__main__":
    main()
