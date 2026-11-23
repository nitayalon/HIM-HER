"""Task C3: counted runs for every method x condition x seed.

Wall-clock hygiene (brief 5.3): one process, runs executed serially,
OMP/MKL/OPENBLAS threads = 1 (set before numpy is imported), and one
discarded warm-up run per method before its timed runs.

Performance columns are recomputed in the counted run and checked against
the stored rebuttal summaries (orig, B_baselines, BPR_mdp; BPR_baseline holds
the first Bernoulli BPR+, compared as BPR+bern) for every
run that exists there; BOCPD has no stored run.

Usage:
  python scripts/rebuttal_cost_runs.py --seeds 5 --out smoke      # smoke test
  python scripts/rebuttal_cost_runs.py --seeds 200 --out full     # full run
Writes AAMAS 2027/results/rebuttal_cost/<out>/{cost_runs.csv, cost_per_step.csv, meta.json,
stored_check.md}; for --out full, cost_runs.csv is also copied to
AAMAS 2027/results/rebuttal_cost/cost_runs.csv.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from himher.experiment import instrumentation as inst  # noqa: E402
from himher.experiment.cost_runner import COST_METHODS, make_agent, run_cost  # noqa: E402

OUT_ROOT = ROOT / "AAMAS 2027" / "results" / "rebuttal_cost"
STORED = ROOT / "AAMAS 2027" / "results" / "rebuttal_fri"
DIAG_SEEDS = range(5)
DIAG_CONDS = ("within_library_switch", "outside_library_switch_m2")


def cpu_name() -> str:
    try:
        import subprocess
        out = subprocess.run(["wmic", "cpu", "get", "name"], capture_output=True, text=True, timeout=10).stdout
        lines = [l.strip() for l in out.splitlines() if l.strip() and l.strip() != "Name"]
        if lines:
            return lines[0]
    except Exception:
        pass
    return platform.processor()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--out", default="full")
    ap.add_argument("--methods", nargs="*", default=list(COST_METHODS))
    args = ap.parse_args()
    out = OUT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    specs = inst.condition_specs()

    rows, steps = [], []
    t0 = time.time()
    for key in args.methods:
        warm = specs[1]
        run_cost(make_agent(key, 0, warm), warm, 0)  # warm-up, discarded
        for spec in specs:
            for seed in range(args.seeds):
                diag = seed in DIAG_SEEDS and spec.label in DIAG_CONDS
                res = run_cost(make_agent(key, seed, spec), spec, seed, record_steps=diag)
                row = {"run_id": f"{spec.label}__{key}__default__s{seed}", "seed": seed,
                       "condition": spec.label, "method": key, "variant": "default", **res["row"]}
                rows.append(row)
                if diag:
                    steps += [{"method": key, "condition": spec.label, "seed": seed, "t": t, "cum_ev_total": v}
                              for t, v in enumerate(res["extras"]["cum_ev"])]
        print(f"{key}: done ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "cost_runs.csv", index=False)
    pd.DataFrame(steps).to_csv(out / "cost_per_step.csv", index=False)

    # Cross-check recomputed performance against the stored rebuttal runs.
    # Compare both sides after the same CSV round trip, parsed exactly.
    df = pd.read_csv(out / "cost_runs.csv", float_precision="round_trip")
    stored = pd.concat([pd.read_csv(STORED / e / "summary.csv", float_precision="round_trip")
                        for e in ("orig", "B_baselines", "BPR_mdp")]
                       + [pd.read_csv(STORED / "BPR_baseline" / "summary.csv", float_precision="round_trip")
                          .replace({"method": {"BPR+": "BPR+bern"}})]
                       + [pd.read_csv(p, float_precision="round_trip")  # Monday Task P runs, when present
                          for p in [STORED.parent / "rebuttal_mon" / "P" / "summary.csv"] if p.exists()])
    stored = stored[stored.variant == "default"].drop_duplicates(["method", "condition", "seed"])
    cols = ["return_pre", "return_post", "return_total", "auc", "regret_post", "recovery_time"]
    j = df.merge(stored[["method", "condition", "seed", *cols]], on=["method", "condition", "seed"],
                 suffixes=("", "_stored"))
    lines = ["| column | runs compared | identical (NaN = NaN) |", "|---|---|---|"]
    all_ok = True
    for c in cols:
        a, b = j[c].to_numpy(float), j[f"{c}_stored"].to_numpy(float)
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        all_ok &= bool(same.all())
        lines.append(f"| {c} | {len(j)} | {int(same.sum())} |")
    n_new = len(df) - len(j)
    lines.append(f"\nRuns with no stored counterpart (recomputed only): {n_new} "
                 f"({', '.join(sorted(set(df.method) - set(j.method))) or 'none'}).")
    lines.append(f"OVERALL: {'PASS' if all_ok else 'FAIL'}")
    (out / "stored_check.md").write_text("\n".join(lines), encoding="utf-8")

    num = df.select_dtypes("number")
    finite = bool(np.isfinite(num.drop(columns=["return_post", "regret_post", "recovery_time",
                                                  "n_detections", "n_accept_reuse", "n_accept_merge",
                                                  "n_accept_construct", "n_reject", "n_voc_declined",
                                                  "n_learned_models"], errors="ignore")).all().all())
    meta = {"command": " ".join(sys.argv), "seeds": f"0..{args.seeds - 1}", "n_runs": len(df),
            "methods": args.methods, "cpu": cpu_name(), "logical_cores": os.cpu_count(),
            "platform": platform.platform(), "python": sys.version.split()[0], "numpy": np.__version__,
            "pandas": pd.__version__, "threads_env": {v: os.environ[v] for v in
                                                      ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "execution": "single process, serial, one discarded warm-up run per method",
            "total_wall_s": time.time() - t0, "stored_check": "PASS" if all_ok else "FAIL",
            "counters_finite": finite, "ev_unattributed_total": int(df.ev_unattributed.sum())}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if args.out == "full":
        shutil.copy(out / "cost_runs.csv", OUT_ROOT / "cost_runs.csv")
    print(f"{len(df)} runs, stored check {meta['stored_check']}, finite={finite}, "
          f"unattributed={meta['ev_unattributed_total']}, {meta['total_wall_s']:.0f}s")


if __name__ == "__main__":
    main()
