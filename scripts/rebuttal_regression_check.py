"""Default-flag regression check (brief rule 6): compares a rebuttal run
directory (original methods, log_level=full) against the stored paper
results in "AAMAS 2027/results" -- per run and, where stored, per step.

Usage: python scripts/rebuttal_regression_check.py results/rebuttal_fri/regression
Exit code 1 on any mismatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STORED = ROOT / "AAMAS 2027" / "results"
PAPER_NAME = {"HH-R": "hh_replay", "HH-A": "hh_ablation", "FB": "fixed_bayes",
              "OB": "oracle_bayes", "CB": "continuous_bayes"}


def compare(name: str, new: pd.Series, old: pd.Series, lines: list[str]) -> bool:
    new = new.astype(float).to_numpy()
    old = old.astype(float).to_numpy()
    both_nan = np.isnan(new) & np.isnan(old)
    diff = np.where(both_nan, 0.0, np.abs(new - old))
    diff = np.where(np.isnan(diff), np.inf, diff)
    bitwise = bool(np.all(both_nan | (new == old)))
    ok = bool(diff.max(initial=0.0) <= 1e-6)
    lines.append(f"| {name} | {len(new)} | {'yes' if bitwise else 'no'} | {diff.max(initial=0.0):.3g} | {'PASS' if ok else 'FAIL'} |")
    return ok


def main() -> None:
    run_dir = Path(sys.argv[1])
    s = pd.read_csv(run_dir / "summary.csv")
    s = s[s.variant == "default"].copy()
    s["paper_method"] = s.method.map(PAPER_NAME)
    s = s.dropna(subset=["paper_method"])
    seeds = sorted(s.seed.unique())
    lines = ["| quantity | n compared | bitwise | max abs diff | verdict |", "|---|---|---|---|---|"]
    ok = True

    main_raw = pd.read_csv(STORED / "main_raw.csv")
    j = s.merge(main_raw, left_on=["condition", "paper_method", "seed"],
                right_on=["condition", "method", "seed"], suffixes=("", "_old"))
    ok &= compare("auc vs cumulative_return", j.auc, j.cumulative_return, lines)
    sw = j[j.condition != "stationary"]
    ok &= compare("regret_post vs post_switch_regret", sw.regret_post, sw.post_switch_regret, lines)
    ok &= compare("recovery_time vs adaptation_delay", sw.recovery_time, sw.adaptation_delay, lines)
    hh = j[j.method.isin(["HH-R", "HH-A"])]
    ok &= compare("n_detections/T vs false_alarm_rate", hh.n_detections / hh["T"], hh.false_alarm_rate, lines)
    ok &= compare("n_accept_reuse", hh.n_accept_reuse, hh.economy_reuse_accepted, lines)
    ok &= compare("n_accept_merge vs revise_accepted", hh.n_accept_merge, hh.economy_revise_accepted, lines)
    ok &= compare("n_accept_construct", hh.n_accept_construct, hh.economy_construct_accepted, lines)
    ok &= compare("n_revisions", hh.n_revisions, hh.economy_revisions, lines)
    ok &= compare("n_policy_updates", hh.n_policy_updates, hh.policy_updates_total, lines)

    phase = pd.read_csv(STORED / "phase_return_raw.csv")
    j = s.merge(phase, left_on=["condition", "paper_method", "seed"], right_on=["condition", "method", "seed"])
    ok &= compare("return_pre vs return_before", j.return_pre, j.return_before, lines)
    ok &= compare("return_post vs return_after", j.return_post, j.return_after, lines)

    npz = run_dir / "per_step.npz"
    if npz.exists():
        d = np.load(npz)
        ps = pd.DataFrame({k: d[k] for k in ("run_id", "t", "model_theta0", "model_theta1", "kld")})
        parts = ps.run_id.str.split("__", expand=True)
        ps["condition"], ps["method"] = parts[0], parts[1].map(PAPER_NAME)
        ps["seed"] = parts[3].str[1:].astype(int)
        trace = pd.read_csv(STORED / "model_trace_raw.csv")
        trace = trace[trace.seed.isin(seeds)]
        j = ps.merge(trace, on=["condition", "method", "seed", "t"])
        ok &= compare("per-step theta0 vs model_trace_raw", j.model_theta0, j.theta0, lines)
        ok &= compare("per-step theta1 vs model_trace_raw", j.model_theta1, j.theta1, lines)
        kld = pd.read_csv(STORED / "kld_raw.csv")
        kld = kld[kld.seed.isin(seeds)]
        j = ps.merge(kld, on=["condition", "method", "seed", "t"], suffixes=("", "_old"))
        ok &= compare("per-step kld vs kld_raw", j.kld, j.kld_old, lines)

    report = "\n".join(lines)
    print(report)
    (run_dir / "regression_check.md").write_text(
        f"Seeds {seeds[0]}..{seeds[-1]} ({len(seeds)}); tolerance 1e-6.\n\n{report}\n\nOVERALL: "
        f"{'PASS' if ok else 'FAIL'}\n", encoding="utf-8")
    print("OVERALL:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
