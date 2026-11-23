"""Audit HER replay against HIM's verification window over all seeds.

For every accepted revision (HH-R, HH-A, HH-R@him_window, HH-R@post_detection):
checks that no replayed transition predates (or postdates) the window HIM
evaluated the accepted candidate on, and counts how many window / replayed
rounds come from a partner behaviour other than the one in force at the
acceptance round ("previous behaviour").

Usage: python scripts/rebuttal_replay_window_audit.py [--seeds 200] [--workers 4]
Writes AAMAS 2027/results/rebuttal_fri/replay_window_audit/acceptances.csv
       AAMAS 2027/reports/replay_window_audit.md
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from himher.experiment import instrumentation as inst  # noqa: E402
from himher.experiment.replay_audit import audit_episode  # noqa: E402

OUT_RUNS = ROOT / "AAMAS 2027" / "results" / "rebuttal_fri" / "replay_window_audit"
OUT_REPORT = ROOT / "AAMAS 2027" / "reports" / "replay_window_audit.md"
METHODS = ("HH-R", "HH-A", "HH-R@him_window", "HH-R@post_detection")
NICE = {"stationary": "Stationary", "within_library_switch": "Within-library",
        "outside_library_switch_m1": "Outside m1", "outside_library_switch_m2": "Outside m2",
        "recurring": "Recurring"}
K = 20


def run_one(task: tuple[str, str, int]) -> list[dict]:
    method, label, seed = task
    audits, _ = audit_episode(method, label, seed)
    first = inst.condition_by_label(label).first_switch
    rows = []
    for a in audits:
        d = asdict(a)
        W, R = a.him_rounds, a.replay_rounds
        rows.append({
            "method": method, "condition": label, "seed": seed, "t": a.t, "regime": a.regime,
            "post_switch": first is not None and a.t >= first,
            "W_first": W[0], "W_last": W[-1], "W_len": len(W),
            "R_first": R[0] if R else None, "R_last": R[-1] if R else None, "R_len": len(R),
            "window_is_last_k": W == tuple(range(max(0, a.t - K + 1), a.t + 1)),
            "him_window_matches": a.him_window_matches,
            "R_subset_W": set(R) <= set(W),
            "R_predates_W": bool(R) and min(R) < min(W),
            "R_equals_W": R == W,
            "W_prev": d["him_prev_behaviour"], "R_prev": d["replay_prev_behaviour"],
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    labels = [s.label for s in inst.condition_specs()]
    tasks = [(m, c, s) for m in METHODS for c in labels for s in range(args.seeds)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        df = pd.DataFrame([r for rows in pool.map(run_one, tasks, chunksize=10) for r in rows])
    OUT_RUNS.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_RUNS / "acceptances.csv", index=False)

    L = ["# HER replay vs HIM's verification window", "",
         f"All accepted revisions, seeds 0–{args.seeds - 1}, 5 conditions. Rounds are identified by the "
         "window actually passed to `HIMVerifier.evaluate` for the accepted candidate (checked "
         "observation-by-observation against the episode) and by object identity of the replayed "
         "`Transition`s. Code: `himher/experiment/replay_audit.py`; tests: `tests/test_replay_window.py`.", "",
         "## 1. Boundary checks", "",
         "| method | acceptances | HIM window = rounds max(0, t−19)..t | window passed to HIM = observed rounds | "
         "replay ⊆ HIM window | replay predates HIM window | replay = HIM window |",
         "|---|---|---|---|---|---|---|"]
    for m in METHODS:
        g = df[df.method == m]
        L.append(f"| {m} | {len(g)} | {g.window_is_last_k.sum()} | {g.him_window_matches.sum()} | "
                 f"{g.R_subset_W.sum()} | {g.R_predates_W.sum()} | {g.R_equals_W.sum()} |")
    L += ["", "## 2. Partner behaviour inside the HIM window and the replay (HH-R)", "",
          "*Previous behaviour* = a round whose true counterpart model differs from the one in force at "
          "the acceptance round t. Shares are over acceptances (rows) or over replayed transitions.", "",
          "| condition | acceptances | after first switch | with ≥1 previous-behaviour round in HIM window | "
          "same, in replay | mean previous rounds per such acceptance | share of all replayed transitions "
          "from previous behaviour | max previous rounds in one replay |",
          "|---|---|---|---|---|---|---|---|"]
    header = L[-2:]
    for m in ("HH-R", "HH-R@post_detection"):
        if m != "HH-R":
            L += ["", f"**{m}** (replay limited to the current detection run)", "", *header]
        for c in labels:
            g = df[(df.method == m) & (df.condition == c)]
            cont = g[g.R_prev > 0]
            share = g.R_prev.sum() / max(g.R_len.sum(), 1)
            L.append(f"| {NICE[c]} | {len(g)} | {int(g.post_switch.sum())} | {int((g.W_prev > 0).sum())} | "
                     f"{len(cont)} | {cont.R_prev.mean() if len(cont) else 0:.1f} | {share:.3f} | "
                     f"{int(g.R_prev.max()) if len(g) else 0} |")
    g = df[(df.method == "HH-R") & (df.R_prev > 0)]
    L += ["", "## 3. HH-R acceptances that replay previous-behaviour rounds, by operator", "",
          "| operator | acceptances | mean previous rounds replayed |", "|---|---|---|"]
    for r, gg in g.groupby("regime"):
        L.append(f"| {r} | {len(gg)} | {gg.R_prev.mean():.1f} |")
    OUT_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote AAMAS 2027/reports/replay_window_audit.md")


if __name__ == "__main__":
    main()
