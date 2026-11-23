"""Monday brief, Task 0: two checks on existing outputs (no new method runs).

0a  n_replayed per acceptance, full vs post_detection (Friday A_replay_scope
    events), per condition: mean and 95% bootstrap CI over seeds (seed means
    first), and the fraction of acceptances with 0 replayed transitions.
0b  per-context contamination under replay_scope = full, from the replay
    diagnostic's per_acceptance.csv (Within, m2, Recurring: Sun 11/22 run;
    m1: rebuttal_mon/task0/replay_diag_m1, same code, 20 seeds), plus each
    condition's pre/post-switch parameters from himher/experiment/conditions.py.

Usage: python analysis/mon_task0.py
Writes AAMAS 2027/reports/MON_task0_checks.md and
       AAMAS 2027/results/rebuttal_mon/task0/{0a_n_replayed,0b_per_context}.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from himher.experiment import instrumentation as inst  # noqa: E402

RES = ROOT / "AAMAS 2027" / "results"
OUT = RES / "rebuttal_mon" / "task0"
REPORT = ROOT / "AAMAS 2027" / "reports" / "MON_task0_checks.md"
N_BOOT, BOOT_SEED = 10_000, 20261122
DELTA = 0.1
CONDS = ("stationary", "within_library_switch", "outside_library_switch_m1",
         "outside_library_switch_m2", "recurring")
NICE = {"stationary": "Stationary", "within_library_switch": "Within-library",
        "outside_library_switch_m1": "Outside m1", "outside_library_switch_m2": "Outside m2",
        "recurring": "Recurring"}


def boot(x) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    bs = x[np.random.default_rng(BOOT_SEED).integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def f(t, d=3) -> str:
    return "—" if np.isnan(t[0]) else f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    L = ["# Monday Task 0 — checks on existing outputs", ""]

    # ---- 0a --------------------------------------------------------------
    ev = pd.read_csv(RES / "rebuttal_fri" / "A_replay_scope" / "events.csv",
                     usecols=["run_id", "t", "event", "op", "n_replayed"])
    ev = ev[ev.event == "accept"].copy()
    parts = ev.run_id.str.split("__")
    ev["condition"], ev["method"], ev["variant"] = parts.str[0], parts.str[1], parts.str[2]
    ev["seed"] = parts.str[3].str[1:].astype(int)
    ev = ev[(ev.method == "HH-R") & ev.variant.isin(["default", "post_detection"])]
    ev["scope"] = ev.variant.replace({"default": "full"})
    rows = []
    for c in CONDS:
        for sc in ("full", "post_detection"):
            g = ev[(ev.condition == c) & (ev.scope == sc)]
            per_seed = g.groupby("seed").n_replayed.mean()
            m = boot(per_seed)
            rows.append({"condition": c, "scope": sc, "n_acceptances": len(g), "n_seeds": per_seed.size,
                         "n_replayed_mean": m[0], "ci_lo": m[1], "ci_hi": m[2],
                         "frac_zero": float((g.n_replayed == 0).mean()) if len(g) else np.nan,
                         "frac_one": float((g.n_replayed == 1).mean()) if len(g) else np.nan,
                         "median": float(g.n_replayed.median()) if len(g) else np.nan})
    a = pd.DataFrame(rows)
    a.to_csv(OUT / "0a_n_replayed.csv", index=False)
    L += ["## 0a. Transitions replayed per acceptance (HH-R; Friday `A_replay_scope`, 200 seeds)", "",
          "Mean of per-seed means [95% bootstrap CI over seeds, 10,000 resamples]. All accepted revisions, "
          "including those before the first switch.", "",
          "| condition | scope | acceptances | n_replayed per acceptance | median | fraction replaying 0 | fraction replaying exactly 1 |",
          "|---|---|---|---|---|---|---|"]
    for _, r in a.iterrows():
        L.append(f"| {NICE[r.condition]} | {r.scope} | {r.n_acceptances} | "
                 f"{f((r.n_replayed_mean, r.ci_lo, r.ci_hi), 2)} | {r['median']:.0f} | {r.frac_zero:.3f} | {r.frac_one:.3f} |")
    L += ["", "`post_detection` always replays at least the acceptance round's own transition (it is in the "
          "buffer and inside the detection run), so it never replays 0.", ""]

    # ---- 0b --------------------------------------------------------------
    L += ["## 0b. Per-context contamination, replay_scope = full", "",
          "**Condition parameters** (`himher/experiment/conditions.py`; θ_c = P(counterpart plays R | context c)). "
          f"'Changes' = |Δθ_c| > δ = {DELTA} (HIM's distinctness threshold); a change of exactly δ is listed as "
          "*borderline*.", "",
          "| condition | switch at | before (θ₀, θ₁) | after (θ₀, θ₁) | context 0 | context 1 |", "|---|---|---|---|---|---|"]
    for c in CONDS:
        spec = inst.condition_by_label(c)
        sched = spec.condition.schedule
        if not spec.switch_times:
            m0 = sched.policy_at(0)
            L.append(f"| {NICE[c]} | — | ({m0.theta0}, {m0.theta1}) | — | no switch | no switch |")
            continue
        for s in spec.switch_times:
            b, af = sched.policy_at(s - 1), sched.policy_at(s)
            def lab(x, y):
                d = abs(y - x)
                if d < 1e-12:
                    return "unchanged"
                if abs(d - DELTA) < 1e-9:
                    return f"borderline (Δ = {d:.1f})"
                return f"changes (Δ = {d:.1f})"
            L.append(f"| {NICE[c]} | {s} | ({b.theta0}, {b.theta1}) | ({af.theta0}, {af.theta1}) | "
                     f"{lab(b.theta0, af.theta0)} | {lab(b.theta1, af.theta1)} |")
    per = pd.concat([pd.read_csv(RES / "rebuttal_replay_diag" / "per_acceptance.csv"),
                     pd.read_csv(RES / "rebuttal_mon" / "task0" / "replay_diag_m1" / "per_acceptance.csv")])
    per = per[(per.replay_scope == "full") & per.t_sw.notna() & per.context.astype(str).isin(["0", "1"])]
    rows = []
    for c in CONDS[1:]:
        for cx in ("0", "1"):
            g = per[(per.condition == c) & (per.context.astype(str) == cx)]
            sm = g.groupby("seed")[["pre_share", "ll_pre_accepted", "ll_pre_true", "gap_pre"]].mean()
            row = {"condition": c, "context": cx, "n_acceptances": len(g), "n_with_pre": int((g.n_R_pre > 0).sum())}
            for k in sm.columns:
                m = boot(sm[k])
                row |= {k: m[0], f"{k}_lo": m[1], f"{k}_hi": m[2]}
            rows.append(row)
    b = pd.DataFrame(rows)
    b.to_csv(OUT / "0b_per_context.csv", index=False)
    L += ["", "**Contamination by context** (HH-R, `replay_scope = full`, seeds 0–19; acceptances at or after the "
          "first switch; per-seed means, then mean [95% bootstrap CI] over seeds). ll = mean log p(a | c) per "
          "pre-switch replayed transition in that context; gap_pre = ll_pre_accepted − ll_pre_true. "
          "Sources: `results/rebuttal_replay_diag/per_acceptance.csv` (Within, m2, Recurring) and "
          "`results/rebuttal_mon/task0/replay_diag_m1/per_acceptance.csv` (m1, same code; its re-runs are "
          "bitwise identical to the stored runs and its replay reconstruction matches 282/282 logged acceptances).", "",
          "| condition | context | acceptances (with pre-switch replay in this context) | pre_share | ll_pre_accepted | ll_pre_true | gap_pre |",
          "|---|---|---|---|---|---|---|"]
    for _, r in b.iterrows():
        L.append(f"| {NICE[r.condition]} | {r.context} | {r.n_acceptances} ({r.n_with_pre}) | "
                 + " | ".join(f((r[k], r[f'{k}_lo'], r[f'{k}_hi'])) for k in ("pre_share", "ll_pre_accepted", "ll_pre_true", "gap_pre"))
                 + " |")
    L += ["", "Recurring has three switches; 'pre-switch' is relative to the most recent switch before each "
          "acceptance, so its rows mix the M_L→m2, m2→M_R and M_R→m2 transitions.", ""]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
