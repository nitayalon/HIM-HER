"""Cumulative reward before / after the first switch and in total (before +
after), per condition, for BPR+, HH-R and HH-A; all pairwise paired t-tests
with one Holm-Bonferroni family.

Reads the stored 200-seed summaries (orig, BPR_mdp); read-only.
The pre-switch phase of Within, m1 and m2 is identical per seed (same
environment until t = 150), so it is tested once.

Usage: python scripts/rebuttal_cumulative_by_condition.py
Writes AAMAS 2027/reports/rebuttal_fri_tables/cumulative_by_condition.{md,csv}
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from himher.experiment import instrumentation as inst  # noqa: E402

RES = ROOT / "AAMAS 2027" / "results" / "rebuttal_fri"
OUT = ROOT / "AAMAS 2027" / "reports" / "rebuttal_fri_tables"
HH = ("HH-R", "HH-A")
WBDB = ("WB-k20", "DB-0.90", "DB-0.95", "DB-0.98")
# set -> (methods, stored experiments, comparisons, output stem, title)
SETS = {
    "bpr": (("BPR+", *HH), ("orig", "BPR_mdp"), list(itertools.combinations(("BPR+", *HH), 2)),
            "cumulative_by_condition", "BPR+ vs HH-R vs HH-A"),
    "wbdb": ((*HH, *WBDB), ("orig", "B_baselines"), [(h, b) for h in HH for b in WBDB],
             "cumulative_wbdb_vs_hh", "WB / DB vs HH-R / HH-A"),
}
COND_ORDER = ("stationary", "within_library_switch", "outside_library_switch_m1",
              "outside_library_switch_m2", "recurring")
NICE = {"stationary": "Stationary", "within_library_switch": "Within-library",
        "outside_library_switch_m1": "Outside m1", "outside_library_switch_m2": "Outside m2",
        "recurring": "Recurring"}
SHARED_PRE = ("within_library_switch", "outside_library_switch_m1", "outside_library_switch_m2")


def holm(p: np.ndarray) -> np.ndarray:
    order, m, adj, run = np.argsort(p), len(p), np.empty(len(p)), 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[k]))
        adj[k] = run
    return adj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=sorted(SETS), default="bpr")
    METHODS, exps, pairs, stem, title = SETS[ap.parse_args().set]
    s = pd.concat([pd.read_csv(RES / e / "summary.csv") for e in exps])
    s = s[(s.variant == "default") & s.method.isin(METHODS)].drop_duplicates(["method", "condition", "seed"])
    rows = []
    for _, r in s.iterrows():
        spec = inst.condition_by_label(r.condition)
        T, fs = spec.condition.episode_length, spec.first_switch
        if fs is None:
            phases = {"total": r.return_total * T}
        else:
            pre, post = r.return_pre * fs, r.return_post * (T - fs)
            phases = {"before": pre, "after": post, "total": pre + post}
            assert abs(phases["total"] - r.auc) < 1e-6
        for ph, v in phases.items():
            rows.append({"method": r.method, "condition": r.condition, "seed": r.seed, "phase": ph, "cum": v})
    long = pd.DataFrame(rows)
    long.to_csv(OUT / f"{stem}_per_seed.csv", index=False)

    desc = (long.groupby(["condition", "phase", "method"]).cum
                .agg(mean="mean", se=lambda x: x.std(ddof=1) / np.sqrt(len(x)), n="size").reset_index())
    desc.to_csv(OUT / f"{stem}.csv", index=False)

    tests = []
    for c in COND_ORDER:
        for ph in ("before", "after", "total"):
            if c == "stationary" and ph != "total":
                continue
            if ph == "before" and c in SHARED_PRE[1:]:
                continue
            w = long[(long.condition == c) & (long.phase == ph)].pivot(index="seed", columns="method", values="cum")
            for a, b in pairs:
                d = w[a] - w[b]
                t, p = stats.ttest_rel(w[a], w[b])
                ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=d.std(ddof=1) / np.sqrt(len(d)))
                tests.append({"condition": c, "phase": ph, "comparison": f"{a} - {b}", "diff": d.mean(),
                              "ci_lo": ci[0], "ci_hi": ci[1], "t": t, "df": len(d) - 1,
                              "dz": d.mean() / d.std(ddof=1), "p_raw": p})
    tests = pd.DataFrame(tests)
    tests["p_holm"] = holm(tests.p_raw.to_numpy())
    tests["significant"] = tests.p_holm < 0.05
    tests.to_csv(OUT / f"{stem}_tests.csv", index=False)

    L = [f"# Cumulative reward by condition: {title}", "",
         "200 seeds, identical environment draws per seed. Cumulative = sum of rewards in the phase; "
         "total = before + after (= AUC). Mean ± SE over seeds. Before = rounds before the first switch "
         "(150; Recurring 75); after = the rest. Stationary has no switch (total only, 300 rounds).", "",
         "| condition | phase | " + " | ".join(f"{m}" for m in METHODS) + " | best |", "|---|---|" + "---|" * (len(METHODS) + 1)]
    for c in COND_ORDER:
        for ph in ("before", "after", "total"):
            g = desc[(desc.condition == c) & (desc.phase == ph)].set_index("method")
            if g.empty:
                continue
            cells = [f"{g.loc[m, 'mean']:.1f} ± {g.loc[m, 'se']:.1f}" for m in METHODS]
            L.append(f"| {NICE[c]} | {ph} | " + " | ".join(cells) + f" | {g['mean'].idxmax()} |")
    L += ["", f"Paired t-tests (df = 199), two-sided; Holm–Bonferroni over all {len(tests)} tests. "
          "The shared before-phase of Within/m1/m2 is tested once (listed under Within-library). "
          "`*` = significant at FWER 0.05.", "",
          "| condition | phase | comparison | Δ cumulative [95% CI] | t | d_z | p (Holm) |", "|---|---|---|---|---|---|---|"]
    for _, r in tests.iterrows():
        L.append(f"| {NICE[r.condition]} | {r.phase} | {r.comparison.replace(' - ', ' − ')} | {r['diff']:+.1f} "
                 f"[{r.ci_lo:+.1f}, {r.ci_hi:+.1f}] | {r.t:.2f} | {r.dz:+.2f} | "
                 f"{r.p_holm:.2g}{' *' if r.significant else ''} |")
    (OUT / f"{stem}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote AAMAS 2027/reports/rebuttal_fri_tables/{stem}.md")


if __name__ == "__main__":
    main()
