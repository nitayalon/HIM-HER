"""Inference-only comparison: cumulative inference regret before / after the
first switch and in total, H+H vs each comparison method; paired t-tests,
one Holm-Bonferroni family.

Inference regret per round = r*(x) - E[r | BR(m_t)] under the true model
(scripts/rebuttal_decompose.py): the cost of the *held model* alone, with
H+H's Q-policy and every agent's exploration removed. HH-R and HH-A hold
identical models every round (the cascade never sees ego actions; asserted
here), so they form one arm, "H+H". Cumulative = per-round mean x rounds.
The pre-switch phase is identical for Within/m1/m2 and is tested once.

Usage: python scripts/rebuttal_inference_tests.py [--vs WB-k20 DB-0.90 ...]
Reads  AAMAS 2027/results/rebuttal_fri/decomposition/per_run.csv
Writes AAMAS 2027/reports/rebuttal_fri_tables/inference_tests.{md,csv}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from himher.experiment import instrumentation as inst  # noqa: E402

PER_RUN = ROOT / "AAMAS 2027" / "results" / "rebuttal_fri" / "decomposition" / "per_run.csv"
OUT = ROOT / "AAMAS 2027" / "reports" / "rebuttal_fri_tables"
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
    ap.add_argument("--vs", nargs="+", default=["WB-k20", "DB-0.90", "DB-0.95", "DB-0.98"])
    vs = ap.parse_args().vs
    d = pd.read_csv(PER_RUN)
    d = d[d.phase.isin(["pre", "post"])]
    hr = d[d.method == "HH-R"].set_index(["condition", "seed", "phase"]).reg_inference
    ha = d[d.method == "HH-A"].set_index(["condition", "seed", "phase"]).reg_inference
    assert np.allclose(hr.sort_index(), ha.loc[hr.index].sort_index(), atol=1e-12), "HH-R/HH-A models differ"
    d = d[d.method.isin(["HH-R", *vs])].replace({"method": {"HH-R": "H+H"}})
    methods = ["H+H", *vs]

    rows = []
    for _, r in d.iterrows():
        spec = inst.condition_by_label(r.condition)
        T, fs = spec.condition.episode_length, spec.first_switch
        n = (fs or T) if r.phase == "pre" else T - fs
        rows.append({"method": r.method, "condition": r.condition, "seed": r.seed,
                     "phase": ("total" if fs is None else "before") if r.phase == "pre" else "after",
                     "cum": r.reg_inference * n, "side": r.model_side_correct})
    long = pd.DataFrame(rows)
    tot = (long[long.phase.isin(["before", "after"])].groupby(["method", "condition", "seed"])
               .cum.sum().reset_index().assign(phase="total"))
    long = pd.concat([long, tot], ignore_index=True)

    desc = (long.groupby(["condition", "phase", "method"]).cum
                .agg(mean="mean", se=lambda x: x.std(ddof=1) / np.sqrt(len(x))).reset_index())
    tests = []
    for c in COND_ORDER:
        for ph in ("before", "after", "total"):
            if c == "stationary" and ph != "total":
                continue
            if ph == "before" and c in SHARED_PRE[1:]:
                continue
            w = long[(long.condition == c) & (long.phase == ph)].pivot(index="seed", columns="method", values="cum")
            if w.empty:
                continue
            for b in vs:
                diff = w["H+H"] - w[b]
                t, p = stats.ttest_rel(w["H+H"], w[b])
                se = diff.std(ddof=1) / np.sqrt(len(diff))
                ci = stats.t.interval(0.95, len(diff) - 1, loc=diff.mean(), scale=se)
                tests.append({"condition": c, "phase": ph, "comparison": f"H+H − {b}", "diff": diff.mean(),
                              "ci_lo": ci[0], "ci_hi": ci[1], "t": t, "dz": diff.mean() / diff.std(ddof=1),
                              "p_raw": p})
    tests = pd.DataFrame(tests)
    ok = tests.p_raw.notna()  # identical arms (zero variance) are not tests
    tests["p_holm"] = np.nan
    tests.loc[ok, "p_holm"] = holm(tests.p_raw[ok].to_numpy())
    tests["significant"] = tests.p_holm < 0.05
    tests.to_csv(OUT / "inference_tests.csv", index=False)

    L = ["# Inference only: cumulative inference regret, H+H vs " + ", ".join(vs), "",
         "Inference regret = r*(x) − E[r | best response to the held model], under the true model "
         "(counterpart noise, H+H's Q-policy and all exploration removed). Cumulative over the phase; "
         "**lower = better inference**. HH-R ≡ HH-A here (identical held models every round), shown as H+H. "
         "Mean ± SE over 200 seeds. Before = rounds before the first switch (150; Recurring 75).", "",
         "| condition | phase | " + " | ".join(methods) + " |", "|---|---|" + "---|" * len(methods)]
    for c in COND_ORDER:
        for ph in ("before", "after", "total"):
            g = desc[(desc.condition == c) & (desc.phase == ph)].set_index("method")
            if g.empty or (ph == "before" and c in SHARED_PRE[1:]):
                continue
            label = "Within / m1 / m2" if (ph == "before" and c in SHARED_PRE) else NICE[c]
            L.append(f"| {label} | {ph} | " + " | ".join(f"{g.loc[m, 'mean']:.2f} ± {g.loc[m, 'se']:.2f}"
                                                         for m in methods) + " |")
    L += ["", f"Paired t-tests (df = 199), two-sided; Holm–Bonferroni over all {int(ok.sum())} tests "
          "(`—`: both arms have zero inference regret in every seed, nothing to test). "
          "Δ = H+H − method; **negative = H+H infers better**. `*` = significant at FWER 0.05.", "",
          "| condition | phase | comparison | Δ cumulative inference regret [95% CI] | t | d_z | p (Holm) |",
          "|---|---|---|---|---|---|---|"]
    for _, r in tests.iterrows():
        lab = "Within / m1 / m2" if (r.phase == "before" and r.condition in SHARED_PRE) else NICE[r.condition]
        tval = "—" if np.isnan(r.t) else f"{r.t:.2f}"
        dz = "—" if np.isnan(r.dz) else f"{r.dz:+.2f}"
        pv = "—" if np.isnan(r.p_holm) else f"{r.p_holm:.2g}{' *' if r.significant else ''}"
        L.append(f"| {lab} | {r.phase} | {r.comparison} | {r['diff']:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] | "
                 f"{tval} | {dz} | {pv} |")
    (OUT / "inference_tests.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote AAMAS 2027/reports/rebuttal_fri_tables/inference_tests.md")


if __name__ == "__main__":
    main()
