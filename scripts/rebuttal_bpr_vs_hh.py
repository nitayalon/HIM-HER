"""BPR+ vs H+H (HH-R, HH-A): reward over the whole episode and the pre-/
post-switch phases, with paired t-tests and Holm-Bonferroni FWER control.

Table 1 (realised reward): per-seed cumulative reward and average reward
per round in each phase, from the stored 200-seed summaries
(orig/summary.csv for HH-R/HH-A, BPR_mdp/summary.csv for BPR+ = the MDP
implementation; the first, Bernoulli BPR+ results are kept in BPR_baseline/).
  whole = rounds 0..T-1, pre = rounds < first switch, post = rounds >=
  first switch (Recurring: first switch at 75, so "post" spans the later
  switches too). Stationary has no switch: whole only.

Table 2 (control-adjusted): the expected reward each agent would get by
best-responding to the model it currently holds, i.e. r* - inference
regret, from decomposition/per_run.csv. This removes H+H's learned
Q-policy and both agents' exploration (BPR+'s epsilon and learning-phase
rounds), so BPR+'s given best-response policies and H+H's learned policy
are compared on inference alone.

Tests: two-sided paired t-tests over seeds (same seed = same contexts,
counterpart draws and switch times for every agent), comparisons BPR+ vs
HH-R and BPR+ vs HH-A in each condition x phase. Cumulative and average
reward differ by the fixed phase length, so one test covers both.
Holm-Bonferroni across all tests of a table (one family per table).

Usage: python scripts/rebuttal_bpr_vs_hh.py
Writes AAMAS 2027/reports/rebuttal_fri_tables/bpr_vs_hh.{md,csv}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "AAMAS 2027" / "results" / "rebuttal_fri"
OUT = ROOT / "AAMAS 2027" / "reports" / "rebuttal_fri_tables"
T = 300
SWITCH = {"stationary": None, "within_library_switch": 150, "outside_library_switch_m1": 150,
          "outside_library_switch_m2": 150, "recurring": 75}
COND_LABEL = {"stationary": "Stationary", "within_library_switch": "Within-library",
              "outside_library_switch_m1": "Outside m1", "outside_library_switch_m2": "Outside m2",
              "recurring": "Recurring"}
METHODS = ("BPR+", "HH-R", "HH-A")
COMPARISONS = (("BPR+", "HH-R"), ("BPR+", "HH-A"))


def phase_len(cond: str, phase: str) -> int:
    s = SWITCH[cond]
    return {"whole": T, "pre": s if s else T, "post": T - s if s else 0}[phase]


# Rounds 0-149 are the same M_L segment with the same seeds in Within-library,
# m1 and m2, so their pre-switch phase is one experiment (asserted below);
# it is reported and tested once, under Within-library.
SHARED_PRE = ("within_library_switch", "outside_library_switch_m1", "outside_library_switch_m2")


def phases(cond: str) -> list[str]:
    if SWITCH[cond] is None:
        return ["whole"]
    if cond in SHARED_PRE[1:]:
        return ["whole", "post"]
    return ["whole", "pre", "post"]


def all_phases(cond: str) -> list[str]:
    return ["whole"] if SWITCH[cond] is None else ["whole", "pre", "post"]


def realised() -> pd.DataFrame:
    s = pd.concat([pd.read_csv(RUNS / "orig" / "summary.csv"),
                   pd.read_csv(RUNS / "BPR_mdp" / "summary.csv")])
    s = s[s.method.isin(METHODS) & (s.variant == "default")]
    rows = []
    for _, r in s.iterrows():
        avg = {"whole": r.return_total, "pre": r.return_pre, "post": r.return_post}
        for ph in all_phases(r.condition):
            rows.append({"method": r.method, "condition": r.condition, "seed": r.seed, "phase": ph,
                         "avg": avg[ph], "cum": avg[ph] * phase_len(r.condition, ph)})
    df = pd.DataFrame(rows)
    # Cross-check: whole-episode cumulative reward equals the stored AUC.
    auc = s.set_index(["method", "condition", "seed"]).auc
    w = df[df.phase == "whole"].set_index(["method", "condition", "seed"]).cum
    assert np.allclose(w.sort_index(), auc.loc[w.index].sort_index(), atol=1e-9)
    check_shared_pre(df)
    return df


def check_shared_pre(df: pd.DataFrame) -> None:
    pre = df[df.phase == "pre"].pivot_table(index=["method", "seed"], columns="condition", values="avg")
    for c in SHARED_PRE[1:]:
        assert np.allclose(pre[c], pre[SHARED_PRE[0]]), c


def control_adjusted() -> pd.DataFrame:
    d = pd.read_csv(RUNS / "decomposition" / "per_run.csv")
    d = d[d.method.isin(METHODS)].copy()
    d["avg"] = d.exp_reward + d.reg_policy + d.reg_exploration  # = r* - inference regret
    rows = []
    for (m, c, seed), g in d.groupby(["method", "condition", "seed"]):
        g = g.set_index("phase")
        if SWITCH[c] is None:
            parts = {"whole": g.loc["pre", "avg"]}
        else:
            pre, post = g.loc["pre", "avg"], g.loc["post", "avg"]
            parts = {"whole": (pre * phase_len(c, "pre") + post * phase_len(c, "post")) / T,
                     "pre": pre, "post": post}
        for ph, a in parts.items():
            rows.append({"method": m, "condition": c, "seed": seed, "phase": ph,
                         "avg": a, "cum": a * phase_len(c, ph)})
    df = pd.DataFrame(rows)
    check_shared_pre(df)
    # H+H's held model depends only on the cascade, which never sees ego
    # actions, so HH-R and HH-A are identical here: keep one "H+H" column.
    key = ["condition", "seed", "phase"]
    r_ = df[df.method == "HH-R"].set_index(key).avg.sort_index()
    a_ = df[df.method == "HH-A"].set_index(key).avg.sort_index()
    assert np.allclose(r_.to_numpy(), a_.to_numpy(), rtol=0, atol=1e-12)
    df = df[df.method != "HH-A"].copy()
    df.loc[df.method == "HH-R", "method"] = "H+H"
    return df


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def tests(df: pd.DataFrame, table: str, comparisons) -> pd.DataFrame:
    out = []
    for c in SWITCH:
        for ph in phases(c):
            sub = df[(df.condition == c) & (df.phase == ph)]
            for a, b in comparisons:
                A = sub[sub.method == a].set_index("seed").avg.sort_index()
                B = sub[sub.method == b].set_index("seed").avg.sort_index()
                assert (A.index == B.index).all() and len(A) == 200
                d = (A - B).to_numpy()
                t, p = stats.ttest_rel(A, B)
                se = d.std(ddof=1) / np.sqrt(len(d))
                q = stats.t.ppf(0.975, len(d) - 1)
                out.append({"table": table, "condition": c, "phase": ph, "comparison": f"{a} − {b}",
                            "mean_diff_avg": d.mean(), "ci_lo": d.mean() - q * se, "ci_hi": d.mean() + q * se,
                            "mean_diff_cum": d.mean() * phase_len(c, ph), "t": t, "df": len(d) - 1,
                            "p_raw": p, "cohen_dz": d.mean() / d.std(ddof=1)})
    res = pd.DataFrame(out)
    res["p_holm"] = holm(res.p_raw.to_numpy())
    res["significant"] = res.p_holm < 0.05
    return res


def describe(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["condition", "phase", "method"])
    return pd.DataFrame({"cum_mean": g.cum.mean(), "cum_se": g.cum.std(ddof=1) / np.sqrt(g.cum.count()),
                         "avg_mean": g.avg.mean(), "avg_se": g.avg.std(ddof=1) / np.sqrt(g.avg.count()),
                         "n": g.avg.count()}).reset_index()


def fmt_p(p: float) -> str:
    return "<1e-300" if p == 0 else (f"{p:.1e}" if p < 1e-3 else f"{p:.3f}")


def render(desc: pd.DataFrame, res: pd.DataFrame, title: str, note: str, methods) -> list[str]:
    lines = [f"## {title}", "", note, "",
             "| condition | phase (rounds) | " + " | ".join(f"{m} cumulative ± SE | {m} average ± SE" for m in methods) + " |",
             "|---|---|" + "---|" * (2 * len(methods))]
    for c in SWITCH:
        for ph in phases(c):
            cells = []
            for m in methods:
                r = desc[(desc.condition == c) & (desc.phase == ph) & (desc.method == m)].iloc[0]
                cells += [f"{r.cum_mean:.1f} ± {r.cum_se:.1f}", f"{r.avg_mean:.3f} ± {r.avg_se:.3f}"]
            ph_label = f"{ph} ({phase_len(c, ph)})" + (" — shared by Within, m1, m2" if ph == "pre" and c in SHARED_PRE else "")
            lines.append(f"| {COND_LABEL[c]} | {ph_label} | " + " | ".join(cells) + " |")
    lines += ["", f"Paired t-tests (df = 199), two-sided; Holm–Bonferroni over all {len(res)} tests in this table. "
              "Δ = first minus second method; positive favours BPR+.", "",
              "| condition | phase | comparison | Δ average [95% CI] | Δ cumulative | t | d_z | p (raw) | p (Holm) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for _, r in res.iterrows():
        star = " *" if r.significant else ""
        ph_label = r.phase + (" (shared)" if r.phase == "pre" and r.condition in SHARED_PRE else "")
        lines.append(f"| {COND_LABEL[r.condition]} | {ph_label} | {r.comparison} | "
                     f"{r.mean_diff_avg:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | {r.mean_diff_cum:+.1f} | "
                     f"{r.t:.2f} | {r.cohen_dz:+.2f} | {fmt_p(r.p_raw)} | {fmt_p(r.p_holm)}{star} |")
    return lines + [""]


def main() -> None:
    real, adj = realised(), control_adjusted()
    res_real = tests(real, "realised", COMPARISONS)
    res_adj = tests(adj, "control_adjusted", (("BPR+", "H+H"),))
    d_real, d_adj = describe(real), describe(adj)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat([d_real.assign(table="realised"), d_adj.assign(table="control_adjusted")]).to_csv(
        OUT / "bpr_vs_hh_descriptives.csv", index=False)
    pd.concat([res_real, res_adj]).to_csv(OUT / "bpr_vs_hh_tests.csv", index=False)
    md = ["# BPR+ vs H+H: reward by phase", "",
          "200 seeds per cell (paper seed list), identical environment draws per seed across agents. "
          "SE = standard error of the mean over seeds. `*` = significant at FWER 0.05 after Holm–Bonferroni.", ""]
    md += render(d_real, res_real, "1. Realised reward",
                 "What each agent actually earned (all exploration and policy learning included).", METHODS)
    md += render(d_adj, res_adj, "2. Control-adjusted reward (inference only)",
                 "Expected reward if each agent best-responded to the model it held that round "
                 "(= r* − inference regret, true-model expectation, so counterpart noise is removed). "
                 "Removes H+H's Q-learning and both agents' exploration (BPR+: ε and learning-phase rounds). "
                 "BPR+'s belief-weighted choice is exactly the best response to its posterior-predictive model. "
                 "HH-R and HH-A hold identical models every round (the cascade never sees ego actions), "
                 "so they form one column, H+H.", ("BPR+", "H+H"))
    (OUT / "bpr_vs_hh.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote AAMAS 2027/reports/rebuttal_fri_tables/bpr_vs_hh.md (+ _descriptives.csv, _tests.csv)")


if __name__ == "__main__":
    main()
