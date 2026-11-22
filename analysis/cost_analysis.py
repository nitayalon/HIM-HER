"""Task C4: cost analysis (standalone; pandas + numpy + matplotlib only).

Inputs (AAMAS 2027/results/rebuttal_cost/):
  cost_runs.csv            C3, one row per run (required)
  hh_phase_split.csv       H+H counters split by phase (T2; optional)
  full/cost_per_step_merged.csv   cumulative ev_total per step, 5 seeds (F3; optional)
  scaling_runs.csv         C6 (F4 + regret table; optional)
Outputs: AAMAS 2027/results/rebuttal_cost/analysis/
  T1_cost_per_1000.{csv,md}, T2_hh_phase.{csv,md}, T3_complexity.md,
  T4_crossover.{csv,md}, U_c.csv, F1..F4 .pdf/.png, T5_scaling_regret.md

Conventions
  * "per 1,000 interactions" = episode total * 1000 / T (T = 300).
  * CIs: 95% percentile bootstrap over seeds, 10,000 resamples, fixed
    index matrix (seed 20261122); paired quantities resample seeds jointly.
  * Regret axis (F1): post-switch regret (realised, summed over rounds
    >= first switch). Stationary has no switch; there the panel shows
    episode regret = 0.9 * T - AUC (optimal expected reward is 0.9 in
    every round under M_L), i.e. the same quantity over the whole episode.
  * U_c = AUC - c * ev_total (both per episode).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "AAMAS 2027" / "results" / "rebuttal_cost"
OUT = BASE / "analysis"
N_BOOT, BOOT_SEED = 10_000, 20261122
T = 300
METHODS = ["HH-R", "HH-A", "FB", "OB", "CB", "WB-k20", "DB-0.90", "DB-0.95", "DB-0.98", "BOCPD", "BPR+", "BPR+bern"]
HH = ["HH-R", "HH-A"]
BASELINES = [m for m in METHODS if m not in HH]
CONDS = ["stationary", "within_library_switch", "outside_library_switch_m1",
         "outside_library_switch_m2", "recurring"]
LABEL = {"stationary": "Stationary", "within_library_switch": "Within-library",
         "outside_library_switch_m1": "Outside m1", "outside_library_switch_m2": "Outside m2",
         "recurring": "Recurring"}
C_GRID = np.concatenate([[0.0], np.logspace(-7, -1, 41)])
# Okabe-Ito (colourblind-safe) + marker per method so 12 methods stay distinct.
OKABE = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
STYLE = {"HH-R": ("#D55E00", "*"), "HH-A": ("#CC79A7", "X"), "FB": ("#56B4E9", "o"),
         "OB": ("#009E73", "s"), "CB": ("#B8A000", "D"), "WB-k20": ("#0072B2", "^"),
         "DB-0.90": ("#000000", "v"), "DB-0.95": ("#555555", "<"), "DB-0.98": ("#999999", ">"),
         "BOCPD": ("#E69F00", "p"), "BPR+": ("#882255", "h"), "BPR+bern": ("#44AA99", "P")}
assert set(STYLE) == set(METHODS) and len({v[0] for v in STYLE.values()}) == len(STYLE)

_IDX: dict[int, np.ndarray] = {}


def boot_idx(n: int) -> np.ndarray:
    if n not in _IDX:
        _IDX[n] = np.random.default_rng(BOOT_SEED).integers(0, n, size=(N_BOOT, n))
    return _IDX[n]


def mean_ci(x: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    if np.all(np.isnan(x)):
        return (np.nan, np.nan, np.nan)
    x = x[~np.isnan(x)]
    bs = x[boot_idx(len(x))].mean(axis=1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def fmt(m: float, lo: float | None = None, hi: float | None = None, d: int = 1) -> str:
    if m is None or np.isnan(m):
        return "—"
    s = f"{m:,.{d}f}"
    return s if lo is None else f"{s} [{lo:,.{d}f}, {hi:,.{d}f}]"


def savefig(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def load() -> pd.DataFrame:
    df = pd.read_csv(BASE / "cost_runs.csv")
    wall_cols = [c for c in df.columns if c.startswith("wall_") and c.endswith("_s") and c != "wall_run_s"]
    df["wall_agent_ms"] = df[wall_cols].sum(axis=1) * 1e3
    df["ev_total_incl_cache"] = df.ev_total + df.cache_hits
    df["regret_axis"] = np.where(df.condition == "stationary", 0.9 * df["T"] - df.auc, df.regret_post)
    return df


# ------------------------------------------------------------------ T1 ----
T1_METRICS = ["ev_total", "ev_monitor", "ev_revision", "ev_revision_him", "ev_voc", "ev_decision",
              "ev_relabel", "cache_hits", "ev_total_incl_cache", "pol_updates", "pol_samples_replayed",
              "n_revision_triggers", "n_accept_reuse", "n_accept_merge", "n_accept_construct",
              "n_voc_declined", "wall_agent_ms"]


def table_t1(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in CONDS:
        for m in METHODS:
            g = df[(df.condition == c) & (df.method == m)]
            for k in T1_METRICS:
                mu, lo, hi = mean_ci(g[k].to_numpy(float) * 1000 / g["T"].to_numpy(float))
                rows.append({"condition": c, "method": m, "metric": f"{k}_per_1000", "mean": mu,
                             "ci_lo": lo, "ci_hi": hi, "n_seeds": len(g)})
    t1 = pd.DataFrame(rows)
    t1.to_csv(OUT / "T1_cost_per_1000.csv", index=False)
    md = ["# T1 — cost per 1,000 interactions",
          "",
          "Mean [95% bootstrap CI] over seeds for `ev_total` and wall-clock. Other columns show the mean only;",
          "every CI is in `T1_cost_per_1000.csv`.",
          "",
          "- *ev* = model evaluations.",
          "- *cache* = reuses of an already-computed evaluation (BOCPD across run lengths).",
          "- *acc r/m/c* = accepted reuse/merge/construct (BPR+: learned models under c).",
          "- *wall* = agent-attributed wall-clock ms per 1,000 steps, from serial runs.",
          ""]
    for c in CONDS:
        md += [f"**{LABEL[c]}**", "",
               "| method | ev_total | monitor | revision (HIM) | decision | relabel | cache | pol updates (replayed) | triggers | acc r/m/c | wall ms |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for m in METHODS:
            g = t1[(t1.condition == c) & (t1.method == m)].set_index("metric")
            v = lambda k: g.loc[f"{k}_per_1000"]  # noqa: E731
            acc = "/".join(fmt(v(k)["mean"], d=1) for k in ("n_accept_reuse", "n_accept_merge", "n_accept_construct"))
            md.append(
                f"| {m} | {fmt(*v('ev_total')[['mean', 'ci_lo', 'ci_hi']], d=0)} | {fmt(v('ev_monitor')['mean'], d=0)} | "
                f"{fmt(v('ev_revision')['mean'], d=0)} ({fmt(v('ev_revision_him')['mean'], d=0)}) | "
                f"{fmt(v('ev_decision')['mean'], d=0)} | {fmt(v('ev_relabel')['mean'], d=0)} | "
                f"{fmt(v('cache_hits')['mean'], d=0)} | {fmt(v('pol_updates')['mean'], d=0)} "
                f"({fmt(v('pol_samples_replayed')['mean'], d=0)}) | {fmt(v('n_revision_triggers')['mean'], d=1)} | "
                f"{acc} | {fmt(*v('wall_agent_ms')[['mean', 'ci_lo', 'ci_hi']], d=1)} |")
        md.append("")
    (OUT / "T1_cost_per_1000.md").write_text("\n".join(md), encoding="utf-8")
    return t1


# ------------------------------------------------------------------ T2 ----
def table_t2() -> None:
    p = BASE / "hh_phase_split.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    cats = ["ev_monitor", "ev_revision", "ev_total", "pol_updates", "pol_samples_replayed"]
    for ph in ("pre", "post", "w50", "rest"):
        d[f"ev_total_{ph}"] = sum(d[f"ev_{c}_{ph}"] for c in ("monitor", "revision", "voc", "decision", "relabel"))
    rows = []
    for m in HH:
        for c in CONDS:
            g = d[(d.method == m) & (d.condition == c)]
            for k in cats:
                for ph in ("pre", "post", "w50", "rest"):
                    steps = g[f"steps_{ph}"].to_numpy(float)
                    vals = np.where(steps > 0, g[f"{k}_{ph}"].to_numpy(float) * 1000 / np.maximum(steps, 1), np.nan)
                    mu, lo, hi = mean_ci(vals)
                    rows.append({"method": m, "condition": c, "metric": k, "phase": ph, "per_1000_steps": mu,
                                 "ci_lo": lo, "ci_hi": hi})
    t2 = pd.DataFrame(rows)
    t2.to_csv(OUT / "T2_hh_phase.csv", index=False)
    md = ["# T2 — H+H cost by phase (per 1,000 steps *within* each phase)",
          "",
          "Mean [95% CI] over 200 seeds.",
          "",
          "- pre/post = before / after the first switch.",
          "- w50 = the 50 steps after *each* switch; rest = all other steps.",
          "- Stationary has no switch, so post and w50 are empty.",
          "- Recurring has switches at 75/150/225, so w50 covers 150 of its 300 steps.",
          ""]
    for k in cats:
        md += [f"**{k}**", "", "| method | condition | pre | post | w50 | rest |", "|---|---|---|---|---|---|"]
        for m in HH:
            for c in CONDS:
                g = t2[(t2.method == m) & (t2.condition == c) & (t2.metric == k)].set_index("phase")
                cells = [fmt(g.loc[ph, "per_1000_steps"], g.loc[ph, "ci_lo"], g.loc[ph, "ci_hi"], d=0)
                         for ph in ("pre", "post", "w50", "rest")]
                md.append(f"| {m} | {LABEL[c]} | " + " | ".join(cells) + " |")
        md.append("")
    (OUT / "T2_hh_phase.md").write_text("\n".join(md), encoding="utf-8")


# ------------------------------------------------------------------ T3 ----
T3 = """# T3 — analytic complexity, as implemented (from the C0 inventory)

Notation:
- k = 20, the window; h = the current window length, h ≤ k.
- |L| = the current library or hypothesis set. For H+H, lib = L₀ plus up to 20 displaced incumbents.
- n_c = candidates tested per trigger (≤ 3).
- C = the comparison set, {incumbent} ∪ lib \\ {incumbent, candidate}.
- R_t = live run lengths (t + 1; no pruning).
- |O| = BPR+'s strategy library, which grows by one per learned model; S = 2 states (contexts), A = 2 actions; n_VI = value-iteration sweeps (γ = 0.9).
- BPR+ performance-table entries (μ, σ) are closed-form: one entry costs S units, one VI sweep S·A units, one Gaussian density 1 unit.
- Unit: one model evaluation = p(a | x, m) for one model on one observation.

| method | monitoring / step | decision / step | per revision trigger | policy updates |
|---|---|---|---|---|
| HH-R | h ≤ k: rescoring the whole window under the incumbent (no incremental update) | 0 (Q lookup) | \\|L\\|·h (reuse retrieval) + C(\\|L\\|,2)·21·h (merge grid; skipped if reuse accepted) + Σ over candidates [(1+\\|C\\|)·h + 4·\\|C\\|] (HIM log-likelihoods + TV distinctness); construct is closed-form (0) | 1/step + up to k replayed per acceptance |
| HH-A | same as HH-R | 0 | same as HH-R | 1/step (buffer discarded on acceptance) |
| FB / OB | \\|L\\| (3; OB: 4 where the true model is added) | \\|L\\| | — | 0 |
| CB | 1 (analytic; a conjugate count update, though the code rescans the history, O(t)) | 1 (analytic) | — | 0 |
| WB-k20 / DB-γ | 1 (analytic) | 1 (analytic) | — | 0 |
| BOCPD | \\|L\\| evaluations + R_t·\\|L\\| reuses (`cache_hits`), O(R_t·\\|L\\|) arithmetic | \\|L\\| (+ O(R_t·\\|L\\|) softmax) | — (no explicit trigger) | 0 |
| BPR+ (MDP) | \\|O\\| Gaussian densities (0 during learning rounds) | 2·\\|O\\|² (probability of improvement over the \\|O\\|×\\|O\\| performance table; 0 on ε and learning rounds) | count-based MDP fit, value iteration (n_VI+1)·S·A, new performance-table row and column (2\\|O\\|−1)·S; after 20 random rounds; the policy is computed, not learned | 0 |
| BPR+bern | \\|O\\| (0 during learning rounds) | \\|O\\|·(\\|O\\|+1), every round (policy selection) | 0 evaluations: closed-form model fit after 20 random rounds | 0 |

So the per-step monitoring cost is O(k) for H+H, O(|L|) for FB, OB and both BPR+ variants (BPR+ adds O(|O|²) per decision), O(1) for CB, WB and DB, and O(R_t·|L|) for BOCPD. The revision cost is O(|L|²·21·k) for H+H (the merge grid dominates) and is paid only on detected rounds.
"""


# ------------------------------------------------------------------ T4 ----
def crossover(d_auc: np.ndarray, d_ev: np.ndarray) -> tuple[str, float]:
    a, e = float(np.mean(d_auc)), float(np.mean(d_ev))
    if e < 0 and a >= 0:
        return "HH dominates", 0.0
    if e < 0 and a < 0:
        return "c* (HH better for c > c*)", a / e
    if e >= 0 and a > 0:
        return "c_max (HH better for c < c_max)", a / e
    return "B dominates", np.nan


def table_t4(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hh in HH:
        for b in BASELINES:
            for c in CONDS:
                A = df[(df.method == hh) & (df.condition == c)].set_index("seed").sort_index()
                B = df[(df.method == b) & (df.condition == c)].set_index("seed").sort_index()
                assert (A.index == B.index).all()
                da = (A.auc - B.auc).to_numpy(float)
                de = (A.ev_total - B.ev_total).to_numpy(float)
                case, val = crossover(da, de)
                idx = boot_idx(len(da))
                ba, be = da[idx].mean(1), de[idx].mean(1)
                cases = np.select([(be < 0) & (ba >= 0), (be < 0) & (ba < 0), (be >= 0) & (ba > 0)],
                                  ["HH dominates", "c* (HH better for c > c*)", "c_max (HH better for c < c_max)"],
                                  "B dominates")
                same = cases == case
                vals = np.where(same, np.divide(ba, be, out=np.zeros_like(ba), where=be != 0), np.nan)
                lo = hi = np.nan
                if case.startswith("c") and same.any():
                    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
                rows.append({"hh": hh, "baseline": b, "condition": c, "d_auc": da.mean(), "d_ev": de.mean(),
                             "case": case, "value": val, "ci_lo": lo, "ci_hi": hi,
                             **{f"share_{k}": float((cases == k).mean()) for k in
                                ("HH dominates", "c* (HH better for c > c*)", "c_max (HH better for c < c_max)",
                                 "B dominates")}})
    t4 = pd.DataFrame(rows)
    t4.to_csv(OUT / "T4_crossover.csv", index=False)
    md = ["# T4 — cost-adjusted utility crossover (H+H vs each baseline, paired seeds)",
          "",
          "ΔAUC = AUC_HH − AUC_B (reward per episode); ΔEV = ev_total_HH − ev_total_B (evaluations per episode). Both are seed-wise means.",
          "",
          "| case | when | what is reported |",
          "|---|---|---|",
          "| *HH dominates* | ΔEV < 0, ΔAUC ≥ 0 | c* = 0 |",
          "| *c\\** | ΔEV < 0, ΔAUC < 0 | H+H has the higher U_c for every c > c* |",
          "| *c_max* | ΔEV ≥ 0, ΔAUC > 0 | H+H has the higher U_c only for c < c_max |",
          "| *B dominates* | ΔEV ≥ 0, ΔAUC ≤ 0 | no crossover |",
          "",
          "The CI is the 2.5–97.5% range of the bootstrap values among resamples in the same case. The shares are the fractions of the 10,000 resamples in each case: HH dominates / c* / c_max / B dominates.",
          ""]
    for hh in HH:
        md += [f"**{hh}**", "", "| baseline | condition | ΔAUC | ΔEV | case | value [95% CI] | shares |",
               "|---|---|---|---|---|---|---|"]
        for _, r in t4[t4.hh == hh].iterrows():
            val = "—" if np.isnan(r.value) else f"{r.value:.3g} [{r.ci_lo:.3g}, {r.ci_hi:.3g}]"
            if r.case == "HH dominates":
                val = "0"
            shares = " / ".join(f"{r[k]:.2f}" for k in ("share_HH dominates", "share_c* (HH better for c > c*)",
                                                        "share_c_max (HH better for c < c_max)", "share_B dominates"))
            md.append(f"| {r.baseline} | {LABEL[r.condition]} | {r.d_auc:+.1f} | {r.d_ev:+,.0f} | {r.case} | {val} | {shares} |")
        md.append("")
    (OUT / "T4_crossover.md").write_text("\n".join(md), encoding="utf-8")
    return t4


# ------------------------------------------------------------------ figs --
def pareto_front(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    order = np.lexsort((y, x))
    front, best = [], np.inf
    for i in order:
        if y[i] < best:
            front.append(i)
            best = y[i]
    return np.array(front)


def fig_f1(df: pd.DataFrame) -> pd.DataFrame:
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), sharex=True)
    pts = []
    for ax, c in zip(axes, CONDS):
        xs, ys = [], []
        for m in METHODS:
            g = df[(df.condition == c) & (df.method == m)]
            x = mean_ci(g.ev_total.to_numpy(float) * 1000 / T)
            y = mean_ci(g.regret_axis.to_numpy(float))
            col, mk = STYLE[m]
            ax.errorbar(x[0], y[0], xerr=[[x[0] - x[1]], [x[2] - x[0]]], yerr=[[y[0] - y[1]], [y[2] - y[0]]],
                        fmt=mk, color=col, ms=8 if m in HH else 6, capsize=2, label=m,
                        mec="black" if m in HH else col)
            xs.append(x[0]); ys.append(y[0])
            pts.append({"condition": c, "method": m, "ev_per_1000": x[0], "ev_lo": x[1], "ev_hi": x[2],
                        "regret": y[0], "regret_lo": y[1], "regret_hi": y[2]})
        xs, ys = np.array(xs), np.array(ys)
        f = pareto_front(xs, ys)
        fx, fy = xs[f], ys[f]
        ax.step(np.r_[fx, fx[-1] * 3], np.r_[fy, fy[-1]], where="post", color="grey", lw=1, ls="--", zorder=0)
        ax.set_xscale("log")
        ax.set_xlabel("model evaluations per 1,000 interactions")
        ax.text(0.03, 0.97, LABEL[c], transform=ax.transAxes, va="top", fontsize=10)
        for i in f:
            pts[-len(METHODS) + i]["on_front"] = True
    axes[0].set_ylabel("post-switch regret (Stationary: episode regret)")
    axes[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    savefig(fig, "F1_pareto")
    p = pd.DataFrame(pts)
    p["on_front"] = p["on_front"].fillna(False).astype(bool) if "on_front" in p else False
    p.to_csv(OUT / "F1_points.csv", index=False)
    return p


def fig_f2(df: pd.DataFrame, t4: pd.DataFrame) -> None:
    rows = []
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2), sharex=True)
    cpos = C_GRID[1:]
    for ax, c in zip(axes, CONDS):
        for m in METHODS:
            g = df[(df.condition == c) & (df.method == m)]
            u = np.array([np.mean(g.auc - cc * g.ev_total) for cc in C_GRID])
            rows += [{"condition": c, "method": m, "c": cc, "U_c_mean": uu} for cc, uu in zip(C_GRID, u)]
            col, mk = STYLE[m]
            ax.plot(cpos, u[1:], color=col, lw=2.2 if m in HH else 1.2, label=m,
                    ls="-" if m in HH else "--")
        for _, r in t4[(t4.hh == "HH-R") & (t4.condition == c)].iterrows():
            if r.case.startswith("c") and cpos[0] <= r.value <= cpos[-1]:
                g = df[(df.condition == c) & (df.method == "HH-R")]
                ax.plot(r.value, np.mean(g.auc - r.value * g.ev_total), "o", mfc="none", mec="black", ms=7)
        ax.set_xscale("log")
        ax.set_xlabel("c (reward units per model evaluation)")
        ax.text(0.03, 0.03, LABEL[c], transform=ax.transAxes, fontsize=10)
    axes[0].set_ylabel("mean U_c = AUC − c·ev_total")
    axes[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    savefig(fig, "F2_utility_vs_c")
    pd.DataFrame(rows).to_csv(OUT / "U_c.csv", index=False)


def fig_f3() -> None:
    p = BASE / "full" / "cost_per_step_merged.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    show = ["HH-R", "HH-A", "CB", "FB", "BOCPD", "BPR+"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, c in zip(axes, ("within_library_switch", "outside_library_switch_m2")):
        for m in show:
            g = d[(d.condition == c) & (d.method == m)]
            piv = g.pivot(index="t", columns="seed", values="cum_ev_total")
            col, _ = STYLE[m]
            for s in piv.columns:
                ax.plot(piv.index, piv[s], color=col, lw=0.4, alpha=0.4)
            ax.plot(piv.index, piv.mean(axis=1), color=col, lw=2, label=m)
        ax.axvline(150, color="grey", ls=":", lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("round t")
        ax.text(0.03, 0.97, LABEL[c], transform=ax.transAxes, va="top")
    axes[0].set_ylabel("cumulative model evaluations")
    axes[1].legend(fontsize=8, loc="lower right")
    savefig(fig, "F3_cumulative_cost")


def fig_f4() -> None:
    p = BASE / "scaling_runs.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d["regret_axis"] = np.where(d.condition == "stationary", 0.9 * d["T"] - d.auc, d.regret_post)
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, c in zip(axes, ("stationary", "within_library_switch")):
        for m in ["HH-R", "FB", "BPR+", "BOCPD", "CB"]:
            g = d[(d.condition == c) & (d.method == m)]
            if g.empty:
                continue
            sizes = sorted(g.L_size.unique())
            stats = [mean_ci(g[g.L_size == L].ev_total.to_numpy(float) * 1000 / T) for L in sizes]
            mu = np.array([s[0] for s in stats])
            col, mk = STYLE[m]
            ax.errorbar(sizes, mu, yerr=[mu - [s[1] for s in stats], [s[2] for s in stats] - mu],
                        color=col, marker=mk, capsize=2, label=m)
            for L, s in zip(sizes, stats):
                rg = mean_ci(g[g.L_size == L].regret_axis.to_numpy(float))
                rows.append({"condition": c, "method": m, "L_size": L, "ev_per_1000": s[0],
                             "ev_lo": s[1], "ev_hi": s[2], "regret": rg[0], "regret_lo": rg[1], "regret_hi": rg[2],
                             "n_seeds": int((g.L_size == L).sum())})
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("library size |L|")
        ax.text(0.03, 0.97, LABEL[c], transform=ax.transAxes, va="top")
    axes[0].set_ylabel("model evaluations per 1,000 interactions")
    axes[1].legend(fontsize=8)
    savefig(fig, "F4_scaling")
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "T5_scaling.csv", index=False)
    md = ["# Scaling — regret and cost vs |L|",
          "",
          "Mean [95% CI] over seeds. Regret is post-switch regret for Within-library and episode regret for Stationary.",
          "`—`: not run. BPR+ at |L| = 128 is infeasible with its frozen belief floor of 0.01, which must be below 1/|L|.",
          ""]
    for c in ("stationary", "within_library_switch"):
        md += [f"**{LABEL[c]}**", "", "| method | " + " | ".join(f"|L|={L}" for L in (2, 8, 32, 128)) + " |",
               "|---|" + "---|" * 4]
        for m in ["HH-R", "FB", "BPR+", "BOCPD", "CB"]:
            cells = []
            for L in (2, 8, 32, 128):
                r = t[(t.condition == c) & (t.method == m) & (t.L_size == L)]
                cells.append("—" if r.empty else fmt(r.regret.iloc[0], r.regret_lo.iloc[0], r.regret_hi.iloc[0], d=1))
            md.append(f"| {m} | " + " | ".join(cells) + " |")
        md.append("")
    (OUT / "T5_scaling_regret.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9})
    df = load()
    assert set(df.method) >= set(METHODS), set(METHODS) - set(df.method)
    table_t1(df)
    table_t2()
    (OUT / "T3_complexity.md").write_text(T3, encoding="utf-8")
    t4 = table_t4(df)
    fig_f1(df)
    fig_f2(df, t4)
    fig_f3()
    fig_f4()
    print(f"wrote {OUT.relative_to(ROOT)}", file=sys.stdout)


if __name__ == "__main__":
    main()
