"""Monday brief: analyses for Task P and Task F.

Paired differences: mean over seeds and 95% percentile-bootstrap CI (10,000
resamples, fixed index matrix seed 20261122 -- the Friday convention), `*`
when the CI excludes 0; same table layout as FRIDAY_SUMMARY.md Section 4.

Usage: python analysis/mon_analysis.py
Writes AAMAS 2027/results/rebuttal_mon/{P,F}/tables.md (+ CSVs)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from himher.experiment import instrumentation as inst  # noqa: E402

RES = ROOT / "AAMAS 2027" / "results"
FRI, MON = RES / "rebuttal_fri", RES / "rebuttal_mon"
N_BOOT, BOOT_SEED = 10_000, 20261122
CONDS = ("stationary", "within_library_switch", "outside_library_switch_m1",
         "outside_library_switch_m2", "recurring")
METRICS = ("return_pre", "return_post", "regret_post", "auc")
DIGITS = {"return_pre": 3, "return_post": 3, "regret_post": 2, "auc": 2}
_IDX: dict[int, np.ndarray] = {}


def mean_ci(x) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    if np.all(np.isnan(x)):
        return (math.nan, math.nan, math.nan)
    n = len(x)
    if n not in _IDX:
        _IDX[n] = np.random.default_rng(BOOT_SEED).integers(0, n, size=(N_BOOT, n))
    with np.errstate(invalid="ignore"):
        bs = np.nanmean(x[_IDX[n]], axis=1)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(np.nanmean(x)), float(lo), float(hi)


def fmt(m, lo, hi, d=3, star=False) -> str:
    if math.isnan(m):
        return ""
    s = f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"
    return s + ("*" if star and (lo > 0 or hi < 0) else "")


def load() -> pd.DataFrame:
    parts = []
    for d in (FRI / "orig", FRI / "B_baselines", FRI / "BPR_mdp", MON / "P", MON / "F"):
        s = pd.read_csv(d / "summary.csv")
        parts.append(s)
    s = pd.concat(parts, ignore_index=True).drop_duplicates("run_id")
    s["arm"] = np.where(s.variant == "default", s.method, s.method + "@" + s.variant)
    return s


def paired(s: pd.DataFrame, a: str, b: str) -> list[dict]:
    out = []
    for c in CONDS:
        A = s[(s.condition == c) & (s.arm == a)].set_index("seed").sort_index()
        B = s[(s.condition == c) & (s.arm == b)].set_index("seed").sort_index()
        assert len(A) == len(B) == 200 and (A.index == B.index).all(), (a, b, c, len(A), len(B))
        for mtr in METRICS:
            m, lo, hi = mean_ci((A[mtr] - B[mtr]).to_numpy(float))
            out.append({"condition": c, "A": a, "B": b, "metric": mtr, "mean": m, "lo": lo, "hi": hi})
    return out


def paired_md(rows: pd.DataFrame, title: str) -> list[str]:
    L = [f"**{title}** Positive = A higher. Mean paired difference over 200 seeds [95% bootstrap CI]. "
         "`*` = CI excludes 0. Stationary has no switch: only return_pre (= whole episode) and auc.", "",
         "| condition | A − B | " + " | ".join(METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for (a, b), g in rows.groupby(["A", "B"], sort=False):
        for c in CONDS:
            gc = g[g.condition == c].set_index("metric")
            cells = [fmt(gc.loc[m, "mean"], gc.loc[m, "lo"], gc.loc[m, "hi"], DIGITS[m], True) for m in METRICS]
            L.append(f"| {c} | {a} − {b} | " + " | ".join(cells) + " |")
    return L + [""]


def detection_delays(events: pd.DataFrame) -> pd.DataFrame:
    det = events[events.event == "detection"]
    rows = []
    for rid, g in events.groupby("run_id"):
        c, _, _, sd = rid.split("__")
        fs = inst.condition_by_label(c).first_switch
        if fs is None:
            continue
        after = det[(det.run_id == rid) & (det.t >= fs)].t
        rows.append({"condition": c, "seed": int(sd[1:]), "delay": float(after.min() - fs) if len(after) else math.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- P ---
def task_p(s: pd.DataFrame) -> list[str]:
    out = MON / "P"
    comps = [("HH-plan", b) for b in ("WB-k20", "DB-0.95", "DB-0.98", "BPR+", "HH-R")] + [("HH-plan-eps", "WB-k20-eps")]
    rows = pd.DataFrame([r for a, b in comps for r in paired(s, a, b)])
    rows.to_csv(out / "paired.csv", index=False)
    L = ["# Task P — HH-plan (H+H cascade + WB-k20 decision rule)", ""]
    L += paired_md(rows[rows.A == "HH-plan"], "HH-plan minus baselines.")
    L += paired_md(rows[rows.A == "HH-plan-eps"], "Matched exploration (ε = 0.15, secondary): HH-plan-eps minus WB-k20-eps.")

    # Model accuracy
    dec = pd.read_csv(FRI / "decomposition" / "per_run.csv")
    hh = s[s.arm == "HH-R"].set_index(["condition", "seed"]).kld_final
    hp = s[s.arm == "HH-plan"].set_index(["condition", "seed"]).kld_final
    same_final = bool(np.array_equal(hh.sort_index().to_numpy(), hp.loc[hh.sort_index().index].to_numpy()))
    ev = pd.read_csv(MON / "P" / "events.csv", usecols=["run_id", "t", "event"])
    ev = ev[ev.run_id.str.contains("__HH-plan__")]
    delays = detection_delays(ev)
    L += ["**Model accuracy.** KL(true ‖ held model), context-averaged (Fig. 6 definition). Post-switch = mean "
          "over rounds after the first switch (Stationary: whole episode); final = t = 299. Mean [95% CI] over "
          "200 seeds. HH-plan's held model is the H+H cascade incumbent, which never sees ego actions, so it is "
          "identical round by round to HH-R's (tests/test_hh_plan.py; here kld_final is identical in all "
          f"1,000 runs: {same_final}); its post-switch KL is therefore read from the HH-R rows of "
          "`rebuttal_fri/decomposition/per_run.csv`, as are WB-k20, DB-0.95 and BPR+.", "",
          "| condition | method | KL post-switch | KL final |", "|---|---|---|---|"]
    for c in CONDS:
        ph = "pre" if c == "stationary" else "post"
        for m, src in (("HH-plan", "HH-R"), ("WB-k20", "WB-k20"), ("DB-0.95", "DB-0.95"), ("BPR+", "BPR+")):
            k = dec[(dec.method == src) & (dec.condition == c) & (dec.phase == ph)].sort_values("seed").kld
            arm = "HH-plan" if m == "HH-plan" else m
            kf = s[(s.arm == arm) & (s.condition == c)].sort_values("seed").kld_final
            L.append(f"| {c} | {m} | {fmt(*mean_ci(k), 4)} | {fmt(*mean_ci(kf), 4)} |")
    L += ["", "**HH-plan detection.** Delay = first detected round at or after the first switch minus the switch "
          "round; false alarms = detections before the first switch (all detections in Stationary). Identical to "
          "HH-R's by construction.", "", "| condition | detection delay | runs with no post-switch detection | false alarms |",
          "|---|---|---|---|"]
    for c in CONDS:
        fa = s[(s.arm == "HH-plan") & (s.condition == c)].n_false_alarms
        if c == "stationary":
            L.append(f"| {c} | — | — | {fmt(*mean_ci(fa), 2)} |")
            continue
        dd = delays[delays.condition == c].delay
        L.append(f"| {c} | {fmt(*mean_ci(dd), 1)} | {int(dd.isna().sum())} | {fmt(*mean_ci(fa), 2)} |")

    # baselines beating HH-plan
    beat = rows[(rows.A == "HH-plan") & (rows.hi < 0) & rows.metric.isin(["auc", "return_pre", "return_post"])]
    beat_reg = rows[(rows.A == "HH-plan") & (rows.metric == "regret_post") & (rows.lo > 0)]
    L += ["", "**Where a baseline beats HH-plan** (CI of HH-plan − baseline entirely below 0 for a return/AUC "
          "metric, or above 0 for regret):", ""]
    if beat.empty and beat_reg.empty:
        L.append("- none")
    for _, r in pd.concat([beat, beat_reg]).sort_values(["B", "condition", "metric"]).iterrows():
        L.append(f"- {r.B} beats HH-plan on {r.metric} in {r.condition}: {r['mean']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]")
    (out / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return L


# --------------------------------------------------------------------- F ---
def task_f(s: pd.DataFrame) -> list[str]:
    out = MON / "F"
    comps = [("HH-R@context_filtered", "HH-A"), ("HH-R@context_filtered", "HH-R"),
             ("HH-R@op_gated", "HH-A"), ("HH-R@op_gated", "HH-R")]
    rows = pd.DataFrame([r for a, b in comps for r in paired(s, a, b)])
    rows.to_csv(out / "paired.csv", index=False)
    L = ["# Task F — context-filtered and operator-gated replay", ""]
    L += paired_md(rows, "Replay variants minus HH-A (no replay) and HH-R (full replay).")

    ev = pd.read_csv(out / "events.csv", float_precision="round_trip")
    acc = ev[ev.event == "accept"].copy()
    p = acc.run_id.str.split("__")
    acc["condition"], acc["scope"], acc["seed"] = p.str[0], p.str[2], p.str[3].str[1:].astype(int)
    ref = pd.read_csv(FRI / "A_replay_scope" / "events.csv", usecols=["run_id", "event", "n_replayed"])
    ref = ref[(ref.event == "accept") & ref.run_id.str.contains("__HH-R__default__")].copy()
    rp = ref.run_id.str.split("__")
    ref["condition"], ref["seed"] = rp.str[0], rp.str[3].str[1:].astype(int)
    L += ["**Transitions replayed per acceptance** (all acceptances; per-seed means, then mean [95% CI] over 200 "
          "seeds). Full (HH-R) from Friday `A_replay_scope`.", "",
          "| condition | scope | acceptances | n_replayed | context 0 | context 1 | acceptances with ≥1 context judged unchanged | no replay at all |",
          "|---|---|---|---|---|---|---|---|"]
    nrows = []
    for c in CONDS:
        g = ref[ref.condition == c]
        L.append(f"| {c} | full (HH-R) | {len(g)} | {fmt(*mean_ci(g.groupby('seed').n_replayed.mean().reindex(range(200))), 2)} | | | | "
                 f"{(g.n_replayed == 0).mean():.3f} |")
        for sc in ("context_filtered", "op_gated"):
            g = acc[(acc.condition == c) & (acc.scope == sc)]
            per = g.groupby("seed")[["n_replayed", "n_replayed_ctx0", "n_replayed_ctx1"]].mean().reindex(range(200))
            unch = (g.unchanged_contexts.astype(str) != "none").mean()
            L.append(f"| {c} | {sc} | {len(g)} | {fmt(*mean_ci(per.n_replayed), 2)} | {fmt(*mean_ci(per.n_replayed_ctx0), 2)} | "
                     f"{fmt(*mean_ci(per.n_replayed_ctx1), 2)} | {unch:.3f} | {(g.n_replayed == 0).mean():.3f} |")
            nrows.append({"condition": c, "scope": sc, "acceptances": len(g), **per.mean().to_dict(),
                          "frac_unchanged_any": unch, "frac_zero": (g.n_replayed == 0).mean()})
    pd.DataFrame(nrows).to_csv(out / "n_replayed.csv", index=False)

    per = pd.read_csv(out / "replay_diag" / "per_acceptance.csv")
    per = per[per.t_sw.notna()]
    L += ["", "**Replay diagnostic** (`analysis/replay_diagnostic.py --scopes full context_filtered`, 20 seeds, "
          "acceptances at or after the first switch; per-seed means, then mean [95% CI]). pre_share = share of "
          "replayed transitions generated before the most recent switch; gap_pre = mean log-likelihood of those "
          "transitions under the accepted model minus under the true pre-switch model (only acceptances with "
          "pre-switch replay contribute). Re-runs bitwise identical to the stored runs; replay reconstruction "
          "840/840.", "", "| condition | scope | context | acceptances (with pre-switch replay) | pre_share | gap_pre |",
          "|---|---|---|---|---|---|"]
    drows = []
    for c in CONDS[1:]:
        for sc in ("full", "context_filtered"):
            for cx in ("all", "0", "1"):
                g = per[(per.condition == c) & (per.replay_scope == sc) & (per.context.astype(str) == cx)]
                sm = g.groupby("seed")[["pre_share", "gap_pre"]].mean()
                ps, gp = mean_ci(sm.pre_share), mean_ci(sm.gap_pre)
                L.append(f"| {c} | {sc} | {cx} | {len(g)} ({int((g.n_R_pre > 0).sum())}) | {fmt(*ps)} | {fmt(*gp)} |")
                drows.append({"condition": c, "scope": sc, "context": cx, "pre_share": ps[0], "pre_share_lo": ps[1],
                              "pre_share_hi": ps[2], "gap_pre": gp[0], "gap_pre_lo": gp[1], "gap_pre_hi": gp[2]})
    pd.DataFrame(drows).to_csv(out / "replay_diag_summary.csv", index=False)
    (out / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return L


def main() -> None:
    s = load()
    task_p(s)
    task_f(s)
    print("wrote AAMAS 2027/results/rebuttal_mon/{P,F}/tables.md")


if __name__ == "__main__":
    main()
