"""Analysis for the Friday rebuttal runs. Reads results/rebuttal_fri/*/ and
writes CSV tables to reports/rebuttal_fri_tables/ plus markdown fragments
(printed / written to reports/rebuttal_fri_tables/*.md) that
reports/FRIDAY_SUMMARY.md is assembled from.

Statistics: mean and 95% percentile-bootstrap CI over seeds (10,000
resamples, fixed resampling seed 20261120). Paired differences resample
seeds jointly (the same index matrix for both arms), so CIs reflect the
paired design. recovery_time is averaged over runs that recovered; the
number that never recovered is reported alongside.

Usage: python scripts/rebuttal_fri_analysis.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "rebuttal_fri"
OUT = ROOT / "reports" / "rebuttal_fri_tables"
EXPERIMENTS = ["orig", "B_baselines", "A_replay_scope", "D_eps_schedule"]
N_BOOT = 10_000
BOOT_SEED = 20261120
SWITCH = {"within_library_switch": 150, "outside_library_switch_m1": 150,
          "outside_library_switch_m2": 150, "recurring": 75}
COND_ORDER = ["stationary", "within_library_switch", "outside_library_switch_m1",
              "outside_library_switch_m2", "recurring"]
ARM_ORDER = ["HH-R|default", "HH-R|him_window", "HH-R|post_detection", "HH-R|decay_reset",
             "HH-R|him_window+decay_reset", "HH-A|default", "FB|default", "OB|default", "CB|default",
             "WB-k20|default", "DB-0.90|default", "DB-0.95|default", "DB-0.98|default"]
METRICS = ["return_pre", "return_post", "return_total", "regret_post", "recovery_time",
           "n_false_alarms", "kld_final"]


def load() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    notes = []
    frames, events = [], []
    for exp in EXPERIMENTS:
        s = pd.read_csv(RUNS / exp / "summary.csv")
        s["experiment"] = exp
        frames.append(s)
        path = RUNS / exp / "events.csv"
        if path.stat().st_size > 2:  # Bayes baselines have no decision events
            e = pd.read_csv(path, low_memory=False)
            e["experiment"] = exp
            events.append(e)
    s = pd.concat(frames, ignore_index=True)
    # Duplicated run_ids (HH-R / HH-A appear in both orig and A_replay_scope):
    # check they are identical, then keep one.
    value_cols = [c for c in s.columns if c not in ("experiment", "wall_clock_s")]
    dup = s[s.duplicated("run_id", keep=False)]
    if not dup.empty:
        g = dup.groupby("run_id")[value_cols].nunique(dropna=False)
        bad = int((g > 1).any(axis=1).sum())
        notes.append(f"{dup.run_id.nunique()} run_ids were run twice (orig and A_replay_scope); "
                     f"{bad} differ in any logged value.")
    s = s.drop_duplicates("run_id", keep="first").reset_index(drop=True)
    e = pd.concat(events, ignore_index=True)
    kept_from = dict(zip(s.run_id, s.experiment))
    e = e[e.experiment == e.run_id.map(kept_from)].copy()
    parts = e.run_id.str.split("__", expand=True)
    e["condition"], e["method"], e["variant"] = parts[0], parts[1], parts[2]
    e["seed"] = parts[3].str[1:].astype(int)
    s["arm"] = s.method + "|" + s.variant
    return s, e, notes


_IDX: np.ndarray | None = None


def boot_idx(n: int) -> np.ndarray:
    global _IDX
    if _IDX is None or _IDX.shape[1] != n:
        _IDX = np.random.default_rng(BOOT_SEED).integers(0, n, size=(N_BOOT, n))
    return _IDX


def mean_ci(x: np.ndarray) -> tuple[float, float, float, int]:
    x = np.asarray(x, dtype=float)
    valid = ~np.isnan(x)
    if valid.sum() == 0:
        return (math.nan, math.nan, math.nan, 0)
    if valid.all():
        bs = x[boot_idx(len(x))].mean(axis=1)
    else:
        with np.errstate(invalid="ignore"):
            bs = np.nanmean(x[boot_idx(len(x))], axis=1)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return (float(np.nanmean(x)), float(lo), float(hi), int(valid.sum()))


def fmt(m, lo, hi, digits=3) -> str:
    if isinstance(m, float) and math.isnan(m):
        return "—"
    return f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


# ---------------------------------------------------------------------------
def main_table(s: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    rows = []
    for cond in COND_ORDER:
        for arm in ARM_ORDER:
            sub = s[(s.condition == cond) & (s.arm == arm)].sort_values("seed")
            if sub.empty:
                continue
            row = {"condition": cond, "arm": arm, "n_seeds": len(sub)}
            for mtr in METRICS:
                m, lo, hi, n = mean_ci(sub[mtr].to_numpy())
                row[mtr], row[f"{mtr}_lo"], row[f"{mtr}_hi"] = m, lo, hi
                if mtr == "recovery_time":
                    row["n_not_recovered"] = int(sub[mtr].isna().sum()) if cond != "stationary" else math.nan
            rows.append(row)
    t = pd.DataFrame(rows)
    md = []
    for cond in COND_ORDER:
        md.append(f"\n**{cond}** (n = 200 seeds per row; mean [95% CI])\n")
        md.append("| method \\| variant | return_pre | return_post | return_total | regret_post | recovery_time (n not recovered) | n_false_alarms | kld_final |")
        md.append("|---|---|---|---|---|---|---|---|")
        for _, r in t[t.condition == cond].iterrows():
            rec = fmt(r.recovery_time, r.recovery_time_lo, r.recovery_time_hi, 1)
            if cond != "stationary":
                rec += f" ({int(r.n_not_recovered)})"
            md.append(f"| {r.arm.replace('|', ' \\| ')} | {fmt(r.return_pre, r.return_pre_lo, r.return_pre_hi)} | "
                      f"{fmt(r.return_post, r.return_post_lo, r.return_post_hi)} | "
                      f"{fmt(r.return_total, r.return_total_lo, r.return_total_hi)} | "
                      f"{fmt(r.regret_post, r.regret_post_lo, r.regret_post_hi, 2)} | {rec} | "
                      f"{fmt(r.n_false_alarms, r.n_false_alarms_lo, r.n_false_alarms_hi, 2)} | "
                      f"{fmt(r.kld_final, r.kld_final_lo, r.kld_final_hi, 4)} |")
    return t, "\n".join(md)


def paired(s: pd.DataFrame, a: str, b: str, metrics: list[str], conds: list[str]) -> list[dict]:
    out = []
    for cond in conds:
        A = s[(s.condition == cond) & (s.arm == a)].set_index("seed").sort_index()
        B = s[(s.condition == cond) & (s.arm == b)].set_index("seed").sort_index()
        assert (A.index == B.index).all()
        for mtr in metrics:
            d = (A[mtr] - B[mtr]).to_numpy(dtype=float)
            m, lo, hi, n = mean_ci(d)
            out.append({"condition": cond, "A": a, "B": b, "metric": mtr, "mean_diff_A_minus_B": m,
                        "ci_lo": lo, "ci_hi": hi, "n_pairs": n})
    return out


def paired_tables(s: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    switch_conds = [c for c in COND_ORDER if c != "stationary"]
    rows = []
    for b in ["WB-k20|default", "DB-0.90|default", "DB-0.95|default", "DB-0.98|default"]:
        rows += [dict(r, block="4a") for r in paired(s, "HH-R|default", b, ["return_post", "regret_post", "auc"], switch_conds)]
    rows += [dict(r, block="4a") for b in ["WB-k20|default", "DB-0.90|default", "DB-0.95|default", "DB-0.98|default"]
             for r in paired(s, "HH-R|default", b, ["auc"], ["stationary"])]
    scope_conds = ["within_library_switch"] + [c for c in switch_conds if c != "within_library_switch"]
    for b in ["HH-R|him_window", "HH-R|post_detection", "HH-A|default"]:
        rows += [dict(r, block="4b") for r in paired(s, "HH-R|default", b, ["return_post", "regret_post", "auc"], scope_conds)]
    rows += [dict(r, block="4b") for r in paired(s, "HH-R|post_detection", "HH-A|default", ["return_post", "regret_post", "auc"], scope_conds)]
    for b in ["HH-R|decay_reset", "HH-R|him_window+decay_reset"]:
        rows += [dict(r, block="4c") for r in paired(s, "HH-R|default", b, ["return_pre", "return_post", "auc"], COND_ORDER)]
    t = pd.DataFrame(rows)
    md = []
    titles = {"4a": "HH-R (default) minus windowed / discounted Bayes",
              "4b": "Replay scope: HH-R full minus each alternative (and post_detection minus HH-A)",
              "4c": "ε schedule: HH-R constant minus decay_reset"}
    for block, title in titles.items():
        md.append(f"\n**{title}.** Positive = A higher. Mean paired difference over 200 seeds [95% bootstrap CI]. "
                  "`*` = CI excludes 0.\n")
        sub = t[t.block == block]
        metrics = list(dict.fromkeys(sub.metric))
        md.append("| condition | A − B | " + " | ".join(metrics) + " |")
        md.append("|---|---|" + "---|" * len(metrics))
        for (cond, a, b), g in sub.groupby(["condition", "A", "B"], sort=False):
            cells = []
            for mtr in metrics:
                r = g[g.metric == mtr]
                if r.empty:
                    cells.append("")
                    continue
                r = r.iloc[0]
                star = "*" if (r.ci_lo > 0 or r.ci_hi < 0) else ""
                digits = 3 if mtr.startswith("return") else 2
                cells.append(fmt(r.mean_diff_A_minus_B, r.ci_lo, r.ci_hi, digits) + star)
            md.append(f"| {cond} | {a.replace('|default', '')} − {b.replace('|default', '')} | " + " | ".join(cells) + " |")
    return t, "\n".join(md)


# ---------------------------------------------------------------------------
def replay_diagnostic() -> tuple[pd.DataFrame, str]:
    from himher.domains.handoff.counterpart import BernoulliHandModel
    d = np.load(RUNS / "A_diag_within_full" / "per_step.npz")
    ps = pd.DataFrame({k: d[k] for k in d.files})
    e = pd.read_csv(RUNS / "A_diag_within_full" / "events.csv")
    acc = e[e.event == "accept"].copy()
    m_pre = BernoulliHandModel(0.1, 0.1)  # true pre-switch model M_L
    rows = []
    for _, a in acc.iterrows():
        steps = ps[(ps.run_id == a.run_id) & (ps.t >= a.replay_start) & (ps.t <= a.replay_end)]
        pre = steps[steps.t < 150]
        cand = BernoulliHandModel(a.cand_theta0, a.cand_theta1)
        cur = BernoulliHandModel(a.cur_theta0, a.cur_theta1)
        def ll(model, df):
            return float(np.mean([model.log_prob(int(c), int(y)) for c, y in zip(df.context, df.counterpart_action)])) if len(df) else math.nan
        rows.append({"run_id": a.run_id, "t": a.t, "t_rel_switch": a.t - 150, "op": a.op,
                     "n_replayed": a.n_replayed, "n_pre_switch": len(pre),
                     "frac_pre_switch": len(pre) / a.n_replayed,
                     "ll_pre_under_accepted": ll(cand, pre),
                     "ll_pre_under_true_pre_switch_model": ll(m_pre, pre),
                     "ll_pre_under_incumbent_before": ll(cur, pre),
                     "ll_post_under_accepted": ll(cand, steps[steps.t >= 150]),
                     "accepted_is_M_R": bool(round(a.cand_theta0, 6) == 0.9 and round(a.cand_theta1, 6) == 0.9)})
    t = pd.DataFrame(rows)
    post = t[t.t >= 150]
    straddle = t[(t.t >= 150) & (t.n_pre_switch > 0)]
    md = [f"Within-library, HH-R `replay_scope=full`, seeds 0–19, `log_level=full`: {len(t)} acceptances in total, "
          f"{(t.t < 150).sum()} before the switch (t < 150) and {len(post)} at or after it.",
          "",
          "| subset | n acceptances | mean n_replayed | mean fraction of replayed transitions from before the switch | "
          "mean log-lik/step of those pre-switch transitions: accepted model | : true pre-switch model M_L | : incumbent before acceptance |",
          "|---|---|---|---|---|---|---|"]
    for name, sub in [("all acceptances", t), ("acceptances at t ≥ 150", post),
                      ("acceptances at t ≥ 150 whose replay includes pre-switch data", straddle)]:
        md.append(f"| {name} | {len(sub)} | {sub.n_replayed.mean():.2f} | {sub.frac_pre_switch.mean():.3f} | "
                  f"{sub.ll_pre_under_accepted.mean():.3f} | {sub.ll_pre_under_true_pre_switch_model.mean():.3f} | "
                  f"{sub.ll_pre_under_incumbent_before.mean():.3f} |")
    first_post = post.sort_values("t").groupby("run_id").head(1)
    md += ["",
           f"First acceptance after the switch, per seed ({len(first_post)} of 20 seeds have one): "
           f"mean t_rel_switch {first_post.t_rel_switch.mean():.1f} (median {first_post.t_rel_switch.median():.0f}); "
           f"mean fraction of replay from before the switch {first_post.frac_pre_switch.mean():.3f}; "
           f"accepted operator counts {first_post.op.value_counts().to_dict()}; "
           f"accepted model = M_R (0.9, 0.9) in {int(first_post.accepted_is_M_R.sum())} of them. "
           f"Over all post-switch acceptances, {int(post.accepted_is_M_R.sum())} of {len(post)} accept M_R."]
    return t, "\n".join(md)


# ---------------------------------------------------------------------------
def m1_control(e: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    acc = e[(e.event == "accept") & (e.method == "HH-R") & (e.variant == "default") &
            (e.condition == "outside_library_switch_m1")].copy()
    md = ["Outside-library m1 = (0.1, 0.5) (`outside_library_switch_m1`), HH-R default, 200 seeds. "
          "The cascade is identical for HH-R, HH-A and every replay/ε variant (it never sees ego actions), "
          "so these counts apply to all of them.", "",
          "| operator | accepted before switch | accepted after switch | per seed (all) | t_rel_switch quantiles (5/25/50/75/95%), post-switch only |",
          "|---|---|---|---|---|"]
    rows = []
    for op in ["reuse", "merge", "construct"]:
        sub = acc[acc.op == op]
        post = sub[sub.t_rel_switch >= 0]
        q = post.t_rel_switch.quantile([.05, .25, .5, .75, .95]).round(0).astype(int).tolist() if len(post) else []
        md.append(f"| {op} | {(sub.t_rel_switch < 0).sum()} | {len(post)} | {len(sub) / 200:.2f} | {q} |")
        rows.append({"op": op, "pre": int((sub.t_rel_switch < 0).sum()), "post": len(post), "q": q})
    hist = acc[acc.t_rel_switch >= 0].assign(bin=lambda d: (d.t_rel_switch // 10) * 10)
    h = hist.pivot_table(index="bin", columns="op", values="t", aggfunc="count", fill_value=0)
    md += ["", "Post-switch accepted revisions by 10-round bin of t_rel_switch:", "",
           "| t_rel_switch bin | " + " | ".join(h.columns) + " |", "|---|" + "---|" * len(h.columns)]
    for b, r in h.iterrows():
        md.append(f"| {int(b)}–{int(b) + 9} | " + " | ".join(str(int(v)) for v in r) + " |")
    final_true = acc[acc.t_rel_switch >= 0]
    hit = final_true[(final_true.cand_theta0.round(6) == 0.1) & (final_true.cand_theta1.round(6) == 0.5)]
    md += ["", f"Accepted post-switch candidates exactly equal to the true m1 (0.1, 0.5): {len(hit)} "
               f"({hit.op.value_counts().to_dict()}), in {hit.run_id.nunique()} of 200 seeds."]
    return acc, "\n".join(md)


def m2_trajectory(e: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    true = np.array([0.2, 0.8])
    ev = e[(e.method == "HH-R") & (e.variant == "default") & (e.condition == "outside_library_switch_m2") &
           (e.t >= 150)]
    rows = []
    for run_id, g in ev.groupby("run_id"):
        g = g.sort_values(["t", "attempt_in_round"], kind="stable")
        cands = g[g.event == "candidate"]
        acc_mc = g[(g.event == "accept") & g.op.isin(["merge", "construct"])]
        final_acc = g[g.event == "accept"].tail(1)
        row = {"run_id": run_id, "seed": int(run_id.split("__s")[1])}
        if acc_mc.empty:
            row.update(steps_to_accept=math.nan, rounds_to_accept=math.nan, dist_at_accept=math.nan, op_at_accept="")
        else:
            first = acc_mc.iloc[0]
            row.update(
                steps_to_accept=int(((cands.t < first.t) | ((cands.t == first.t) & (cands.attempt_in_round <= first.attempt_in_round))).sum()),
                rounds_to_accept=int(first.t - 150),
                dist_at_accept=float(np.linalg.norm([first.cand_theta0 - true[0], first.cand_theta1 - true[1]])),
                op_at_accept=first.op)
        row["n_post_accepts"] = int((g.event == "accept").sum())
        if not final_acc.empty:
            fa = final_acc.iloc[0]
            row["dist_final_accepted"] = float(np.linalg.norm([fa.cand_theta0 - true[0], fa.cand_theta1 - true[1]]))
        rows.append(row)
    t = pd.DataFrame(rows).sort_values("seed")
    ok = t.dropna(subset=["steps_to_accept"])
    q = ok.steps_to_accept.quantile([.25, .5, .75], interpolation="nearest").tolist()
    reps = [int(ok.iloc[(ok.steps_to_accept - v).abs().argsort().iloc[0]].seed) for v in q]
    md = [f"Outside-library m2 = (0.2, 0.8), HH-R default, 200 seeds. Of these, {len(ok)} accept a merge or construct candidate "
          f"after the switch, and {200 - len(ok)} never do. "
          "'Search steps' are every HIM-evaluated candidate (reuse, merge or construct) from t = 150 until the "
          "first accepted merge or construct. Merge's internal 21-point × all-pairs grid is an argmax, so it is not "
          "logged per grid point (see Anomalies).", "",
          "| statistic over seeds | mean | median | 5% | 95% |", "|---|---|---|---|---|"]
    for col, label in [("steps_to_accept", "candidates evaluated until accept"),
                       ("rounds_to_accept", "rounds after switch until accept"),
                       ("dist_at_accept", "‖θ_accepted − θ_true‖₂ at that acceptance"),
                       ("dist_final_accepted", "‖θ − θ_true‖₂ of the last accepted model")]:
        x = t[col].dropna()
        md.append(f"| {label} | {x.mean():.3f} | {x.median():.3f} | {x.quantile(.05):.3f} | {x.quantile(.95):.3f} |")
    md.append(f"\nOperator of that first post-switch merge/construct acceptance: {ok.op_at_accept.value_counts().to_dict()}.")
    md.append(f"\nRepresentative seeds (nearest to the 25th/50th/75th percentile of steps-to-accept): {reps}.")
    for seed in reps:
        run_id = f"outside_library_switch_m2__HH-R__default__s{seed}"
        g = ev[(ev.run_id == run_id)].sort_values(["t", "attempt_in_round"], kind="stable")
        r = t[t.seed == seed].iloc[0]
        stop_t = 150 + r.rounds_to_accept
        g = g[(g.t <= stop_t) & g.event.isin(["accept", "reject"])]
        md += ["", f"*Seed {seed}* (true θ = (0.2, 0.8); τ = 1.0, δ = 0.1):", "",
               "| t | t_rel | op | step | candidate θ | incumbent θ | Δloglik (binding) | binding comparison | min TV | outcome |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for _, x in g.iterrows():
            md.append(f"| {int(x.t)} | {int(x.t_rel_switch)} | {x.op} | {int(x.step_index)} | ({x.cand_theta0:.3f}, {x.cand_theta1:.3f}) | "
                      f"({x.cur_theta0:.3f}, {x.cur_theta1:.3f}) | {x.delta_loglik:.3f} | {x.comparison_model} | "
                      f"{x.min_distinctness:.3f} | {x.event} |")
    return t, "\n".join(md)


# ---------------------------------------------------------------------------
def reuse_audit(s: pd.DataFrame, e: pd.DataFrame) -> str:
    stored = pd.read_csv(ROOT / "AAMAS 2027" / "results" / "main_raw.csv")
    stored = stored[stored.method == "hh_replay"].groupby("condition")[
        ["economy_reuse_accepted", "economy_revise_accepted", "economy_construct_accepted"]].sum()
    new = s[s.arm == "HH-R|default"].groupby("condition")[["n_accept_reuse", "n_accept_merge", "n_accept_construct"]].sum()
    acc = e[(e.event == "accept") & (e.method == "HH-R") & (e.variant == "default") & (e.op == "reuse")]
    md = ["| condition | reuse (logged) | reuse (stored) | merge (logged / stored) | construct (logged / stored) | "
          "reuse accepts whose binding comparison was the incumbent / another library model | median Δloglik of accepted reuse |",
          "|---|---|---|---|---|---|---|"]
    for cond in COND_ORDER:
        a = acc[acc.condition == cond]
        inc = a.comparison_model.str.startswith("incumbent").sum()
        md.append(f"| {cond} | {int(new.loc[cond, 'n_accept_reuse'])} | {int(stored.loc[cond, 'economy_reuse_accepted'])} | "
                  f"{int(new.loc[cond, 'n_accept_merge'])} / {int(stored.loc[cond, 'economy_revise_accepted'])} | "
                  f"{int(new.loc[cond, 'n_accept_construct'])} / {int(stored.loc[cond, 'economy_construct_accepted'])} | "
                  f"{inc} / {len(a) - inc} | {a.delta_loglik.median():.2f} |")
    lib0 = {"(0.1,0.1)", "(0.9,0.9)", "(0.1,0.9)"}
    acc = acc.assign(cand=acc.apply(lambda r: f"({r.cand_theta0:.6g},{r.cand_theta1:.6g})", axis=1))
    md.append(f"\nOf all {len(acc)} accepted reuses, {acc.cand.isin(lib0).sum()} retrieve an L₀ model and "
              f"{(~acc.cand.isin(lib0)).sum()} retrieve a learned (previously accepted) model.")
    return "\n".join(md)


def compute_table() -> str:
    path = RUNS / "timing_serial" / "timing.csv"
    if not path.exists():
        return "(timing_serial not run)"
    t = pd.read_csv(path)
    import json
    meta = json.loads((RUNS / "timing_serial" / "meta.json").read_text(encoding="utf-8"))
    hw = meta["hardware"]
    md = [f"Serial run (1 worker, `log_level=none`, seeds 0–19 × 5 conditions = 100 runs per row). "
          f"Hardware: {hw['cpu']}, {hw['logical_cores']} logical cores, {hw['platform']}; Python {hw['python']}, "
          f"NumPy {hw['numpy']}; no JAX, so there is no compile time. "
          "Likelihood evaluations = calls to `BernoulliHandModel.log_prob` made by the agent itself. "
          "Logging's own re-scoring is excluded, and FB/OB evaluate every hypothesis each round.", "",
          "| method | variant | wall/step mean (µs) | wall/step std (µs) | wall/run mean (ms) | loglik evals/run | revisions/run |",
          "|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        rev = "—" if pd.isna(r.revisions_per_run_mean) else f"{r.revisions_per_run_mean:.2f}"
        md.append(f"| {r.method} | {r.variant} | {1e6 * r.wall_per_step_mean_s:.1f} | {1e6 * r.wall_per_step_std_s:.1f} | "
                  f"{1e3 * r.wall_per_run_mean_s:.2f} | {r.loglik_evals_per_run_mean:,.0f} | {rev} |")
    return "\n".join(md)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s, e, notes = load()
    nan_rows = s[s[["return_pre", "return_total", "auc", "kld_final"]].isna().any(axis=1)]
    notes.append(f"{len(s)} unique runs loaded; {len(nan_rows)} have a NaN in return_pre/return_total/auc/kld_final; "
                 f"non-finite kld_final: {int((~np.isfinite(s.kld_final)).sum())}.")
    t_main, md_main = main_table(s)
    t_main.to_csv(OUT / "main_table.csv", index=False)
    t_pair, md_pair = paired_tables(s)
    t_pair.to_csv(OUT / "paired_differences.csv", index=False)
    t_diag, md_diag = replay_diagnostic()
    t_diag.to_csv(OUT / "replay_diagnostic.csv", index=False)
    t_m1, md_m1 = m1_control(e)
    t_m2, md_m2 = m2_trajectory(e)
    t_m2.to_csv(OUT / "m2_trajectory_per_seed.csv", index=False)
    md_reuse = reuse_audit(s, e)
    md_comp = compute_table()
    for name, md in [("main", md_main), ("paired", md_pair), ("diag", md_diag), ("m1", md_m1), ("m2", md_m2),
                     ("reuse", md_reuse), ("compute", md_comp), ("notes", "\n".join(f"- {n}" for n in notes))]:
        (OUT / f"{name}.md").write_text(md + "\n", encoding="utf-8")
    print("\n".join(notes))
    print(f"wrote tables to {OUT.relative_to(ROOT)}")

    # Assemble reports/FRIDAY_SUMMARY.md from the hand-written template.
    tpl = (ROOT / "reports" / "FRIDAY_SUMMARY.template.md").read_text(encoding="utf-8")
    for k in ["reuse", "main", "paired", "diag", "m1", "m2", "compute"]:
        tpl = tpl.replace("{{" + k + "}}", (OUT / f"{k}.md").read_text(encoding="utf-8").strip())
    tpl = tpl.replace("{{notes}}", " ".join(notes))
    assert "{{" not in tpl
    (ROOT / "reports" / "FRIDAY_SUMMARY.md").write_text(tpl, encoding="utf-8")
    print("wrote reports/FRIDAY_SUMMARY.md")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    main()
