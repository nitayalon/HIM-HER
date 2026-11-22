"""Separate inference from control (rebuttal follow-up to Tasks B/A).

Every agent here acts on a held model m_t = (theta0, theta1):
  H+H     -> cascade incumbent, but the *action* comes from the Q-table
             (epsilon-greedy over Q(x, m_t, a)), not from m_t directly;
  CB/WB/DB -> point estimate, action = best response BR(m_t);
  FB/OB   -> posterior-predictive P(R | x), action = BR of that.
BR(m)_x = R iff theta_x > 0.5 (ties -> L), the game's analytic optimum.

Per round, with the true model's expected reward
  E[r | a, x] = theta*_x if a == R else 1 - theta*_x,
the expected regret r*(x) - E[r | a_t, x] splits exactly into
  inference   = r*(x)                - E[r | BR(m_t)]       (model on the wrong side of 1/2)
  policy      = E[r | BR(m_t)]       - E[r | greedy_Q]      (H+H only: Q disagrees with its own model)
  exploration = E[r | greedy_Q]      - E[r | a_t]           (H+H only: epsilon draws)
For Bayes agents policy = exploration = 0 by construction. Expected
rewards remove the counterpart's sampling noise; realised returns are
also kept and checked against the stored rebuttal summary.

The episode loop is identical to runner.run_episode (same RNG use), and
all extra bookkeeping is read-only.

Usage: python scripts/rebuttal_decompose.py [--seeds 200] [--workers 12]
Writes AAMAS 2027/results/rebuttal_fri/decomposition/{per_run.csv,curves.csv}
and AAMAS 2027/reports/rebuttal_fri_tables/decomposition.md
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from himher.domains.handoff.env import Hand, SwitchingHandoffGame  # noqa: E402
from himher.experiment import instrumentation as inst  # noqa: E402
from himher.experiment.rebuttal_methods import METHODS  # noqa: E402

OUT_RUNS = ROOT / "AAMAS 2027" / "results" / "rebuttal_fri" / "decomposition"
OUT_TABLES = ROOT / "AAMAS 2027" / "reports" / "rebuttal_fri_tables"
METHOD_KEYS = ("HH-R", "HH-A", "FB", "OB", "CB", "WB-k20", "DB-0.90", "DB-0.95", "DB-0.98")
EARLY = 50  # "early post-switch" = first 50 rounds after the first switch


def br(theta_x: float) -> int:
    return int(Hand.R) if theta_x > 0.5 else int(Hand.L)


def exp_r(a: int, theta_true_x: float) -> float:
    return theta_true_x if a == int(Hand.R) else 1.0 - theta_true_x


def run_one(args: tuple[str, str, int]) -> dict:
    key, label, seed = args
    spec = inst.condition_by_label(label)
    agent = METHODS[key].factory(seed, spec)
    is_hh = hasattr(agent, "cascade")
    schedule = spec.condition.schedule
    T = spec.condition.episode_length
    rng = np.random.default_rng(seed)
    game = SwitchingHandoffGame(schedule, rng)
    rows = np.zeros((T, 9))
    for t in range(T):
        x = game.sample_context()
        true_m = schedule.policy_at(t)
        th_true = true_m.theta1 if x else true_m.theta0
        held = inst.held_theta(agent)
        a_br = br(held[x])
        a_greedy = int(agent.policy.greedy_action(x, agent.cascade.incumbent)) if is_hh else a_br
        a = int(agent.act(x))
        res = game.step(x, Hand(a))
        agent.update(x, Hand(a), res.counterpart_action, res.reward)
        r_star = max(th_true, 1 - th_true)
        e_br, e_gr, e_a = exp_r(a_br, th_true), exp_r(a_greedy, th_true), exp_r(a, th_true)
        decisive = th_true != 0.5
        rows[t] = (res.reward, e_a, r_star - e_br, e_br - e_gr, e_gr - e_a,
                   inst.context_avg_kl(true_m, held),
                   float(a_br == br(th_true)) if decisive else np.nan,
                   float(decisive), float(a != a_greedy) if is_hh else 0.0)
    return {"key": key, "condition": label, "seed": seed, "rows": rows}


COLS = ("reward", "exp_reward", "reg_inference", "reg_policy", "reg_exploration",
        "kld", "model_side_correct", "decisive", "explored")


def phase_stats(rows: np.ndarray, sl: slice) -> dict:
    r = rows[sl]
    d = dict(zip(COLS, np.nanmean(r, axis=0)))
    d["model_side_correct"] = np.nanmean(r[:, 6]) if np.any(r[:, 7] > 0) else np.nan
    d["reg_total"] = d["reg_inference"] + d["reg_policy"] + d["reg_exploration"]
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    labels = [s.label for s in inst.condition_specs()]
    tasks = [(k, c, s) for k in METHOD_KEYS for c in labels for s in range(args.seeds)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = pool.map(run_one, tasks, chunksize=20)

    per_run, curves = [], {}
    for res in results:
        spec = inst.condition_by_label(res["condition"])
        T, fs = spec.condition.episode_length, spec.first_switch
        phases = {"pre": slice(0, fs or T)}
        if fs is not None:
            phases |= {"post": slice(fs, T), "early_post": slice(fs, fs + EARLY)}
        for ph, sl in phases.items():
            per_run.append({"method": res["key"], "condition": res["condition"], "seed": res["seed"],
                            "phase": ph, **phase_stats(res["rows"], sl)})
        curves.setdefault((res["key"], res["condition"]), []).append(res["rows"])

    OUT_RUNS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(per_run)
    df.to_csv(OUT_RUNS / "per_run.csv", index=False)
    curve_rows = []
    for (k, c), stack in curves.items():
        mean = np.nanmean(np.stack(stack), axis=0)
        for t in range(mean.shape[0]):
            curve_rows.append({"method": k, "condition": c, "t": t, **dict(zip(COLS, mean[t]))})
    pd.DataFrame(curve_rows).to_csv(OUT_RUNS / "curves.csv", index=False)

    # Sanity: realised returns must equal the stored rebuttal summaries.
    summ = pd.concat([pd.read_csv(ROOT / "AAMAS 2027/results/rebuttal_fri" / e / "summary.csv")
                      for e in ("orig", "B_baselines")])
    summ = summ[summ.variant == "default"].drop_duplicates(["method", "condition", "seed"])
    chk = df[df.phase == "post"].merge(summ, on=["method", "condition", "seed"])
    max_diff = float((chk.reward - chk.return_post).abs().max())
    print(f"sanity: {len(chk)} runs matched, max |realised - stored return_post| = {max_diff:.2e}")

    agg = (df.groupby(["condition", "phase", "method"], sort=False)
             [["reward", "exp_reward", "reg_total", "reg_inference", "reg_policy",
               "reg_exploration", "model_side_correct", "kld", "explored"]].mean().reset_index())
    agg.to_csv(OUT_TABLES / "decomposition.csv", index=False)
    lines = ["Per-round means over 200 seeds. Regrets are *expected* per-round regret under the true "
             "model (counterpart noise removed); inference + policy + exploration = total exactly. "
             "`model side correct` = fraction of decisive rounds (θ*_x ≠ 0.5) on which the held model "
             "is on the correct side of 1/2. `kld` = Fig. 6 context-averaged KL(true ‖ held). "
             f"`early_post` = first {EARLY} rounds after the first switch.", "",
             f"Sanity check: realised post-switch return equals stored `return_post` for {len(chk)} runs "
             f"(max diff {max_diff:.1e}).", ""]
    for (c, ph), g in agg.groupby(["condition", "phase"], sort=False):
        lines += [f"**{c} — {ph}**", "",
                  "| method | realised return | expected regret | ← inference | ← policy (Q≠BR(m)) | ← exploration | model side correct | KLD | explore rate |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            side = "—" if np.isnan(r.model_side_correct) else f"{r.model_side_correct:.3f}"
            lines.append(f"| {r.method} | {r.reward:.3f} | {r.reg_total:.4f} | {r.reg_inference:.4f} | "
                         f"{r.reg_policy:.4f} | {r.reg_exploration:.4f} | {side} | {r.kld:.4f} | {r.explored:.3f} |")
        lines.append("")
    (OUT_TABLES / "decomposition.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote AAMAS 2027/reports/rebuttal_fri_tables/decomposition.md")


if __name__ == "__main__":
    main()
