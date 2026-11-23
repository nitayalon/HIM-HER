"""Monday Task X: why does decay_reset earn less pre-switch return than
constant epsilon? Investigation only (no tuning, no method change).

20 seeds x 5 conditions, episode loop identical to runner.run_episode (same
RNG use). Arms:
  constant        HH-R            (eps = 0.15, replay)
  decay_reset     HH-R@decay_reset (eps0 0.15, floor 0.02, half-life 50, reset on detection, replay)
  constant/noHER  HH-A            (eps = 0.15, no replay)        -- diagnostic only
  decay/noHER     make_hh_agent(replay=False, eps_schedule=decay_reset, same params) -- diagnostic only
Per round it records (read-only): epsilon used, the greedy action of the
Q-table under the incumbent *before* acting, the action taken, the optimal
action under the true model, whether both Q-entries of (context, incumbent)
are still 0 (untrained key: TabularQPolicy.greedy_action breaks the tie to R),
detections and revisions.

Usage: python analysis/mon_task_x.py [--seeds 20]
Writes AAMAS 2027/results/rebuttal_mon/X/{per_step.csv,tables.md}
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

from himher.baselines.hh_agent import decay_reset_epsilon  # noqa: E402
from himher.domains.handoff.env import Hand, SwitchingHandoffGame  # noqa: E402
from himher.experiment import instrumentation as inst  # noqa: E402
from himher.experiment import rebuttal_methods as rm  # noqa: E402

OUT = ROOT / "AAMAS 2027" / "results" / "rebuttal_mon" / "X"
LABELS = [s.label for s in inst.condition_specs()]
NICE = {"stationary": "Stationary", "within_library_switch": "Within-library",
        "outside_library_switch_m1": "Outside m1", "outside_library_switch_m2": "Outside m2",
        "recurring": "Recurring"}
ARMS = ("constant", "decay_reset", "constant/noHER", "decay/noHER")
BINS = [(0, 25), (25, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 300)]


def make(arm: str, seed: int, spec):
    if arm == "constant":
        return rm.METHODS["HH-R"].factory(seed, spec)
    if arm == "decay_reset":
        return rm.METHODS["HH-R@decay_reset"].factory(seed, spec)
    if arm == "constant/noHER":
        return rm.METHODS["HH-A"].factory(seed, spec)
    return rm.make_hh_agent(seed, replay=False, eps_schedule="decay_reset", eps0=rm.EPS0,
                            eps_min=rm.EPS_MIN, eps_half_life=rm.EPS_HALF_LIFE)


def run(task):
    arm, label, seed = task
    spec = inst.condition_by_label(label)
    sched = spec.condition.schedule
    agent = make(arm, seed, spec)
    game = SwitchingHandoffGame(sched, np.random.default_rng(seed))
    rows = []
    for t in range(spec.condition.episode_length):
        x = game.sample_context()
        inc = agent.cascade.incumbent
        q = agent.policy
        untrained = q.q_value(x, inc, Hand.L) == 0.0 and q.q_value(x, inc, Hand.R) == 0.0
        a_greedy = int(q.greedy_action(x, inc))
        a = int(agent.act(x))
        eps = agent.last_epsilon
        res = game.step(x, Hand(a))
        agent.update(x, Hand(a), res.counterpart_action, res.reward)
        tm = sched.policy_at(t)
        th = tm.theta1 if x else tm.theta0
        opt = int(Hand.R) if th > 0.5 else int(Hand.L)
        log = agent.last_step_log
        rows.append((arm, label, seed, t, x, eps, a_greedy, a, opt, th != 0.5, untrained,
                     res.reward, log.detected, log.revised))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(a, c, s) for a in ARMS for c in LABELS for s in range(args.seeds)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        res = pool.map(run, tasks, chunksize=4)
    d = pd.DataFrame([r for rows in res for r in rows],
                     columns=["arm", "condition", "seed", "t", "context", "eps", "a_greedy", "a", "opt",
                              "decisive", "untrained", "reward", "detected", "revised"])
    d.to_csv(OUT / "per_step.csv", index=False)
    first = {c: inst.condition_by_label(c).first_switch or 300 for c in LABELS}
    d["pre"] = d.t < d.condition.map(first)
    d["greedy_opt"] = np.where(d.decisive, (d.a_greedy == d.opt).astype(float), np.nan)
    d["explored_diff"] = (d.a != d.a_greedy).astype(float)

    L = ["# Task X — epsilon-schedule anomaly (20 seeds × 5 conditions)", ""]

    # 1. schedule as specified?
    dr = d[d.arm.isin(["decay_reset", "decay/noHER"])].sort_values(["arm", "condition", "seed", "t"])
    maxdev = 0.0
    resets_pre = []
    for (arm, c, s), g in dr.groupby(["arm", "condition", "seed"]):
        t_reset, expect = 0, []
        for t, det in zip(g.t, g.detected):
            expect.append(decay_reset_epsilon(t, t_reset, rm.EPS0, rm.EPS_MIN, rm.EPS_HALF_LIFE))
            if det:
                t_reset = t  # known after round t's action -> applies from t + 1
        maxdev = max(maxdev, float(np.max(np.abs(np.array(expect) - g.eps.to_numpy()))))
        if arm == "decay_reset":
            resets_pre.append({"condition": c, "seed": s, "resets_pre": int(g[g.t < first[c]].detected.sum())})
    rp = pd.DataFrame(resets_pre).groupby("condition").resets_pre.agg(["mean", "max"])
    L += ["## 1. Is the schedule implemented as specified?", "",
          f"Realised ε_t vs the formula ε_t = max(0.02, 0.15·2^(−(t − t_reset)/50)), with t_reset = the last "
          f"detected round before t (0 at the start), over every step of every decay_reset run: "
          f"max |difference| = {maxdev:.1e}.", "",
          "| t | " + " | ".join(f"seed {s}" for s in range(3)) + " | formula with no reset |", "|---|---|---|---|---|"]
    st = d[(d.arm == "decay_reset") & (d.condition == "stationary")]
    for t in (0, 50, 100, 150, 200, 250, 299):
        cells = [f"{st[(st.seed == s) & (st.t == t)].eps.iloc[0]:.4f}" for s in range(3)]
        L.append(f"| {t} | " + " | ".join(cells) + f" | {decay_reset_epsilon(t, 0, rm.EPS0, rm.EPS_MIN, rm.EPS_HALF_LIFE):.4f} |")
    L += ["", "t = 500 does not exist (T = 300). Resets before the first switch (all detections in Stationary):", "",
          "| condition | mean resets per episode | max |", "|---|---|---|"]
    for c in LABELS:
        L.append(f"| {NICE[c]} | {rp.loc[c, 'mean']:.2f} | {int(rp.loc[c, 'max'])} |")
    det_rounds = st[st.detected].t.tolist()
    L += ["", f"Stationary, seeds 0–2, detection rounds: " + "; ".join(
        f"seed {s}: {st[(st.seed == s) & st.detected].t.tolist()}" for s in range(3)), ""]

    # 3. policy learning
    L += ["## 3. Policy learning: greedy action optimal (decisive rounds), by time bin", "",
          "Fraction of rounds whose greedy Q-action under the incumbent equals the optimal action under the "
          "true model; mean over 20 seeds. `untrained` = fraction of rounds where both Q(x, incumbent, ·) are "
          "still 0, so `greedy_action` breaks the tie to **R**, which is wrong under M_L (θ = 0.1).", ""]
    for c in ("stationary", "within_library_switch"):
        L += [f"**{NICE[c]}**", "",
              "| arm | " + " | ".join(f"t {a}–{b - 1}" for a, b in BINS) + " |", "|---|" + "---|" * len(BINS)]
        for arm in ARMS:
            g = d[(d.arm == arm) & (d.condition == c)]
            cells = []
            for a, b in BINS:
                gb = g[(g.t >= a) & (g.t < b)]
                cells.append(f"{gb.groupby('seed').greedy_opt.mean().mean():.3f} (ε {gb.eps.mean():.3f}; untr. {gb.untrained.mean():.2f})")
            L.append(f"| {arm} | " + " | ".join(cells) + " |")
        L.append("")

    # pre-switch return decomposition
    L += ["## 3b. Pre-switch return and where it is lost (mean per round over seeds)", "",
          "| condition | arm | return_pre | greedy optimal | untrained-key rounds | realised action ≠ greedy | mean ε |",
          "|---|---|---|---|---|---|---|"]
    for c in LABELS:
        for arm in ARMS:
            g = d[(d.arm == arm) & (d.condition == c) & d.pre]
            ps = g.groupby("seed")
            L.append(f"| {NICE[c]} | {arm} | {ps.reward.mean().mean():.3f} | {ps.greedy_opt.mean().mean():.3f} | "
                     f"{ps.untrained.mean().mean():.3f} | {ps.explored_diff.mean().mean():.3f} | {ps.eps.mean().mean():.3f} |")
    # paired constant − decay
    L += ["", "**Paired difference constant − decay_reset in return_pre** (20 seeds), with and without HER:", "",
          "| condition | with HER (HH-R) | without HER (HH-A) |", "|---|---|---|"]
    for c in LABELS:
        cells = []
        for a1, a2 in (("constant", "decay_reset"), ("constant/noHER", "decay/noHER")):
            r1 = d[(d.arm == a1) & (d.condition == c) & d.pre].groupby("seed").reward.mean()
            r2 = d[(d.arm == a2) & (d.condition == c) & d.pre].groupby("seed").reward.mean()
            diff = (r1 - r2).to_numpy()
            bs = diff[np.random.default_rng(20261122).integers(0, len(diff), (10_000, len(diff)))].mean(1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            cells.append(f"{diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
        L.append(f"| {NICE[c]} | " + " | ".join(cells) + " |")
    (OUT / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {(OUT / 'tables.md').relative_to(ROOT)}; max eps deviation {maxdev:.1e}")


if __name__ == "__main__":
    main()
