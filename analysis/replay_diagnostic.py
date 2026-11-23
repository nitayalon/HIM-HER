"""Replay-contamination diagnostic (rebuttal brief, Sun 11/22). Analysis only.

Question: does HER feed the policy pre-switch experience that the accepted
model does not explain?

For every accepted revision of HH-R (replay_scope in full / him_window /
post_detection), with R the replayed transitions and t_sw the most recent
switch <= t_acc:
  pre_share        = |R_pre| / |R|, R_pre = {t in R : t < t_sw}
  ll_pre_accepted  = mean log p(a_t | c_t) over R_pre under the accepted model
  ll_pre_prev      = same under the incumbent just before the acceptance
  ll_pre_true      = same under the true pre-switch counterpart
  gap_pre          = ll_pre_accepted - ll_pre_true
  ll_post_*        = the same three on R_post (reference)
for context in {all, 0, 1}. The likelihood is the code's own
BernoulliHandModel.log_prob (probabilities clipped to [1e-12, 1 - 1e-12]).

Inputs: the stored 200-seed runs log events only (no per-step context or
counterpart action), so the needed runs are re-run deterministically with
run_instrumented(log_level="full"); their returns are checked bitwise
against the stored summaries before use, and the replayed set is
reconstructed from the buffer rules and checked against the logged
(n_replayed, replay_start, replay_end).

Usage: python analysis/replay_diagnostic.py [--seeds 20] [--workers 4]
Writes AAMAS 2027/results/rebuttal_replay_diag/
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from himher.domains.handoff.counterpart import BernoulliHandModel  # noqa: E402
from himher.domains.handoff.env import Hand  # noqa: E402
from himher.experiment import instrumentation as inst  # noqa: E402
from himher.experiment import rebuttal_methods as rm  # noqa: E402
from himher.experiment.rebuttal_methods import METHODS  # noqa: E402

STORED = ROOT / "AAMAS 2027" / "results" / "rebuttal_fri"
OUT = ROOT / "AAMAS 2027" / "results" / "rebuttal_replay_diag"
CONDS = ("within_library_switch", "outside_library_switch_m2", "recurring")
NICE = {"within_library_switch": "Within-library", "outside_library_switch_m2": "Outside m2",
        "recurring": "Recurring", "outside_library_switch_m1": "Outside m1", "stationary": "Stationary"}
SCOPES = {"full": "HH-R", "him_window": "HH-R@him_window", "post_detection": "HH-R@post_detection",
          "context_filtered": "HH-R@context_filtered"}  # last: Monday Task F (opt-in via --scopes)
DEFAULT_SCOPES = ("full", "him_window", "post_detection")
MON = ROOT / "AAMAS 2027" / "results" / "rebuttal_mon"
CAPACITY = rm.WINDOW  # HHAgent: ReplayBuffer(maxlen=cascade.window), FIFO deque
N_BOOT, BOOT_SEED = 10_000, 20261122
RETURN_COLS = ("return_pre", "return_post", "return_total", "auc")
LL_COLS = [f"ll_{p}_{m}" for p in ("pre", "post") for m in ("accepted", "prev", "true")]


# ----------------------------------------------------------------- core ----
def loglik(theta: tuple[float, float], contexts, actions) -> np.ndarray:
    """Per-transition log p(a | c) under the code's own likelihood."""
    model = BernoulliHandModel(*theta)
    return np.array([model.log_prob(int(c), Hand(int(a))) for c, a in zip(contexts, actions)], float)


def replay_rounds(scope: str, t_acc: int, detected: np.ndarray, accept_ts: set[int],
                  contexts=None, unchanged: tuple[int, ...] = ()) -> list[int]:
    """Rounds replayed at an acceptance in round t_acc, from the buffer rules:
    FIFO buffer of the last CAPACITY rounds, never cleared under HH-R;
    full = whole buffer; him_window = HIM's window (the same last-CAPACITY
    rounds, history and buffer are appended in the same call);
    post_detection = from the first round of the uninterrupted run of
    detected rounds ending at t_acc (a run restarts after an acceptance),
    within the buffer; context_filtered (Monday Task F) = the post_detection
    rounds plus earlier buffered rounds whose context is in ``unchanged``."""
    lo = max(0, t_acc - CAPACITY + 1)
    if scope in ("full", "him_window"):
        return list(range(lo, t_acc + 1))
    if scope not in ("post_detection", "context_filtered"):
        raise ValueError(scope)
    start = t_acc
    while start - 1 >= 0 and detected[start - 1] and (start - 1) not in accept_ts:
        start -= 1
    if scope == "post_detection":
        return list(range(max(lo, start), t_acc + 1))
    return [r for r in range(lo, t_acc + 1) if r >= start or int(contexts[r]) in unchanged]


def acceptance_metrics(R, t_sw, contexts, actions, cand, cur, true_pre, context=None) -> dict:
    """``contexts``/``actions`` are indexed by round. ``context`` restricts R
    to rounds in that context (None = all)."""
    R = [r for r in R if context is None or int(contexts[r]) == context]
    pre = [r for r in R if t_sw is not None and r < t_sw]
    post = [r for r in R if not (t_sw is not None and r < t_sw)]
    out = {"n_R": len(R), "n_R_pre": len(pre),
           "pre_share": len(pre) / len(R) if R else math.nan}
    for name, rr in (("pre", pre), ("post", post)):
        c = [contexts[r] for r in rr]
        a = [actions[r] for r in rr]
        for tag, th in (("accepted", cand), ("prev", cur), ("true", true_pre)):
            out[f"ll_{name}_{tag}"] = float(loglik(th, c, a).mean()) if rr else math.nan
    out["gap_pre"] = out["ll_pre_accepted"] - out["ll_pre_true"]
    return out


# --------------------------------------------------------------- re-run ----
def rerun_one(task: tuple[str, str, int]) -> dict:
    scope, label, seed = task
    key = SCOPES[scope]
    spec = inst.condition_by_label(label)
    ms = METHODS[key]
    res = inst.run_instrumented(ms.factory(seed, spec), spec, seed, ms.method, ms.variant, log_level="full")
    ps = res.per_step
    return {"scope": scope, "condition": label, "seed": seed, "summary": res.summary,
            "events": res.events,
            "steps": pd.DataFrame({"run_id": ps["run_id"], "t": ps["t"], "context": ps["context"],
                                   "counterpart_action": ps["counterpart_action"]})}


def boot_ci(x: np.ndarray) -> tuple[float, float, float, int]:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (math.nan, math.nan, math.nan, 0)
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(x), size=(N_BOOT, len(x)))
    bs = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(x)


def fmt(m, lo, hi, d=3) -> str:
    return "—" if np.isnan(m) else f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


# ---------------------------------------------------------------- main ----
def main() -> None:
    global OUT, CONDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--conditions", nargs="+", default=list(CONDS))
    ap.add_argument("--scopes", nargs="+", default=list(DEFAULT_SCOPES), choices=list(SCOPES))
    ap.add_argument("--out", default=None, help="output dir relative to AAMAS 2027/results (default rebuttal_replay_diag)")
    args = ap.parse_args()
    CONDS = tuple(args.conditions)
    if args.out:
        OUT = ROOT / "AAMAS 2027" / "results" / args.out
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))

    # ---- Step 1: inputs --------------------------------------------------
    orig_ev = pd.read_csv(STORED / "orig" / "events.csv", nrows=1)
    a_ev = pd.read_csv(STORED / "A_replay_scope" / "events.csv", float_precision="round_trip")
    if "context_filtered" in args.scopes:
        a_ev = pd.concat([a_ev, pd.read_csv(MON / "F" / "events.csv", float_precision="round_trip")],
                         ignore_index=True)
    per_step_files = sorted(p.relative_to(ROOT).as_posix() for p in STORED.glob("*/per_step.npz"))
    inputs = [
        "# Step 1 — inputs", "",
        "| needed per replayed transition | found in stored 200-seed results? | used |",
        "|---|---|---|",
        "| t, run_id | yes (`events.csv`) | re-run `per_step` |",
        "| context c_t | **no** — `per_step` was logged only for small runs: " + ", ".join(per_step_files)
        + " (none covers HH-R × {full, him_window, post_detection} × Within/m2/Recurring × 20 seeds) | re-run `per_step` |",
        "| counterpart action a_t | **no** (same) | re-run `per_step` |",
        "| replayed set per acceptance | `A_replay_scope/events.csv` `accept` rows: `n_replayed`, `replay_start`, `replay_end` "
        f"(present: {all(c in a_ev for c in ('n_replayed', 'replay_start', 'replay_end'))}); "
        f"`orig/events.csv` has them: {'n_replayed' in orig_ev.columns} | reconstructed from buffer rules, checked against logged fields |",
        "| accepted model | `events.csv` `cand_theta0/1` on `accept` rows (exact floats) | re-run events |",
        "| model before acceptance | `events.csv` `cur_theta0/1` on `accept` rows (= incumbent before the round's revision) | re-run events |",
        "| true parameters, switch times | `events.csv` `true_theta0/1`; `summary.csv` `t_switch`; condition schedule | schedule (`policy_at`) |",
        "",
        "## Buffer rules (from the code)", "",
        f"- **Capacity:** {CAPACITY} transitions — `HHAgent.__init__`: `ReplayBuffer(maxlen=cascade.window)`, "
        "a `collections.deque`; `cascade.window` = `rebuttal_methods.WINDOW` = 20.",
        "- **Eviction:** FIFO (deque with `maxlen`: appending the 21st transition drops the oldest).",
        "- **Clearing:** HH-R never clears the buffer (only HH-A, `replay=False`, clears it on acceptance).",
        "- **Order within a round** (`HHAgent.update`): `cascade.step` (appends (c_t, a_t) to HIM's history, "
        "runs detection/HIM) → buffer append of round t's transition → online Q-update of round t under the "
        "pre-acceptance model → if revised, replay.",
        "- **`replay_scope=full`** replays every buffered transition, i.e. rounds max(0, t_acc − 19) … t_acc "
        "(including round t_acc itself, which was also just updated online under the previous model).",
        "- **`him_window`** replays rounds t_acc − |history| + 1 … t_acc. The history is also a 20-slot deque "
        "appended in the same call, so this is the same set as `full`.",
        "- **`post_detection`** replays buffered rounds ≥ the first round of the uninterrupted run of detected "
        "rounds that ends at t_acc (the run restarts after each acceptance).",
        "",
        f"## Re-run", "",
        f"HH-R × {{full, him_window, post_detection}} × {{{', '.join(NICE[c] for c in CONDS)}}} × seeds "
        f"{seeds[0]}–{seeds[-1]} (first {len(seeds)} of the paper's list), `run_instrumented(log_level=\"full\")`, "
        "which adds a per-step record of (run_id, t, context, counterpart_action) and does not change the run.",
        ""]

    # ---- re-run -----------------------------------------------------------
    tasks = [(s, c, seed) for s in args.scopes for c in CONDS for seed in seeds]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        runs = pool.map(rerun_one, tasks, chunksize=2)

    # bitwise check against stored summaries
    stored_dirs = [STORED / "orig", STORED / "A_replay_scope"] + ([MON / "F"] if "context_filtered" in args.scopes else [])
    stored = pd.concat([pd.read_csv(d / "summary.csv", float_precision="round_trip") for d in stored_dirs])
    stored = stored[stored.method == "HH-R"].drop_duplicates(["condition", "variant", "seed"])
    new = pd.DataFrame([{**r["summary"], "condition": r["condition"], "seed": r["seed"],
                         "variant": METHODS[SCOPES[r["scope"]]].variant} for r in runs])
    # pass through CSV so both sides have the same float parsing
    tmp = OUT / "_rerun_summary.csv"
    new.to_csv(tmp, index=False)
    new = pd.read_csv(tmp, float_precision="round_trip")
    tmp.unlink()
    chk = new.merge(stored, on=["condition", "variant", "seed"], suffixes=("", "_stored"))
    mism = {c: int((~((chk[c] == chk[f"{c}_stored"]) | (chk[c].isna() & chk[f"{c}_stored"].isna()))).sum())
            for c in RETURN_COLS}
    bitwise_ok = len(chk) == len(new) and not any(mism.values())
    inputs += [f"**Bitwise check vs stored summaries** (`orig`, `A_replay_scope`): {len(chk)} of {len(new)} runs "
               f"matched; mismatches per column {mism} → **{'PASS' if bitwise_ok else 'FAIL'}**.", ""]
    (OUT / "00_inputs.md").write_text("\n".join(inputs), encoding="utf-8")
    if not bitwise_ok:
        print("Re-run returns differ from stored runs: stopping (see 00_inputs.md).")
        return

    steps = pd.concat([r["steps"] for r in runs], ignore_index=True)
    steps.to_csv(OUT / "rerun_per_step.csv", index=False)
    events = pd.DataFrame([e for r in runs for e in r["events"]])
    events.to_csv(OUT / "rerun_events.csv", index=False)

    # ---- Step 2: per acceptance -----------------------------------------
    rows, recon = [], []
    stored_acc = a_ev[a_ev.event == "accept"].set_index(["run_id", "t"])
    for r in runs:
        spec = inst.condition_by_label(r["condition"])
        sched = spec.condition.schedule
        T = spec.condition.episode_length
        st = r["steps"].sort_values("t")
        ctx, act = st.context.to_numpy(), st.counterpart_action.to_numpy()
        ev = pd.DataFrame(r["events"])
        detected = np.zeros(T, bool)
        detected[ev[ev.event == "detection"].t.to_numpy(int)] = True
        acc = ev[ev.event == "accept"].sort_values("t")
        accept_ts = set(acc.t.astype(int))
        run_id = acc.run_id.iloc[0] if len(acc) else None
        first_after: set[int] = set()
        for _, a in acc.iterrows():
            t_acc = int(a.t)
            unchanged = ()
            if r["scope"] == "context_filtered":
                from himher.baselines.hh_agent import unchanged_contexts
                unchanged = unchanged_contexts(BernoulliHandModel(float(a.cur_theta0), float(a.cur_theta1)),
                                               BernoulliHandModel(float(a.cand_theta0), float(a.cand_theta1)),
                                               rm.DELTA)
            R = replay_rounds(r["scope"], t_acc, detected, accept_ts, ctx, unchanged)
            # reconstruction check vs the logged replay fields (re-run and stored)
            logged = (int(a.n_replayed), int(a.replay_start), int(a.replay_end))
            mine = (len(R), R[0], R[-1])
            st_row = stored_acc.loc[(a.run_id, t_acc)] if (a.run_id, t_acc) in stored_acc.index else None
            st_val = None if st_row is None else (int(st_row.n_replayed), int(st_row.replay_start),
                                                   int(st_row.replay_end))
            recon.append({"run_id": a.run_id, "t_acc": t_acc, "reconstructed": mine, "logged_rerun": logged,
                          "logged_stored": st_val, "match_rerun": mine == logged,
                          "match_stored": st_val is None or mine == st_val})
            prior = [s for s in spec.switch_times if s <= t_acc]
            t_sw = prior[-1] if prior else None
            true_pre = sched.policy_at(t_sw - 1) if t_sw is not None else sched.policy_at(t_acc)
            is_first = t_sw is not None and t_sw not in first_after
            if is_first:
                first_after.add(t_sw)
            cand = (float(a.cand_theta0), float(a.cand_theta1))
            cur = (float(a.cur_theta0), float(a.cur_theta1))
            for cx in (None, 0, 1):
                m = acceptance_metrics(R, t_sw, ctx, act, cand, cur, (true_pre.theta0, true_pre.theta1), cx)
                rows.append({"run_id": a.run_id, "condition": r["condition"], "seed": r["seed"],
                             "replay_scope": r["scope"], "t_acc": t_acc,
                             "t_sw": t_sw if t_sw is not None else math.nan,
                             "t_since_switch": t_acc - t_sw if t_sw is not None else math.nan,
                             "first_after_switch": is_first, "op": a.op,
                             "context": "all" if cx is None else str(cx),
                             "cand_theta0": cand[0], "cand_theta1": cand[1], "cur_theta0": cur[0],
                             "cur_theta1": cur[1], "true_pre_theta0": true_pre.theta0,
                             "true_pre_theta1": true_pre.theta1, **m})
    per = pd.DataFrame(rows)
    per.to_csv(OUT / "per_acceptance.csv", index=False)
    rc = pd.DataFrame(recon)
    rc.to_csv(OUT / "reconstruction_check.csv", index=False)

    # ---- Step 3: aggregate ----------------------------------------------
    cols = ["pre_share", "ll_pre_accepted", "ll_pre_prev", "ll_pre_true", "gap_pre",
            "ll_post_accepted", "ll_post_prev", "ll_post_true"]
    main_cols = ["pre_share", "ll_pre_accepted", "ll_pre_prev", "ll_pre_true", "gap_pre"]
    post_sw = per[per.t_sw.notna()]

    def agg(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        if df.empty:  # condition not in this run (e.g. --conditions without Within-library)
            return pd.DataFrame(columns=[*dict.fromkeys([*keys, "op"]), "n_acceptances", "n_with_pre",
                                         *[f"{c}{x}" for c in cols for x in ("", "_lo", "_hi", "_nseeds")]])
        out = []
        for k, g in df.groupby(keys, sort=False):
            k = k if isinstance(k, tuple) else (k,)
            seed_means = g.groupby("seed")[cols].mean()
            row = dict(zip(keys, k))
            row["n_acceptances"] = len(g)
            row["n_with_pre"] = int((g.n_R_pre > 0).sum())
            for c in cols:
                m, lo, hi, n = boot_ci(seed_means[c].to_numpy())
                row |= {c: m, f"{c}_lo": lo, f"{c}_hi": hi, f"{c}_nseeds": n}
            out.append(row)
        return pd.DataFrame(out)

    def with_all_ops(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        return pd.concat([agg(df, keys + ["op"]), agg(df.assign(op="all ops"), keys + ["op"])], ignore_index=True)

    def md_table(t: pd.DataFrame, lead: list[str], title: str, note: str) -> list[str]:
        head = "| " + " | ".join(lead) + " | acceptances (with pre-switch replay) | " + " | ".join(main_cols) + " |"
        lines = [f"## {title}", "", note, "", head, "|" + "---|" * (len(lead) + 1 + len(main_cols))]
        for _, r in t.iterrows():
            lines.append("| " + " | ".join(str(r[c]) for c in lead) + f" | {r.n_acceptances} ({r.n_with_pre}) | "
                         + " | ".join(fmt(r[c], r[f'{c}_lo'], r[f'{c}_hi']) for c in main_cols) + " |")
        return lines + [""]

    ll_note = ("Mean [95% bootstrap CI over seeds]; per-seed means first. ll_* are mean log p(a|c) per "
               "pre-switch replayed transition (only acceptances with ≥1 such transition contribute). "
               "gap_pre = ll_pre_accepted − ll_pre_true (< 0: the accepted model explains the pre-switch "
               "replayed data worse than its true generator). Acceptances before the first switch are excluded.")
    order = {"reuse": 0, "merge": 1, "construct": 2, "all ops": 3}
    w_full = post_sw[(post_sw.condition == "within_library_switch") & (post_sw.replay_scope == "full")]
    t1 = with_all_ops(w_full, ["context"]).sort_values(["context", "op"], key=lambda s: s.map(order) if s.name == "op" else s)
    t2 = agg(post_sw[(post_sw.condition == "within_library_switch")].assign(op="all ops"),
             ["replay_scope", "context"])
    t3 = agg(post_sw[(post_sw.replay_scope == "full") & (post_sw.context == "all")].assign(op="all ops"),
             ["condition"])
    t3["condition"] = t3.condition.map(NICE)
    first = w_full[w_full.first_after_switch]
    t4 = with_all_ops(first, ["context"]).sort_values(["context", "op"], key=lambda s: s.map(order) if s.name == "op" else s)
    for name, t in (("T1_within_full", t1), ("T2_scope_check", t2), ("T3_conditions", t3), ("T4_first_acceptance", t4)):
        t.to_csv(OUT / f"{name}.csv", index=False)

    # ll_post reference (Within, full, all ops)
    ref = agg(w_full.assign(op="all ops"), ["context"])

    # ---- anomalies --------------------------------------------------------
    allc = per[per.context == "all"]
    restricted = allc[(allc.replay_scope != "full") & (allc.n_R_pre > 0)]
    nan_ll = int(per[(per.n_R_pre > 0)][["ll_pre_accepted", "ll_pre_prev", "ll_pre_true"]].isna().any(axis=1).sum())
    anomalies = [
        f"- Acceptances with empty R: {int((allc.n_R == 0).sum())}.",
        f"- NaN log-likelihoods where R_pre is non-empty: {nan_ll}.",
        f"- Acceptances before the first switch (no t_sw; excluded from T1–T4): "
        f"{int(allc.t_sw.isna().sum())} of {len(allc)}.",
        f"- pre_share > 0 under restricted scopes: him_window {int((restricted.replay_scope == 'him_window').sum())}, "
        f"post_detection {int((restricted.replay_scope == 'post_detection').sum())} acceptances"
        + ("" if not (restricted.replay_scope == "post_detection").any() else
           f" (post_detection cases: {', '.join(f'{r.run_id} t={r.t_acc} ({r.n_R_pre}/{r.n_R})' for _, r in restricted[restricted.replay_scope == 'post_detection'].head(8).iterrows())})")
        + ". him_window > 0 is expected: it replays the same rounds as full (see 00_inputs.md).",
        f"- Reconstruction vs logged replay fields: re-run {int(rc.match_rerun.sum())}/{len(rc)} match; "
        f"stored `A_replay_scope` {int(rc.match_stored.sum())}/{len(rc)} match "
        f"({int(rc.logged_stored.notna().sum())} have a stored row).",
    ]

    md = ["# Replay-contamination diagnostic — summary", "",
          f"**Inputs.** Re-run (stored 200-seed runs lack per-step context and counterpart action): HH-R × "
          f"{{full, him_window, post_detection}} × {{Within-library, m2, Recurring}} × seeds 0–{len(seeds) - 1}. "
          f"Returns are bitwise identical to the stored runs ({len(chk)} runs; see `00_inputs.md`). "
          "Likelihood: the code's `BernoulliHandModel.log_prob` (clip 1e-12, not 1e-6; no θ in these runs is "
          "within 1e-6 of 0 or 1 unless noted in anomalies).", "",
          f"**Reconstruction check.** Replayed sets rebuilt from the buffer rules (FIFO, capacity {CAPACITY}; "
          f"scope rules in `00_inputs.md`) match the re-run's logged (n_replayed, replay_start, replay_end) in "
          f"{int(rc.match_rerun.sum())}/{len(rc)} acceptances and the stored `A_replay_scope` log in "
          f"{int(rc.match_stored.sum())}/{len(rc)}.", ""]
    md += md_table(t1, ["context", "op"], "T1 — Within-library, replay_scope = full", ll_note)
    md += ["One line: " + "{{T1}}", ""]
    md += md_table(t2, ["replay_scope", "context"], "T2 — scope check (Within-library, all ops)", ll_note)
    md += ["One line: " + "{{T2}}", ""]
    md += md_table(t3, ["condition"], "T3 — across conditions (full, all ops, all contexts)", ll_note)
    md += ["One line: " + "{{T3}}", ""]
    md += md_table(t4, ["context", "op"], "T4 — first acceptance after each switch (Within-library, full)", ll_note)
    md += ["One line: " + "{{T4}}", ""]
    md += ["## Reference — post-switch replayed transitions (Within-library, full, all ops)", "",
           "| context | ll_post_accepted | ll_post_prev | ll_post under the pre-switch truth |", "|---|---|---|---|"]
    for _, r in ref.iterrows():
        md.append(f"| {r.context} | " + " | ".join(fmt(r[c], r[f'{c}_lo'], r[f'{c}_hi'])
                                                  for c in ("ll_post_accepted", "ll_post_prev", "ll_post_true")) + " |")
    md += ["", "(The last column is a contrast: post-switch data scored under the pre-switch generator.)", "",
           "## Anomalies", "", *anomalies, ""]
    (OUT / "REPLAY_DIAG_SUMMARY.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}; bitwise {bitwise_ok}; reconstruction "
          f"{int(rc.match_rerun.sum())}/{len(rc)} (re-run), {int(rc.match_stored.sum())}/{len(rc)} (stored)")


if __name__ == "__main__":
    main()
