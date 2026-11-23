"""Task C6: cost scaling with library size |L| in {2, 8, 32, 128}.

Library for a run: the condition's true model(s) (Stationary: M_L;
Within-library: M_L, M_R), then the remaining L0 models in LIBRARY_0 order,
then distractors theta ~ U[0,1]^2 from a dedicated generator
np.random.default_rng([DISTRACTOR_SEED, seed]) -- never the run's RNG --
truncated to |L|. Every method in a (seed, |L|) cell gets the same library.
Each method uses it where its design puts a library: H+H's initial lib_0
(incumbent M_L, as in the paper), FB's hypotheses, BPR+'s opponent
strategies/policies, BOCPD's hypotheses. CB has no library (reference).
All other hyper-parameters frozen (rebuttal_methods).

BPR+ is not run at |L| = 128: its frozen belief floor 0.01 must be below
1/|L| (BPRPlusAgent rejects it); changing the floor would change the method.

Counts are deterministic, so runs execute in parallel; the wall-clock
columns here are therefore contended (use C3's serial timings).

Usage: python scripts/rebuttal_cost_scaling.py [--seeds 50] [--workers 14]
Writes AAMAS 2027/results/rebuttal_cost/scaling_runs.csv (+ scaling_meta.json)
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from himher.baselines.bocd_bayes import BOCDBayesAgent  # noqa: E402
from himher.baselines.continuous_bayes import ContinuousBayesAgent  # noqa: E402
from himher.baselines.fixed_library_bayes import FixedLibraryBayesAgent  # noqa: E402
from himher.baselines.hh_agent import HHAgent  # noqa: E402
from himher.core.cascade import HHCascade  # noqa: E402
from himher.core.him import HIMVerifier  # noqa: E402
from himher.core.library import ModelLibrary  # noqa: E402
from himher.domains.handoff.construct import construct_bernoulli_model  # noqa: E402
from himher.domains.handoff.counterpart import BernoulliHandModel  # noqa: E402
from himher.domains.handoff.policy import TabularQPolicy  # noqa: E402
from himher.domains.handoff.revise import revise_mixture_model  # noqa: E402
from himher.experiment import conditions as cond  # noqa: E402
from himher.experiment import instrumentation as inst  # noqa: E402
from himher.experiment import rebuttal_methods as rm  # noqa: E402
from himher.experiment.cost_runner import run_cost  # noqa: E402

OUT = ROOT / "AAMAS 2027" / "results" / "rebuttal_cost"
L_SIZES = (2, 8, 32, 128)
CONDS = ("stationary", "within_library_switch")
METHODS = ("HH-R", "FB", "BPR+", "BOCPD", "CB")
DISTRACTOR_SEED = 20261122


def build_library(label: str, seed: int, size: int) -> tuple[BernoulliHandModel, ...]:
    spec = inst.condition_by_label(label)
    lib: list[BernoulliHandModel] = []
    for m in (*spec.condition.true_models_in_order, *cond.LIBRARY_0):
        if m not in lib:
            lib.append(m)
    rng = np.random.default_rng([DISTRACTOR_SEED, seed])
    while len(lib) < size:
        m = BernoulliHandModel(*map(float, rng.random(2)))
        if m not in lib:
            lib.append(m)
    return tuple(lib[:size])


def make(method: str, seed: int, lib: tuple[BernoulliHandModel, ...]):
    if method == "HH-R":  # identical to rebuttal_methods.make_hh_agent except lib_0
        cascade = HHCascade(
            incumbent=cond.M_L,
            library=ModelLibrary(lib, max_learned=rm.MAX_LEARNED),
            kappa=rm.KAPPA, window=rm.WINDOW,
            verifier=HIMVerifier(tau=rm.TAU, delta=rm.DELTA, contexts=rm.CONTEXTS),
            revise=revise_mixture_model,
            construct=lambda w: construct_bernoulli_model(w, rm.ALPHA),
        )
        policy = TabularQPolicy(learning_rate=rm.LEARNING_RATE, epsilon=rm.EPSILON)
        return HHAgent(cascade, policy, np.random.default_rng(seed), replay=True)
    if method == "FB":
        return FixedLibraryBayesAgent(lib)
    if method == "BPR+":  # the MDP implementation (registered "BPR+")
        return rm.make_bpr_agent(seed, library=lib)
    if method == "BOCPD":
        return BOCDBayesAgent(lib, hazard=rm.BOCPD_HAZARD)
    if method == "CB":
        return ContinuousBayesAgent(alpha=rm.ALPHA)
    raise KeyError(method)


def feasible(method: str, size: int) -> bool:
    return not (method == "BPR+" and rm.BPR_BELIEF_FLOOR >= 1.0 / size)


def run_one(task: tuple[str, str, int, int]) -> dict:
    method, label, size, seed = task
    spec = inst.condition_by_label(label)
    lib = build_library(label, seed, size)
    res = run_cost(make(method, seed, lib), spec, seed)
    return {"run_id": f"{label}__{method}__L{size}__s{seed}", "seed": seed, "condition": label,
            "method": method, "variant": "default", "L_size": size, **res["row"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--sizes", nargs="*", type=int, default=list(L_SIZES))
    args = ap.parse_args()
    tasks = [(m, c, L, s) for L in args.sizes for c in CONDS for m in METHODS if feasible(m, L)
             for s in range(args.seeds)]
    skipped = sorted({(m, L) for L in args.sizes for m in METHODS if not feasible(m, L)})
    # Longest tasks first for better load balance.
    tasks.sort(key=lambda t: (t[0] != "HH-R", -t[2]))
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(run_one, tasks, chunksize=1)):
            rows.append(r)
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows).sort_values(["L_size", "condition", "method", "seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "scaling_runs.csv", index=False)
    meta = {"command": " ".join(sys.argv), "seeds": f"0..{args.seeds - 1}", "L_sizes": args.sizes,
            "distractor_seed": DISTRACTOR_SEED, "workers": args.workers,
            "not_run": [f"{m} at |L|={L}: belief floor {rm.BPR_BELIEF_FLOOR} >= 1/|L|" for m, L in skipped],
            "wall_clock_note": "parallel workers: wall_* columns are contended",
            "total_wall_s": time.time() - t0, "n_runs": len(df)}
    (OUT / "scaling_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{len(df)} runs in {time.time() - t0:.0f}s; not run: {meta['not_run']}")


if __name__ == "__main__":
    main()
