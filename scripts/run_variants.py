"""Compares H+H design variants, one change at a time against the baseline.

Variants (each differs from the baseline in exactly one setting):
  baseline      production config: cost-ordered regimes, no warm-up,
                window k=20, tau=0.5, kappa=0.6705
  best_of       Reuse and Revise both evaluated; larger accepted improvement wins
  warmup        revisions blocked until the verification window is full
  window50      verification window k=50 (detection and HIM both use it)
  tau0          HIM acceptance margin tau=0.0 (was 0.5)
  kappa_sens    detection threshold kappa=0.3618 (more sensitive; ~35% stationary false alarms)

Per episode the script records return, adaptation delay, late-window L2
distance of the incumbent to the true model, regime acceptance counts,
and false-alarm rate. Writes results/variants/episodes.csv and summary.csv.

Usage: python scripts/run_variants.py [--seeds N] [--out DIR]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from himher.baselines.hh_agent import HHAgent
from himher.core.cascade import HHCascade
from himher.core.him import HIMVerifier
from himher.core.library import ModelLibrary
from himher.domains.handoff.construct import construct_bernoulli_model
from himher.domains.handoff.policy import TabularQPolicy
from himher.domains.handoff.revise import revise_mixture_model
from himher.experiment import conditions as cond
from himher.experiment import metrics as m
from himher.experiment.aggregate import summarize
from himher.experiment.runner import run_episode

CONTEXTS = (0, 1)
ALPHA = 1.0
LEARNING_RATE = 0.3
EPSILON = 0.15


@dataclass(frozen=True)
class Config:
    name: str
    window: int = 20
    tau: float = 0.5
    kappa: float = 0.6705027574263254
    best_of: bool = False
    min_revision_fill: int = 0


VARIANTS = [
    Config("baseline"),
    Config("best_of", best_of=True),
    Config("warmup", min_revision_fill=20),
    Config("window50", window=50),
    Config("tau0", tau=0.0),
    Config("kappa_sens", kappa=0.3618),
]


def make_agent(config: Config, seed: int) -> HHAgent:
    cascade = HHCascade(
        incumbent=cond.LIBRARY_0[0],
        library=ModelLibrary(cond.LIBRARY_0, max_learned=20),
        kappa=config.kappa,
        window=config.window,
        verifier=HIMVerifier(tau=config.tau, delta=0.1, contexts=CONTEXTS),
        revise=revise_mixture_model,
        construct=lambda w: construct_bernoulli_model(w, ALPHA),
        best_of=config.best_of,
        min_revision_fill=config.min_revision_fill,
    )
    return HHAgent(cascade, TabularQPolicy(learning_rate=LEARNING_RATE, epsilon=EPSILON),
                   np.random.default_rng(seed), replay=True)


def _specs() -> list[tuple[str, cond.Condition, int]]:
    return [
        ("stationary", cond.stationary(), 0),
        ("within_library_switch", cond.within_library_switch(), 150),
        ("outside_library_switch_m1", cond.outside_library_switch(cond.M_NEW_1), 150),
        ("outside_library_switch_m2", cond.outside_library_switch(cond.M_NEW_2), 150),
        ("recurring", cond.recurring(cond.M_NEW_2), 75),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--out", type=str, default="results/variants")
    parser.add_argument("--variant", type=str, default="all", help="run one variant name, or all")
    parser.add_argument("--merge", action="store_true", help="merge per-variant CSVs into summary.csv")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        frames = [pd.read_csv(p) for p in sorted(out_dir.glob("episodes_*.csv"))]
        episodes = pd.concat(frames, ignore_index=True)
        episodes.to_csv(out_dir / "episodes.csv", index=False)
        value_columns = [c for c in episodes.columns if c not in ("variant", "condition", "seed")]
        summary = summarize(episodes, group_by=["variant", "condition"], value_columns=value_columns)
        summary.to_csv(out_dir / "summary.csv", index=False)
        print(f"merged {len(frames)} variants into {out_dir}/summary.csv")
        return

    selected = [c for c in VARIANTS if args.variant in ("all", c.name)]
    rows = []
    for config in selected:
        for name, condition, switch in _specs():
            true_theta = np.array(
                [condition.schedule.policy_at(t).theta for t in range(condition.episode_length)]
            )
            for seed in range(args.seeds):
                agent = make_agent(config, seed)
                logs = run_episode(agent, condition, seed)
                held = np.array([s.incumbent_after.theta for s in agent.step_logs])
                distance = np.linalg.norm(held - true_theta, axis=1)
                economy = m.computational_economy(agent.step_logs)
                row = {
                    "variant": config.name,
                    "condition": name,
                    "seed": seed,
                    "cumulative_return": m.cumulative_return(logs),
                    "late_l2": float(distance[250:300].mean()),
                    "false_alarm_rate": m.false_alarm_rate(agent.step_logs),
                    "revisions": economy["revisions"],
                    "reuse_accepted": economy["reuse_accepted"],
                    "revise_accepted": economy["revise_accepted"],
                    "construct_accepted": economy["construct_accepted"],
                }
                if name != "stationary":
                    row["adaptation_delay"] = m.adaptation_delay(logs, condition.schedule.policy_at, switch)
                    row["detection_delay"] = m.detection_delay(agent.step_logs, switch)
                rows.append(row)
            print(f"[{config.name}] {name} done", flush=True)

    episodes = pd.DataFrame(rows)
    episodes.to_csv(out_dir / f"episodes_{args.variant}.csv", index=False)
    print(f"wrote {out_dir}/episodes_{args.variant}.csv")


if __name__ == "__main__":
    main()
