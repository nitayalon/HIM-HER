"""One counted episode (Task C3). The loop is runner.run_episode's,
statement for statement (same RNG use); counting wraps only the agent's
act/update calls, and every logging read is done under ``paused``.

Performance columns are recomputed here with the same metric functions
as instrumentation.run_instrumented, and cross-checked against the stored
rebuttal summaries by scripts/rebuttal_cost_runs.py.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from himher.domains.handoff.env import SwitchingHandoffGame
from himher.experiment import cost_counters as cc
from himher.experiment import instrumentation as inst
from himher.experiment import metrics as m
from himher.experiment.rebuttal_methods import METHODS
from himher.experiment.runner import RoundLog

COST_METHODS = ("HH-R", "HH-A", "FB", "OB", "CB", "WB-k20", "DB-0.90", "DB-0.95", "DB-0.98",
                "BOCPD", "BPR+")


def event_counts(agent: Any) -> dict[str, float]:
    keys = ("n_detections", "n_accept_reuse", "n_accept_merge", "n_accept_construct", "n_reject",
            "n_voc_declined", "n_learned_models")
    out: dict[str, float] = dict.fromkeys(keys, math.nan)
    if getattr(agent, "cascade", None) is not None:
        logs = agent.step_logs
        out["n_detections"] = sum(1 for s in logs if s.detected)
        for regime, name in inst.OP_NAME.items():
            out[f"n_accept_{name}"] = sum(1 for s in logs if s.revised and s.regime == regime)
        out["n_reject"] = len(agent.cascade.rejected)
        out["n_voc_declined"] = 0
    elif hasattr(agent, "detection_times"):  # BPR+
        out["n_detections"] = len(agent.detection_times)
        out["n_learned_models"] = len(agent.learned_models)
        out["n_accept_construct"] = len(agent.learned_models)
    return out


def run_cost(agent: Any, spec: inst.ConditionSpec, seed: int, *, count: bool = True,
             record_steps: bool = False, record_trajectory: bool = False) -> dict[str, Any]:
    condition = spec.condition
    schedule = condition.schedule
    T = condition.episode_length
    counter = cc.CostCounter()
    rng = np.random.default_rng(seed)  # identical to run_episode
    game = SwitchingHandoffGame(schedule, rng)
    logs: list[RoundLog] = []
    cum_ev: list[int] = []
    trajectory: list[tuple[float, float]] = []

    def agent_call(fn, *args):
        if not count:
            return fn(*args)
        counter.push("other")
        try:
            return fn(*args)
        finally:
            counter.pop()

    ctx = cc.counting(counter) if count else _null()
    start = time.perf_counter()
    with ctx:
        for t in range(T):
            context = game.sample_context()
            ego_action = agent_call(agent.act, context)
            result = game.step(context, ego_action)
            agent_call(agent.update, context, ego_action, result.counterpart_action, result.reward)
            logs.append(RoundLog(t, context, ego_action, result.counterpart_action, result.reward))
            if record_steps:
                cum_ev.append(counter.ev_total)
            if record_trajectory:
                with cc.paused():
                    trajectory.append(tuple(inst.held_theta(agent)))
    wall_run = time.perf_counter() - start

    first = spec.first_switch
    before, after = m.phase_average_return(logs, first if first is not None else T)
    row: dict[str, Any] = {
        "T": T, "t_switch": ",".join(str(s) for s in spec.switch_times),
        "return_pre": before, "return_post": after if first is not None else math.nan,
        "return_total": m.cumulative_return(logs) / T, "auc": m.cumulative_return(logs),
        "regret_post": (m.post_switch_regret(logs, schedule.policy_at, first) if first is not None else math.nan),
        "recovery_time": math.nan, "wall_run_s": wall_run,
    }
    if first is not None:
        delay = m.adaptation_delay(logs, schedule.policy_at, first)
        row["recovery_time"] = math.nan if delay is None else delay
    with cc.paused():
        row.update(event_counts(agent))
    if count:
        row.update(counter.as_dict())
        if hasattr(agent, "detection_times"):  # BPR+: a detection triggers learning
            row["n_revision_triggers"] = len(agent.detection_times)
            row["n_candidates"] = len(agent.learned_models)
    extras = {"rewards": [l.reward for l in logs], "actions": [int(l.ego_action) for l in logs],
              "cum_ev": cum_ev, "trajectory": trajectory}
    return {"row": row, "extras": extras}


class _null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


def make_agent(key: str, seed: int, spec: inst.ConditionSpec) -> Any:
    return METHODS[key].factory(seed, spec)
