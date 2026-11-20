"""Rebuttal instrumentation (Task L): an episode loop that is step-for-step
identical to ``himher.experiment.runner.run_episode`` but additionally
records, *read-only*, everything the rebuttal analyses need:

  - one summary row per run (returns, regret, recovery time, detection /
    regime counts, final KLD, wall clock, likelihood-evaluation count),
  - decision events (``log_level >= "events"``): detection, every HIM-
    evaluated candidate with the exact quantity compared against tau and
    the binding comparison model, accept / reject, epsilon resets,
  - per-step traces (``log_level == "full"``).

Nothing here mutates agent state or consumes any RNG: the library and
incumbent are snapshotted *before* each update, and events are
reconstructed afterwards from the agent's existing CascadeStepLog. The
re-scoring of comparison models for ``delta_loglik`` runs with the
likelihood counter paused, so the counter reflects only the agent's own
work. tests/test_instrumentation.py checks that returns are bitwise
identical across log levels and against ``run_episode``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from himher.core.him import score
from himher.core.regime import Regime
from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Hand, SwitchingHandoffGame
from himher.experiment import conditions as cond
from himher.experiment import metrics as m
from himher.experiment.runner import RoundLog

LOG_LEVELS = ("none", "events", "full")

# The brief's operator names; the code's Regime.REVISE is the brief's "merge".
OP_NAME = {Regime.REUSE: "reuse", Regime.REVISE: "merge", Regime.CONSTRUCT: "construct"}

_KL_EPS = 1e-12


# ---------------------------------------------------------------------------
# Likelihood-evaluation counter (Section 4.4 of the brief: "add a counter")
# ---------------------------------------------------------------------------
class _LoglikCounter:
    count = 0
    paused = False
    installed = False


def install_loglik_counter() -> None:
    """Wraps BernoulliHandModel.log_prob with a counting shim. Pure
    bookkeeping (returns the original value unchanged); idempotent."""
    if _LoglikCounter.installed:
        return
    original = BernoulliHandModel.log_prob

    def counting_log_prob(self, context, action):  # type: ignore[no-untyped-def]
        if not _LoglikCounter.paused:
            _LoglikCounter.count += 1
        return original(self, context, action)

    BernoulliHandModel.log_prob = counting_log_prob  # type: ignore[method-assign]
    _LoglikCounter.installed = True


# ---------------------------------------------------------------------------
# Conditions (labels exactly as in scripts/run_experiments.py)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConditionSpec:
    label: str
    condition: cond.Condition

    @property
    def switch_times(self) -> tuple[int, ...]:
        return tuple(start for start, _ in self.condition.schedule.segments[1:])

    @property
    def first_switch(self) -> int | None:
        return self.switch_times[0] if self.switch_times else None


def condition_specs() -> list[ConditionSpec]:
    return [
        ConditionSpec("stationary", cond.stationary()),
        ConditionSpec("within_library_switch", cond.within_library_switch()),
        ConditionSpec("outside_library_switch_m1", cond.outside_library_switch(cond.M_NEW_1)),
        ConditionSpec("outside_library_switch_m2", cond.outside_library_switch(cond.M_NEW_2)),
        ConditionSpec("recurring", cond.recurring(cond.M_NEW_2)),
    ]


def condition_by_label(label: str) -> ConditionSpec:
    for spec in condition_specs():
        if spec.label == label:
            return spec
    raise KeyError(label)


# ---------------------------------------------------------------------------
# Model read-out helpers
# ---------------------------------------------------------------------------
def kl_bernoulli(p: float, q: float) -> float:
    q = min(max(q, _KL_EPS), 1.0 - _KL_EPS)
    total = 0.0
    if p > 0.0:
        total += p * math.log(p / q)
    if p < 1.0:
        total += (1.0 - p) * math.log((1.0 - p) / (1.0 - q))
    return total


def context_avg_kl(true_model: BernoulliHandModel, theta: tuple[float, float]) -> float:
    """The Fig. 6 KLD: 1/2 * sum_x KL(Bern(theta_true_x) || Bern(theta_hat_x))
    (verified against AAMAS 2027/results/kld_raw.csv to 4e-16)."""
    return 0.5 * (kl_bernoulli(true_model.theta0, theta[0]) + kl_bernoulli(true_model.theta1, theta[1]))


def held_theta(agent: Any) -> tuple[float, float]:
    """The model an agent currently acts on, as (theta0, theta1): the
    incumbent for H+H agents, the point estimate for CB/WB/DB, and the
    posterior-predictive P(R | x) for FB/OB (not reported in the paper)."""
    cascade = getattr(agent, "cascade", None)
    if cascade is not None:
        return cascade.incumbent.theta
    current_model = getattr(agent, "current_model", None)
    if current_model is not None:
        return current_model().theta
    posterior = agent.posterior
    hyps = agent.hypotheses
    return (
        float(sum(p * h.theta0 for p, h in zip(posterior, hyps))),
        float(sum(p * h.theta1 for p, h in zip(posterior, hyps))),
    )


def _fmt(model: BernoulliHandModel | None) -> str:
    return "" if model is None else f"({model.theta0:.6g},{model.theta1:.6g})"


# ---------------------------------------------------------------------------
# The instrumented episode loop
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    summary: dict[str, Any]
    events: list[dict[str, Any]]
    per_step: dict[str, np.ndarray] | None
    step_times: np.ndarray


def run_instrumented(
    agent: Any,
    spec: ConditionSpec,
    seed: int,
    method: str,
    variant: str = "default",
    log_level: str = "none",
) -> RunResult:
    if log_level not in LOG_LEVELS:
        raise ValueError(f"log_level must be one of {LOG_LEVELS}")
    condition = spec.condition
    schedule = condition.schedule
    T = condition.episode_length
    switch_times = spec.switch_times
    first_switch = spec.first_switch
    run_id = f"{spec.label}__{method}__{variant}__s{seed}"

    is_hh = getattr(agent, "cascade", None) is not None
    want_events = log_level in ("events", "full")
    want_steps = log_level == "full"

    events: list[dict[str, Any]] = []
    logs: list[RoundLog] = []
    step_times = np.empty(T)
    per_step: dict[str, list] = {k: [] for k in (
        "t", "context", "counterpart_action", "ego_action", "ego_reward", "optimal_action",
        "epsilon", "kld", "model_theta0", "model_theta1", "true_theta0", "true_theta1")}

    # Detection-run bookkeeping for step_index (logging only).
    detect_run_attempts = 0
    prev_detected = False
    n_eps_reset_logged = 0

    _LoglikCounter.count = 0
    rng = np.random.default_rng(seed)  # identical to run_episode
    game = SwitchingHandoffGame(schedule, rng)
    run_start = time.perf_counter()
    for t in range(T):
        if is_hh and want_events:
            lib_before = tuple(agent.cascade.library)
            inc_before = agent.cascade.incumbent

        t0 = time.perf_counter()
        context = game.sample_context()
        ego_action = agent.act(context)
        result = game.step(context, ego_action)
        agent.update(context, ego_action, result.counterpart_action, result.reward)
        step_times[t] = time.perf_counter() - t0
        logs.append(RoundLog(t, context, ego_action, result.counterpart_action, result.reward))

        true_model = schedule.policy_at(t)

        if want_steps:
            theta = held_theta(agent)
            policy = getattr(agent, "policy", None)
            eps_used = getattr(agent, "last_epsilon", None)
            if eps_used is None:
                eps_used = policy.epsilon if policy is not None else 0.0
            theta_true_x = true_model.theta1 if context else true_model.theta0
            per_step["t"].append(t)
            per_step["context"].append(context)
            per_step["counterpart_action"].append(int(result.counterpart_action))
            per_step["ego_action"].append(int(ego_action))
            per_step["ego_reward"].append(result.reward)
            per_step["optimal_action"].append(int(Hand.R if theta_true_x > 0.5 else Hand.L))
            per_step["epsilon"].append(eps_used)
            per_step["kld"].append(context_avg_kl(true_model, theta))
            per_step["model_theta0"].append(theta[0])
            per_step["model_theta1"].append(theta[1])
            per_step["true_theta0"].append(true_model.theta0)
            per_step["true_theta1"].append(true_model.theta1)

        if not (is_hh and want_events):
            continue

        # ------------------------- event reconstruction (read-only) ----------
        _LoglikCounter.paused = True
        try:
            step_log = agent.last_step_log
            prior_switches = [s for s in switch_times if s <= t]
            if prior_switches:
                t_rel = t - prior_switches[-1]
            elif first_switch is not None:
                t_rel = t - first_switch
            else:
                t_rel = math.nan
            window = agent.cascade.history.as_tuple()
            w_start, w_end = t - len(window) + 1, t
            base = {
                "run_id": run_id, "t": t, "t_rel_switch": t_rel,
                "true_theta0": true_model.theta0, "true_theta1": true_model.theta1,
                "cur_theta0": inc_before.theta0, "cur_theta1": inc_before.theta1,
            }

            for reset_t in getattr(agent, "eps_reset_times", ())[n_eps_reset_logged:]:
                events.append({**base, "t": reset_t, "event": "eps_reset", "op": ""})
            n_eps_reset_logged = len(getattr(agent, "eps_reset_times", ()))

            if step_log.detected:
                if not prev_detected:
                    detect_run_attempts = 0
                events.append({**base, "event": "detection", "op": "",
                               "statistic": step_log.statistic,
                               "window_start": w_start, "window_end": w_end})
            prev_detected = step_log.detected

            for attempt_idx, attempt in enumerate(step_log.attempts):
                cand = attempt.model
                pool = (inc_before, *(mm for mm in lib_before if mm != inc_before and mm != cand))
                cand_score = score(cand, window)
                diffs = [cand_score - score(mm, window) for mm in pool]
                j = int(np.argmin(diffs))
                delta = diffs[j]
                if not math.isclose(delta, attempt.result.min_improvement, rel_tol=0, abs_tol=1e-9):
                    raise AssertionError(
                        f"reconstructed delta {delta} != HIM min_improvement "
                        f"{attempt.result.min_improvement} ({run_id}, t={t})")
                comp = pool[j]
                row = {**base, "op": OP_NAME[attempt.regime],
                       "cand_theta0": cand.theta0, "cand_theta1": cand.theta1,
                       "delta_loglik": delta,
                       "comparison_model": ("incumbent" if j == 0 else "library") + _fmt(comp),
                       "min_distinctness": attempt.result.min_distinctness,
                       "pool_size": len(pool),
                       "window_start": w_start, "window_end": w_end,
                       "step_index": detect_run_attempts,
                       "attempt_in_round": attempt_idx}
                events.append({**row, "event": "candidate"})
                # select_regime (best_of=False, as in every paper config)
                # returns on the first accepted attempt, so the accepted one
                # is the last attempt of a revised round.
                accepted = (step_log.revised and attempt_idx == len(step_log.attempts) - 1
                            and attempt.regime == step_log.regime
                            and cand == step_log.incumbent_after)
                if accepted:
                    replay = getattr(agent, "last_replay", None)
                    extra = {}
                    if replay is not None and replay[0] == t:
                        extra = {"n_replayed": replay[1], "replay_start": replay[2],
                                 "replay_end": replay[3]}
                    events.append({**row, "event": "accept", **extra})
                    detect_run_attempts = 0
                else:
                    events.append({**row, "event": "reject"})
                    detect_run_attempts += 1
            if step_log.revised:
                prev_detected = False  # the next detection starts a new run
        finally:
            _LoglikCounter.paused = False
    wall = time.perf_counter() - run_start
    n_loglik = _LoglikCounter.count

    # ------------------------------- summary row ---------------------------
    final_theta = held_theta(agent)
    before, after = m.phase_average_return(logs, first_switch if first_switch is not None else T)
    auc = m.cumulative_return(logs)
    summary: dict[str, Any] = {
        "run_id": run_id, "seed": seed, "condition": spec.label, "method": method,
        "variant": variant, "T": T,
        "t_switch": ",".join(str(s) for s in switch_times),
        "return_pre": before,
        "return_post": after if first_switch is not None else math.nan,
        "return_total": auc / T,
        "auc": auc,
        "regret_post": (m.post_switch_regret(logs, schedule.policy_at, first_switch)
                        if first_switch is not None else math.nan),
        "recovery_time": math.nan,
        "kld_final": context_avg_kl(schedule.policy_at(T - 1), final_theta),
        "wall_clock_s": wall,
        "n_loglik_evals": n_loglik,
    }
    if first_switch is not None:
        delay = m.adaptation_delay(logs, schedule.policy_at, first_switch)
        summary["recovery_time"] = math.nan if delay is None else delay

    for key in ("n_detections", "n_false_alarms", "n_accept_reuse", "n_accept_merge",
                "n_accept_construct", "n_reject", "n_voc_declined", "n_revisions",
                "n_policy_updates"):
        summary[key] = math.nan
    if is_hh:
        step_logs = agent.step_logs
        cutoff = first_switch if first_switch is not None else T
        summary["n_detections"] = sum(1 for s in step_logs if s.detected)
        summary["n_false_alarms"] = sum(1 for i, s in enumerate(step_logs) if s.detected and i < cutoff)
        for regime, name in OP_NAME.items():
            summary[f"n_accept_{name}"] = sum(1 for s in step_logs if s.revised and s.regime == regime)
        summary["n_reject"] = len(agent.cascade.rejected)
        # VOC is AlwaysProceedVOC in this codebase: a detected round is
        # never declined by VOC (min_revision_fill=0 in every config).
        summary["n_voc_declined"] = 0
        summary["n_revisions"] = sum(1 for s in step_logs if s.revised)
        summary["n_policy_updates"] = agent.policy.total_updates

    per_step_arrays = None
    if want_steps:
        per_step_arrays = {k: np.asarray(v) for k, v in per_step.items()}
        per_step_arrays["run_id"] = np.array([run_id] * T)
    return RunResult(summary, events, per_step_arrays, step_times)


def roundlogs_equal(a: Sequence[RoundLog], b: Sequence[RoundLog]) -> bool:
    return list(a) == list(b)


AgentFactory = Callable[[int], Any]
