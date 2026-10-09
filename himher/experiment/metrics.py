"""Measures from Section 5.5. Task-performance measures (1-3) operate on
RoundLog trajectories, which any agent produces; detection-quality,
model-accuracy, and computational-economy measures (1, 4, 5) need the
H+H-specific CascadeStepLog trajectory, since the Bayesian baselines have
no detection/regime-selection process to report on.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from himher.core.cascade import CascadeStepLog
from himher.core.regime import Regime
from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Context
from himher.domains.handoff.policy import TabularQPolicy
from himher.experiment.runner import RoundLog

TrueModelAt = Callable[[int], BernoulliHandModel]


def steady_state_return(model: BernoulliHandModel) -> float:
    """Expected reward of the optimal responder against ``model``,
    averaged over a uniform context -- the ceiling adaptation speed and
    regret are measured against.
    """
    best_at_0 = max(model.theta0, 1 - model.theta0)
    best_at_1 = max(model.theta1, 1 - model.theta1)
    return float(np.mean([best_at_0, best_at_1]))


def optimal_reward_at(context: Context, true_model: BernoulliHandModel) -> float:
    theta_x = true_model.theta1 if context else true_model.theta0
    return max(theta_x, 1 - theta_x)


def cumulative_return(logs: Sequence[RoundLog]) -> float:
    return float(sum(log.reward for log in logs))


def phase_average_return(logs: Sequence[RoundLog], switch_round: int) -> tuple[float, float]:
    """Mean per-round reward before and from/after ``switch_round``,
    separately -- so phases of different length (e.g. Recurring's
    75-round segments vs. the other conditions' 150-round halves) are
    comparable on the same per-round scale instead of being swamped by
    phase length when summed.
    """
    before = [log.reward for log in logs if log.t < switch_round]
    after = [log.reward for log in logs if log.t >= switch_round]
    before_mean = float(np.mean(before)) if before else float("nan")
    after_mean = float(np.mean(after)) if after else float("nan")
    return before_mean, after_mean


def post_switch_regret(
    logs: Sequence[RoundLog], true_model_at: TrueModelAt, switch_round: int
) -> float:
    """Sum, over rounds from ``switch_round`` onward, of (optimal expected
    reward under the true model active at that round) - (realized
    reward) (Section 5.5 item 3).
    """
    total = 0.0
    for log in logs:
        if log.t < switch_round:
            continue
        total += optimal_reward_at(log.context, true_model_at(log.t)) - log.reward
    return total


def adaptation_delay(
    logs: Sequence[RoundLog],
    true_model_at: TrueModelAt,
    switch_round: int,
    window: int = 20,
    fraction: float = 0.9,
) -> int | None:
    """Rounds after ``switch_round`` until a trailing, full ``window``-
    round average reward first reaches ``fraction`` of the steady-state
    return under the new true model (Section 5.5 item 2). None if it
    never does within the episode.
    """
    target = fraction * steady_state_return(true_model_at(switch_round))
    rewards = [log.reward for log in logs]

    for t in range(switch_round + window - 1, len(rewards)):
        trailing = rewards[t - window + 1 : t + 1]
        if np.mean(trailing) >= target:
            return t - switch_round
    return None


def false_alarm_rate(step_logs: Sequence[CascadeStepLog]) -> float:
    """Fraction of rounds flagged as a mismatch (Section 5.5 item 1).
    Intended for a stationary-counterpart episode, where every detection
    is, by construction, a false alarm.
    """
    if not step_logs:
        return 0.0
    return sum(1 for log in step_logs if log.detected) / len(step_logs)


def detection_delay(step_logs: Sequence[CascadeStepLog], switch_round: int) -> int | None:
    """Rounds between a known switch and the first detection at or after
    it (Section 5.5 item 1).
    """
    for t, log in enumerate(step_logs):
        if t >= switch_round and log.detected:
            return t - switch_round
    return None


def computational_economy(step_logs: Sequence[CascadeStepLog]) -> dict[str, int]:
    """Counts of detections, attempted constructions, accepted revisions
    overall, and accepted uses of each regime individually (Section 5.5
    item 5) -- the per-regime acceptance counts make it possible to
    check, e.g., whether Revise is actually being used for a
    mixture-representable switch rather than Construct.
    """
    attempted = {regime: 0 for regime in Regime}
    accepted = {regime: 0 for regime in Regime}
    for log in step_logs:
        for attempt in log.attempts:
            attempted[attempt.regime] += 1
        if log.revised and log.regime is not None:
            accepted[log.regime] += 1

    return {
        "detections": sum(1 for log in step_logs if log.detected),
        "constructions": attempted[Regime.CONSTRUCT],
        "revise_attempts": attempted[Regime.REVISE],
        "reuse_accepted": accepted[Regime.REUSE],
        "revise_accepted": accepted[Regime.REVISE],
        "construct_accepted": accepted[Regime.CONSTRUCT],
        "revisions": sum(1 for log in step_logs if log.revised),
    }


def policy_updates(policy: TabularQPolicy) -> dict[str, int | dict[BernoulliHandModel, int]]:
    """Total Q-table updates (online + replay) and the per-model
    breakdown (Section 5.5 item 5's "...and policy updates") -- the
    per-model counts show how much attention each model's Q-entries
    received, e.g. a correctly-identified incumbent accumulating many
    updates against a quickly-discarded wrong one accumulating few.
    """
    return {"total": policy.total_updates, "per_model": policy.update_counts}


def model_accuracy(
    step_logs: Sequence[CascadeStepLog], true_model: BernoulliHandModel
) -> float | None:
    """||theta_hat - theta_new|| for the first accepted Revise or
    Construct regime (Section 5.5 item 4) -- both produce a new
    continuous-valued parameter estimate, unlike Reuse, which only
    retrieves an exact existing library entry. None if neither ever
    happened or was accepted.
    """
    for log in step_logs:
        if log.revised and log.regime in (Regime.REVISE, Regime.CONSTRUCT):
            candidate = log.incumbent_after
            return float(
                np.linalg.norm(np.array(candidate.theta) - np.array(true_model.theta))
            )
    return None


def replay_benefit(delay_with_replay: int | None, delay_without_replay: int | None) -> int | None:
    """Positive means replay adapts faster (Section 5.5 item 6); None if
    either run never adapted within its episode.
    """
    if delay_with_replay is None or delay_without_replay is None:
        return None
    return delay_without_replay - delay_with_replay
