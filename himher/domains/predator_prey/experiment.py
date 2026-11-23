"""Conditions, episode runner and per-episode metrics for the Predator-Prey game.

An episode is N_GAMES games; the prey's habit switches only between games, at
the fixed games in each condition. Game start positions come from their own
RNG stream, so every method faces the same starts for a given seed.

Metrics (one record per episode):
  1. Reward:   cumulative reward, reward per game.
  2. l2:       ||w(most likely model) - w(true)||, per round.
  3. KL:       KL(true prey move probs || predicted), at each visited context.
  4. Catching: catch / den-escape / timeout rates, rounds to catch.
  5. Oracle:   agreement with the oracle's optimal action, and the value lost
               per round under the true model (oracle Q gap).
  plus, for H+H: false-alarm rate, detection delay, accepted regimes.
Model metrics are reported over the whole episode and over the last phase
(games from the final switch onward).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from himher.core.regime import Regime
from himher.core.schedule import SwitchSchedule
from himher.domains.predator_prey.agents import (
    LIBRARY_0,
    BayesPredator,
    ContinuousBayesPredator,
    HHPredator,
    OraclePredator,
    make_hh_predator,
)
from himher.domains.predator_prey.counterpart import (
    CAUTIOUS_B_MODEL,
    DEN_A_MODEL,
    DEN_B_MODEL,
    EVADE_MODEL,
    WeightedPreyModel,
)
from himher.domains.predator_prey.env import T, resolve_round, sample_start
from himher.domains.predator_prey.learning import LearnedHHPredator, make_learned_hh_predator
from himher.domains.predator_prey.planner import plan

N_GAMES = 40
_EPS = 1e-12


@dataclass(frozen=True)
class Condition:
    name: str
    schedule: SwitchSchedule

    @property
    def last_switch(self) -> int:
        points = self.schedule.switch_points
        return points[-1] if points else 0

    @property
    def first_switch(self) -> int | None:
        points = self.schedule.switch_points
        return points[0] if points else None


def conditions() -> list[Condition]:
    A, B, E, C = DEN_A_MODEL, DEN_B_MODEL, EVADE_MODEL, CAUTIOUS_B_MODEL
    return [
        Condition("stationary", SwitchSchedule([(0, A)])),
        Condition("within_library_switch", SwitchSchedule([(0, A), (20, B)])),
        Condition("outside_library_switch_construct", SwitchSchedule([(0, A), (20, E)])),
        Condition("outside_library_switch_merge", SwitchSchedule([(0, A), (10, E), (20, C)])),
        Condition("recurring", SwitchSchedule([(0, A), (10, E), (20, B), (30, E)])),
    ]


METHODS = (
    "hh_replay",  # H+H with learned Q and HER (HH-R)
    "hh_ablation",  # H+H with learned Q, no replay (HH-A)
    "hh_planner",  # H+H cascade, but acting by planning on the incumbent (no policy learning)
    "fixed_bayes",
    "oracle_bayes",
    "continuous_bayes",
    "oracle",
)


def make_predator(method: str, condition: Condition, rng: np.random.Generator, kappa: float):
    if method == "hh_replay":
        return make_learned_hh_predator(rng, kappa, replay=True)
    if method == "hh_ablation":
        return make_learned_hh_predator(rng, kappa, replay=False)
    if method == "hh_planner":
        return make_hh_predator(rng, kappa)
    if method == "fixed_bayes":
        return BayesPredator(LIBRARY_0, rng)
    if method == "oracle_bayes":
        true_types = [policy for _, policy in condition.schedule.segments]
        return BayesPredator((*LIBRARY_0, *true_types), rng)
    if method == "continuous_bayes":
        return ContinuousBayesPredator(rng)
    if method == "oracle":
        return OraclePredator(rng)
    raise ValueError(method)


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, _EPS, 1.0)
    q = np.clip(q, _EPS, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def run_episode(
    method: str, condition: Condition, seed: int, kappa: float, n_games: int = N_GAMES
) -> dict[str, Any]:
    start_rng = np.random.default_rng([seed, 0])
    prey_rng = np.random.default_rng([seed, 1])
    agent_rng = np.random.default_rng([seed, 2])
    predator = make_predator(method, condition, agent_rng, kappa)

    total_reward = 0.0
    game_rewards: list[float] = []
    outcomes: list[str] = []
    catch_lengths: list[int] = []
    l2, kl, agree, gap, in_last_phase, game_of_round = [], [], [], [], [], []

    for game in range(n_games):
        true_model: WeightedPreyModel = condition.schedule.policy_at(game)
        if isinstance(predator, OraclePredator):
            predator.true_model = true_model
        context = sample_start(start_rng)
        oracle_q = plan(true_model)
        game_reward = 0.0
        for t in range(T):
            p, q = context
            pred_action = predator.act(context, t)

            q_row = oracle_q[t, p, q]
            agree.append(q_row[pred_action] >= q_row.max() - 1e-9)
            gap.append(float(q_row.max() - q_row[pred_action]))
            l2.append(float(np.linalg.norm(np.subtract(predator.map_model().w, true_model.w))))
            kl.append(_kl(true_model.predict_proba(context), predator.predictive(context)))
            in_last_phase.append(game >= condition.last_switch)
            game_of_round.append(game)

            prey_action = true_model.sample(context, prey_rng)
            predator.observe(context, prey_action)
            result = resolve_round(context, pred_action, prey_action, t)
            if isinstance(predator, LearnedHHPredator):
                predator.learn(t, context, pred_action, result.reward, result.next_context, result.done)
            total_reward += result.reward
            game_reward += result.reward
            context = result.next_context
            if result.done:
                outcomes.append(result.outcome)
                if result.outcome == "catch":
                    catch_lengths.append(t + 1)
                break
        game_rewards.append(game_reward)

    l2, kl, agree, gap = map(np.asarray, (l2, kl, agree, gap))
    last = np.asarray(in_last_phase)
    record: dict[str, Any] = {
        "condition": condition.name,
        "method": method,
        "seed": seed,
        "cumulative_reward": total_reward,
        "reward_per_game": total_reward / n_games,
        "reward_before_switch": float(np.mean(game_rewards[: condition.first_switch or n_games])),
        "reward_after_switch": (
            float(np.mean(game_rewards[condition.first_switch :]))
            if condition.first_switch is not None
            else np.nan
        ),
        "catch_rate": outcomes.count("catch") / n_games,
        "den_rate": outcomes.count("den") / n_games,
        "timeout_rate": outcomes.count("timeout") / n_games,
        "rounds_to_catch": float(np.mean(catch_lengths)) if catch_lengths else np.nan,
        "rounds_total": len(l2),
        "l2_mean": l2.mean(),
        "l2_last_phase": l2[last].mean(),
        "l2_final": l2[-1],
        "kl_mean": kl.mean(),
        "kl_last_phase": kl[last].mean(),
        "oracle_agreement": agree.mean(),
        "oracle_value_gap": gap.mean(),
        "final_model": repr(predator.map_model()),
    }

    if isinstance(predator, LearnedHHPredator):
        record["replay_updates"] = predator.replay_updates
    if isinstance(predator, (HHPredator, LearnedHHPredator)):
        logs = predator.step_logs
        detected = np.array([log.detected for log in logs])
        games = np.asarray(game_of_round)
        record["false_alarm_rate"] = float(detected[games < (condition.first_switch or n_games)].mean())
        if condition.first_switch is not None:
            after = np.flatnonzero((games >= condition.first_switch) & detected)
            first_round = int(np.argmax(games >= condition.first_switch))
            record["detection_delay"] = float(after[0] - first_round) if len(after) else np.nan
        accepted = [log.regime for log in logs if log.revised]
        record["revisions"] = len(accepted)
        record["reuse_accepted"] = accepted.count(Regime.REUSE)
        record["merge_accepted"] = accepted.count(Regime.REVISE)
        record["construct_accepted"] = accepted.count(Regime.CONSTRUCT)
        record["regime_sequence"] = " ".join(
            f"g{g}:{log.regime.value}->{log.incumbent_after!r}"
            for g, log in zip(game_of_round, logs)
            if log.revised
        )
    return record
