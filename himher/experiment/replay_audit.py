"""Read-only audit of HER replay vs HIM's verification window.

For every accepted revision of an HH agent it records, by *round index*:
  * W: the rounds in the window HIM evaluated the accepted candidate on
       (the ``window`` argument actually passed to HIMVerifier.evaluate);
  * R: the rounds of the transitions replayed under the accepted model
       (identified by object identity of the Transition, which the agent
       creates once per round and stores in its buffer).
and the partner behaviour (true counterpart model) at each of those rounds.

Nothing the agent computes is changed: the wrappers are pass-through and
installed on the instances only. The episode loop is runner.run_episode's
(same RNG use).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from himher.domains.handoff.env import SwitchingHandoffGame
from himher.experiment import instrumentation as inst
from himher.experiment.rebuttal_methods import METHODS


@dataclass(frozen=True)
class AcceptanceAudit:
    t: int                         # round of the acceptance
    regime: str                    # reuse / merge / construct
    him_rounds: tuple[int, ...]    # W, in window order
    him_window_matches: bool       # window passed to HIM == recorded observations of W
    replay_rounds: tuple[int, ...]  # R, in replay order
    behaviour_at_t: tuple[float, float]
    him_prev_behaviour: int        # |{w in W : true model at w != true model at t}|
    replay_prev_behaviour: int     # same, for R


def audit_episode(method: str, label: str, seed: int) -> tuple[list[AcceptanceAudit], list[float]]:
    spec = inst.condition_by_label(label)
    schedule = spec.condition.schedule
    agent = METHODS[method].factory(seed, spec)
    cascade, policy = agent.cascade, agent.policy

    # --- pass-through wrappers (instance attributes only) -----------------
    obs: list[tuple[Any, Any]] = []          # (context, counterpart_action) per round
    hist_rounds: deque[int] = deque(maxlen=cascade.window)
    clock = {"t": -1}
    orig_append = cascade.history.append

    def append(context, action):
        hist_rounds.append(clock["t"])
        return orig_append(context, action)

    cascade.history.append = append  # type: ignore[method-assign]

    evaluated: list[tuple[Any, tuple, tuple]] = []  # (candidate, window, rounds) this round
    verifier = cascade._verifier
    orig_eval = verifier.evaluate

    def evaluate(candidate, incumbent, pool, window):
        evaluated.append((candidate, tuple(window), tuple(hist_rounds)))
        return orig_eval(candidate, incumbent, pool, window)

    verifier.evaluate = evaluate  # type: ignore[method-assign]

    round_of: dict[int, int] = {}            # id(Transition) -> round
    keep: list[Any] = []                     # keep objects alive so ids stay unique
    updates: list[tuple[Any, Any]] = []      # (transition, model) this round
    orig_update = policy.update

    def update(transition, model):
        if id(transition) not in round_of:
            round_of[id(transition)] = clock["t"]
            keep.append(transition)
        updates.append((transition, model))
        return orig_update(transition, model)

    policy.update = update  # type: ignore[method-assign]

    # --- episode loop: identical to runner.run_episode --------------------
    rng = np.random.default_rng(seed)
    game = SwitchingHandoffGame(schedule, rng)
    audits: list[AcceptanceAudit] = []
    rewards: list[float] = []
    for t in range(spec.condition.episode_length):
        clock["t"] = t
        evaluated.clear()
        updates.clear()
        context = game.sample_context()
        action = agent.act(context)
        res = game.step(context, action)
        obs.append((context, res.counterpart_action))
        agent.update(context, action, res.counterpart_action, res.reward)
        rewards.append(res.reward)
        log = agent.last_step_log
        if not log.revised:
            continue
        accepted = log.incumbent_after
        cand = [e for e in evaluated if e[0] == accepted]
        assert len(cand) == 1, "accepted candidate must be HIM-evaluated exactly once"
        _, window, w_rounds = cand[0]
        replayed = [round_of[id(tr)] for tr, m in updates[1:]]  # updates[0] = online update of round t
        assert all(m == accepted for _, m in updates[1:])
        truth = lambda r: schedule.policy_at(r)  # noqa: E731
        now = truth(t)
        audits.append(AcceptanceAudit(
            t=t, regime=inst.OP_NAME[log.regime], him_rounds=w_rounds,
            him_window_matches=window == tuple(obs[r] for r in w_rounds),
            replay_rounds=tuple(replayed),
            behaviour_at_t=(now.theta0, now.theta1),
            him_prev_behaviour=sum(truth(r) != now for r in w_rounds),
            replay_prev_behaviour=sum(truth(r) != now for r in replayed),
        ))
    return audits, rewards
