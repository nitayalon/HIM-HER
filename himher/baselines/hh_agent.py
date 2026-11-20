"""Wraps HHCascade (Stages 1-4: SWM adaptation) and TabularQPolicy
(Stage 5: HER-based policy adaptation, Section 4.5) into the common Agent
interface, so it can be run and compared against the Bayesian baselines
through the same experiment runner.

The replay buffer is bounded to the cascade's window length (the same
k used by HIM's verification window h_{t-k:t}, Remark 4.2): relabeling is
only licensed for transitions HIM actually checked something about, not
the agent's entire history, so the buffer must never hold more than that.
"""

from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np

from himher.core.cascade import CascadeStepLog, HHCascade
from himher.core.her import ReplayBuffer
from himher.domains.handoff.env import Context, Hand
from himher.domains.handoff.policy import TabularQPolicy, Transition


REPLAY_SCOPES = ("full", "him_window", "post_detection")


class HHAgent:
    def __init__(
        self,
        cascade: HHCascade,
        policy: TabularQPolicy,
        rng: np.random.Generator,
        replay: bool = True,
        replay_scope: str = "full",
    ) -> None:
        """``replay`` toggles Section 5.2's ablation: when False, the
        buffer of pre-acceptance transitions is discarded on revision
        instead of being replayed under the newly accepted model.

        ``replay_scope`` (rebuttal Task A; only used when ``replay``):
          - "full" (default, the paper's HH-R): replay the whole buffer,
            i.e. the last ``cascade.window`` transitions. Unchanged code path.
          - "him_window": replay only transitions inside the window HIM
            evaluated the accepted candidate on (t - |history| + 1 .. t).
            Because the buffer and HIM's history are both the last k
            observations, this selects the same transitions as "full".
          - "post_detection": replay only transitions from the first round
            of the detection run that ended in this acceptance (the earliest
            round of the uninterrupted streak of detected rounds since the
            previous acceptance), limited to what the buffer still holds.
        """
        if replay_scope not in REPLAY_SCOPES:
            raise ValueError(f"replay_scope must be one of {REPLAY_SCOPES}")
        self._cascade = cascade
        self._policy = policy
        self._rng = rng
        self._replay = replay
        self._replay_scope = replay_scope
        self._buffer: ReplayBuffer[Transition] = ReplayBuffer(maxlen=cascade.window)
        # Bookkeeping only (never read on the default path): the round index
        # of each buffered transition, and the start of the current
        # detection run.
        self._buffer_times: Deque[int] = deque(maxlen=cascade.window)
        self._t = 0
        self._detect_run_start: int | None = None
        # (t, n_replayed, first replayed round, last replayed round), per acceptance.
        self.last_replay: tuple[int, int, int, int] | None = None
        self.replay_log: list[tuple[int, int, int, int]] = []
        self.last_step_log: CascadeStepLog | None = None
        self.step_logs: list[CascadeStepLog] = []

    @property
    def cascade(self) -> HHCascade:
        return self._cascade

    @property
    def policy(self) -> TabularQPolicy:
        return self._policy

    @property
    def buffer(self) -> ReplayBuffer[Transition]:
        return self._buffer

    @property
    def replay_enabled(self) -> bool:
        return self._replay

    @property
    def replay_scope(self) -> str:
        return self._replay_scope

    def act(self, context: Context) -> Hand:
        return self._policy.act(context, self._cascade.incumbent, self._rng)

    def update(
        self, context: Context, ego_action: Hand, counterpart_action: Hand, reward: float
    ) -> None:
        t = self._t
        transition = Transition(context, ego_action, counterpart_action, reward)
        model_at_action_time = self._cascade.incumbent

        step_log = self._cascade.step(context, counterpart_action)
        self.last_step_log = step_log
        self.step_logs.append(step_log)

        self._buffer.append(transition)
        self._buffer_times.append(t)
        self._policy.update(transition, model_at_action_time)

        if step_log.detected:
            if self._detect_run_start is None:
                self._detect_run_start = t
        else:
            self._detect_run_start = None

        if step_log.revised:
            if self._replay:
                if self._replay_scope == "full":
                    n = len(self._buffer)
                    self._buffer.relabel_and_replay(self._cascade.incumbent, self._policy.update)
                    first = self._buffer_times[0]
                else:
                    if self._replay_scope == "him_window":
                        first_allowed = t - len(self._cascade.history) + 1
                    else:  # post_detection
                        first_allowed = self._detect_run_start
                    selected = [
                        (tt, tr) for tt, tr in zip(self._buffer_times, self._buffer)
                        if tt >= first_allowed
                    ]
                    for _, tr in selected:
                        self._policy.update(tr, self._cascade.incumbent)
                    n = len(selected)
                    first = selected[0][0]
                self.last_replay = (t, n, first, t)
            else:
                # Ablation (Section 5.2): discard the buffered,
                # pre-acceptance experience instead of replaying it;
                # adapt only from interactions collected from here on.
                self._buffer.clear()
                self._buffer_times.clear()
                self.last_replay = (t, 0, -1, -1)
            self.replay_log.append(self.last_replay)
            self._detect_run_start = None
        self._t += 1
