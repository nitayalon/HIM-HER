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

import numpy as np

from himher.core.cascade import CascadeStepLog, HHCascade
from himher.core.her import ReplayBuffer
from himher.domains.handoff.env import Context, Hand
from himher.domains.handoff.policy import TabularQPolicy, Transition


class HHAgent:
    def __init__(
        self,
        cascade: HHCascade,
        policy: TabularQPolicy,
        rng: np.random.Generator,
        replay: bool = True,
    ) -> None:
        """``replay`` toggles Section 5.2's ablation: when False, the
        buffer of pre-acceptance transitions is discarded on revision
        instead of being replayed under the newly accepted model.
        """
        self._cascade = cascade
        self._policy = policy
        self._rng = rng
        self._replay = replay
        self._buffer: ReplayBuffer[Transition] = ReplayBuffer(maxlen=cascade.window)
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

    def act(self, context: Context) -> Hand:
        return self._policy.act(context, self._cascade.incumbent, self._rng)

    def update(
        self, context: Context, ego_action: Hand, counterpart_action: Hand, reward: float
    ) -> None:
        transition = Transition(context, ego_action, counterpart_action, reward)
        model_at_action_time = self._cascade.incumbent

        step_log = self._cascade.step(context, counterpart_action)
        self.last_step_log = step_log
        self.step_logs.append(step_log)

        self._buffer.append(transition)
        self._policy.update(transition, model_at_action_time)

        if step_log.revised:
            if self._replay:
                self._buffer.relabel_and_replay(self._cascade.incumbent, self._policy.update)
            else:
                # Ablation (Section 5.2): discard the buffered,
                # pre-acceptance experience instead of replaying it;
                # adapt only from interactions collected from here on.
                self._buffer.clear()
