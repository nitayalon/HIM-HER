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
from himher.core.regime import Regime
from himher.domains.handoff.env import Context, Hand
from himher.domains.handoff.policy import TabularQPolicy, Transition


REPLAY_SCOPES = ("full", "him_window", "post_detection", "context_filtered", "op_gated",
                 "cp_joint", "cp_per_context")
TASK_F_SCOPES = ("context_filtered", "op_gated", "cp_joint", "cp_per_context")
EPS_SCHEDULES = ("constant", "decay_reset")


def unchanged_contexts(prev, accepted, delta: float, contexts=(0, 1)) -> tuple[int, ...]:
    """Contexts c whose parameter moved by at most delta at this acceptance,
    judged on the agent's own models (never the ground truth)."""
    th_p, th_a = prev.theta, accepted.theta
    return tuple(c for c in contexts if abs(th_a[c] - th_p[c]) <= delta)


def select_context_filtered(times, transitions, first_detection: int, unchanged: tuple[int, ...]):
    """Monday Task F: from the buffer (= HIM window), every transition at or
    after the detection run's first round, plus earlier ones only in
    unchanged contexts. Buffer order is kept."""
    return [(tt, tr) for tt, tr in zip(times, transitions)
            if tt >= first_detection or tr.context in unchanged]


def best_split(rounds, ll_prev, ll_acc, w_end: int) -> tuple[int, float]:
    """Task C: split s in {rounds[0], ..., w_end + 1} maximising
    LL(s) = sum_{t < s} ll_prev(t) + sum_{t >= s} ll_acc(t) over the given
    (time-ordered) interactions; ties -> the largest s. Returns
    (s_hat, LL(s_hat) - LL(first round)). With no interactions, s_hat = w_end + 1."""
    n = len(rounds)
    if n == 0:
        return w_end + 1, 0.0
    pre = [0.0]
    post = [0.0]
    for a, b in zip(ll_prev, ll_acc):
        pre.append(pre[-1] + a)
        post.append(post[-1] + b)
    total_acc = post[-1]
    best_i, best = 0, pre[0] + (total_acc - post[0])
    ll0 = best
    for i in range(1, n + 1):  # split before element i (i = n: after the last)
        v = pre[i] + (total_acc - post[i])
        if v >= best:  # >= : ties go to the larger split
            best_i, best = i, v
    s_hat = rounds[best_i] if best_i < n else w_end + 1
    return s_hat, best - ll0


def select_changepoint(times, transitions, prev, accepted, delta: float, per_context: bool):
    """Task C (cp_joint / cp_per_context): replay every window interaction in
    contexts whose parameter moved by <= delta, and in changed contexts only
    interactions at or after the estimated change point. Likelihoods are the
    codebase's own BernoulliHandModel.log_prob; ground truth is never used.
    Returns (selected [(t, transition)], changed contexts, s_hat dict, LL gain)."""
    changed = tuple(c for c in (0, 1) if c not in unchanged_contexts(prev, accepted, delta))
    times, transitions = list(times), list(transitions)
    w_end = times[-1]
    ll = [(t, tr, prev.log_prob(tr.context, tr.counterpart_action),
           accepted.log_prob(tr.context, tr.counterpart_action)) for t, tr in zip(times, transitions)]
    s_hat: dict[str, int] = {}
    gain = 0.0
    if per_context:
        cut = {}
        for c in changed:
            sub = [x for x in ll if x[1].context == c]
            cut[c], g = best_split([x[0] for x in sub], [x[2] for x in sub], [x[3] for x in sub], w_end)
            s_hat[f"ctx{c}"] = cut[c]
            gain += g
        keep = lambda t, tr: tr.context not in changed or t >= cut[tr.context]  # noqa: E731
    else:
        # unchanged-context interactions contribute 0 to both sums, so the
        # split over all window rounds equals the split over changed ones
        # with ties resolved to the largest round.
        rounds = [x[0] for x in ll]
        lp = [x[2] if x[1].context in changed else 0.0 for x in ll]
        la = [x[3] if x[1].context in changed else 0.0 for x in ll]
        cut_all, gain = best_split(rounds, lp, la, w_end)
        s_hat["joint"] = cut_all
        keep = lambda t, tr: tr.context not in changed or t >= cut_all  # noqa: E731
    selected = [(t, tr) for t, tr in zip(times, transitions) if keep(t, tr)]
    return selected, changed, s_hat, gain


def decay_reset_epsilon(t: int, t_reset: int, eps0: float, eps_min: float, half_life: float) -> float:
    """eps_t = max(eps_min, eps0 * 2^(-(t - t_reset) / h)) (rebuttal D3)."""
    return max(eps_min, eps0 * 2.0 ** (-(t - t_reset) / half_life))


class HHAgent:
    def __init__(
        self,
        cascade: HHCascade,
        policy: TabularQPolicy,
        rng: np.random.Generator,
        replay: bool = True,
        replay_scope: str = "full",
        eps_schedule: str = "constant",
        eps0: float = 0.15,
        eps_min: float = 0.02,
        eps_half_life: float = 50.0,
        context_delta: float = 0.1,
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
          - "context_filtered" (Monday Task F): the post_detection set plus
            pre-detection buffered transitions in contexts c with
            |theta_acc,c - theta_prev,c| <= ``context_delta`` (theta_prev =
            the incumbent before this round's revision; agent's models only).
          - "op_gated" (Monday Task F): as "full", but nothing is replayed
            when the accepted operator is reuse (the buffer is kept).
          - "cp_joint" / "cp_per_context" (Monday Task C): in contexts whose
            parameter moved by more than ``context_delta``, replay only the
            window interactions at or after a change point estimated by
            maximum likelihood under (theta_prev before, theta_acc after);
            one split for the whole window (joint) or one per changed
            context. Unchanged contexts: the whole window.

        ``eps_schedule`` (rebuttal Task D):
          - "constant" (default): the policy's own fixed epsilon. Unchanged.
          - "decay_reset": before acting in round t, set the policy's
            epsilon to max(eps_min, eps0 * 2^(-(t - t_reset)/eps_half_life)),
            where t_reset is 0 at episode start and is set to the round of
            every detection (mismatch flagged), whether or not a revision
            is then accepted. A detection in round t is only known after
            round t's action, so the first action using the reset is round
            t+1 (epsilon = eps0 * 2^(-1/h)).
        """
        if eps_schedule not in EPS_SCHEDULES:
            raise ValueError(f"eps_schedule must be one of {EPS_SCHEDULES}")
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
        # Task F logging (None for the pre-existing scopes, so their event
        # logs are unchanged): per-context replay counts and the contexts
        # judged unchanged at the last acceptance.
        self.last_replay_detail: dict | None = None
        self._context_delta = context_delta
        self.last_step_log: CascadeStepLog | None = None
        self.step_logs: list[CascadeStepLog] = []
        self._eps_schedule = eps_schedule
        self._eps_params = (eps0, eps_min, eps_half_life)
        self._t_reset = 0
        self.eps_reset_times: list[int] = [0] if eps_schedule == "decay_reset" else []
        self.last_epsilon: float | None = None

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

    @property
    def eps_schedule(self) -> str:
        return self._eps_schedule

    def act(self, context: Context) -> Hand:
        if self._eps_schedule == "decay_reset":
            self._policy.epsilon = decay_reset_epsilon(self._t, self._t_reset, *self._eps_params)
        self.last_epsilon = self._policy.epsilon
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
            if self._eps_schedule == "decay_reset":
                self._t_reset = t
                self.eps_reset_times.append(t)
        else:
            self._detect_run_start = None

        if step_log.revised:
            if self._replay:
                if self._replay_scope in TASK_F_SCOPES:
                    self._replay_task_f(t, model_at_action_time, step_log)
                elif self._replay_scope == "full":
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
                if self._replay_scope not in TASK_F_SCOPES:
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

    def _replay_task_f(self, t: int, prev_model, step_log: CascadeStepLog) -> None:
        accepted = self._cascade.incumbent
        extra: dict = {}
        if self._replay_scope in ("cp_joint", "cp_per_context"):
            unchanged = ()
            selected, changed, s_hat, gain = select_changepoint(
                self._buffer_times, self._buffer, prev_model, accepted, self._context_delta,
                per_context=self._replay_scope == "cp_per_context")
            extra = {"changed_contexts": ";".join(map(str, changed)) if changed else "none",
                     "ll_gain": gain}
            if self._replay_scope == "cp_joint":
                extra["s_hat"] = s_hat["joint"]
            else:
                for c in (0, 1):
                    extra[f"s_hat_ctx{c}"] = s_hat.get(f"ctx{c}", -1)
        elif self._replay_scope == "op_gated":
            unchanged: tuple[int, ...] = ()
            if step_log.regime == Regime.REUSE:
                selected = []
            else:
                selected = list(zip(self._buffer_times, self._buffer))
        else:
            unchanged = unchanged_contexts(prev_model, accepted, self._context_delta)
            selected = select_context_filtered(self._buffer_times, self._buffer,
                                               self._detect_run_start, unchanged)
        for _, tr in selected:
            self._policy.update(tr, accepted)
        n = len(selected)
        self.last_replay = (t, n, selected[0][0], t) if n else (t, 0, -1, -1)
        self.last_replay_detail = {
            "n_replayed_ctx0": sum(1 for _, tr in selected if tr.context == 0),
            "n_replayed_ctx1": sum(1 for _, tr in selected if tr.context == 1),
        }
        if extra:
            self.last_replay_detail |= extra
        else:
            self.last_replay_detail["unchanged_contexts"] = ";".join(map(str, unchanged)) if unchanged else "none"
