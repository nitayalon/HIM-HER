"""Hindsight relabeling (Section 4.5): replaying buffered transitions
under a newly accepted model instead of the incumbent they were collected
under, so "no interaction is wasted" (Section 4.5).

This buffer is domain-agnostic: it stores whatever transition type a
domain defines and replays them through a caller-supplied update
function. Relabeling itself -- whether that means recomputing anything
about the transition, or (as in the Switching Handoff Game, where the
counterpart's action is always fully observed) simply replaying the same
data against a different model -- is entirely up to that update function.

``maxlen`` matters for soundness, not just memory: Remark 4.2 licenses
relabeling because HIM's hindsight-consistency condition certified that
the accepted model explains the verification window h_{t-k:t} -- a
*specific*, length-k window, not the agent's entire history. A buffer
that grows without bound (e.g. across a long stationary stretch before
the first-ever revision) would contain transitions HIM never checked
anything about, and relabeling those would not be licensed by anything in
the paper. Passing ``maxlen`` equal to the cascade's window keeps the
buffer's scope identical to what HIM actually verified.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Generic, Iterator, TypeVar

TransitionT = TypeVar("TransitionT")
ModelT = TypeVar("ModelT")


class ReplayBuffer(Generic[TransitionT]):
    def __init__(self, maxlen: int | None = None) -> None:
        self._transitions: Deque[TransitionT] = deque(maxlen=maxlen)

    def append(self, transition: TransitionT) -> None:
        self._transitions.append(transition)

    def __len__(self) -> int:
        return len(self._transitions)

    def __iter__(self) -> Iterator[TransitionT]:
        return iter(self._transitions)

    def clear(self) -> None:
        self._transitions.clear()

    def relabel_and_replay(
        self, model: ModelT, update_fn: Callable[[TransitionT, ModelT], None]
    ) -> None:
        """Replay every buffered transition against ``model`` instead of
        whichever incumbent was active when it was collected (Remark 4.2:
        sound exactly when ``model`` has passed HIM's hindsight-
        consistency condition against this buffer's window).
        """
        for transition in self._transitions:
            update_fn(transition, model)
