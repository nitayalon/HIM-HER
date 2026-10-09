"""A fixed-length sliding window of recent (context, action) observations.

Shared by detection (the surprise statistic D_t, Section 4.1), regime
selection and HIM verification (the window h_{t-k:t} in Definition 4.1),
and model construction (Section 5.2) -- all three are defined over the
same trailing window, so there is a single source of truth for it rather
than separate buffers that could drift out of sync.
"""

from __future__ import annotations

from collections import deque
from typing import Deque

from himher.core.model import Action, Context


class RecentHistory:
    def __init__(self, window: int) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self._window = window
        self._observations: Deque[tuple[Context, Action]] = deque(maxlen=window)

    def append(self, context: Context, action: Action) -> None:
        self._observations.append((context, action))

    def __len__(self) -> int:
        return len(self._observations)

    @property
    def is_full(self) -> bool:
        return len(self._observations) == self._window

    def as_tuple(self) -> tuple[tuple[Context, Action], ...]:
        return tuple(self._observations)
