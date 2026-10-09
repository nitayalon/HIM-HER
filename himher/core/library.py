"""Model library (Section 3): the set of previously held models, including
displaced incumbents and HIM-rejected candidates (Section 4.4 -- "rejected
candidates are retained in lib_t, so that future construction can learn
from them"). Models are deduplicated by value, so a model that is
constructed, forgotten, and later reconstructed identically is recognized
as the same entry (Section 5.3's Recurring-counterpart condition).

``max_learned`` bounds how many *learned* entries (displaced incumbents
and rejected candidates) are retained, evicting the oldest first; the
``initial_models`` ("core" entries, e.g. the hand-specified lib_0 of
Section 5.1) are never evicted. This isn't in the paper's literal
Algorithm 1 -- Definition 4.1 says "retained" with no eviction -- but an
unbounded library creates a real pathology: HIM's acceptance test
requires beating *every* comparison-pool member by margin tau, so as
noise-driven rejects accumulate without limit, the odds that *some*
accumulated entry is a coincidentally-good fit for the current window
keep rising, making it progressively harder for the cascade to ever
recover a correct model once it drifts (confirmed empirically: ~12.5% of
stationary-condition episodes get permanently stuck on an early,
noise-accepted wrong model once the library grows into the dozens).
Bounding the learned portion keeps that recency window finite without
touching the hand-specified core hypotheses.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterator, Sequence

from himher.core.model import Action, Context, CounterpartModel, history_log_likelihood


class ModelLibrary:
    def __init__(
        self, initial_models: Sequence[CounterpartModel] = (), max_learned: int | None = None
    ) -> None:
        self._core: list[CounterpartModel] = list(dict.fromkeys(initial_models))
        self._learned: Deque[CounterpartModel] = deque(maxlen=max_learned)

    def __len__(self) -> int:
        return len(self._core) + len(self._learned)

    def __iter__(self) -> Iterator[CounterpartModel]:
        return iter((*self._core, *self._learned))

    def __contains__(self, model: CounterpartModel) -> bool:
        return model in self._core or model in self._learned

    def add(self, model: CounterpartModel) -> None:
        """Insert ``model`` as a learned entry if not already present
        (by value) among the core or learned entries. Exceeding
        ``max_learned`` evicts the oldest learned entry; core entries are
        never evicted.
        """
        if model not in self:
            self._learned.append(model)

    def best_fit(
        self, window: Sequence[tuple[Context, Action]]
    ) -> CounterpartModel | None:
        """argmax_{m in lib} log p(window; m); None if the library is empty."""
        models = (*self._core, *self._learned)
        if not models:
            return None
        return max(models, key=lambda m: history_log_likelihood(m, window))
