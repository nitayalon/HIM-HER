"""Debugging/inspection helpers -- not part of the cascade's production
path, used by scripts/run_debug_trace.py to answer "is the framework
behaving as intended?" by exposing what HIM and the detector are actually
seeing, round by round.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from himher.core.model import Action, Context, CounterpartModel, history_log_likelihood


def likelihood_table(
    models: Mapping[str, CounterpartModel], window: Sequence[tuple[Context, Action]]
) -> dict[str, float]:
    """Average per-step log-likelihood of ``window`` under each named
    model -- "how well each model captures the observed behavior"
    (Section 5.5 measure 4's underlying quantity), reported per-step
    rather than summed so it's comparable across windows of different
    lengths and reads directly as negative surprise.
    """
    if not window:
        return {name: 0.0 for name in models}
    return {name: history_log_likelihood(model, window) / len(window) for name, model in models.items()}
