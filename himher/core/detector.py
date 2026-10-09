"""Model-failure detection (Section 4.1, instantiated in Section 5.2): the
aggregated surprise statistic D_t and the kappa-threshold trigger rule.

D_t is recomputed from the RecentHistory window on demand rather than
tracked incrementally, so it is always consistent with whichever model is
currently the incumbent -- including immediately after the cascade swaps
the incumbent for an accepted candidate.
"""

from __future__ import annotations

from himher.core.history import RecentHistory
from himher.core.model import CounterpartModel, history_log_likelihood


def surprise_statistic(window: RecentHistory, model: CounterpartModel) -> float:
    """D_t = (1/|window|) * sum_{tau in window} sigma_tau, where
    sigma_tau = -log P(a^j_tau | x_tau; m) (Eq. "surprise", Section 4.1;
    Eq. D_t, Section 5.2).

    Uses a partial-window average before the window has filled, so
    detection can fire before k observations have accumulated, trading
    early sensitivity for a noisier early estimate.
    """
    observations = window.as_tuple()
    if not observations:
        return 0.0
    return -history_log_likelihood(model, observations) / len(observations)


def mismatch_detected(statistic: float, kappa: float) -> bool:
    """D_t > kappa (Section 4.1)."""
    return statistic > kappa
