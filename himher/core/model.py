"""Domain-agnostic counterpart model interface and generic helpers.

A ``CounterpartModel`` is m_j from Definition 2.1 (Social world model): a
simulable model of agent j. It must satisfy Assumption 2.2 (Simulability) by
exposing a predictive distribution over actions, from which both a
log-likelihood and a sampler can be derived. Any object with this shape
works with the detector (Eq. 1) and HIM (Definition 4.1) below -- neither
module needs to know how the model is represented internally.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

Context = Any
Action = Any


@runtime_checkable
class CounterpartModel(Protocol):
    """A simulable model of a counterpart agent j."""

    def predict_proba(self, context: Context) -> np.ndarray:
        """Predictive distribution over actions given context.

        Returned as a probability vector in a fixed, model-specific action
        ordering; callers that need log_prob/sample for a single action get
        them for free via the mixins below, as long as this is implemented.
        """
        ...

    def log_prob(self, context: Context, action: Action) -> float:
        """log P(a^j = action | context; this model)."""
        ...


def history_log_likelihood(
    model: CounterpartModel, history: Sequence[tuple[Context, Action]]
) -> float:
    """log p(h_{t-k:t}; m) = sum_tau log P(a^j_tau | x_tau; m).

    Shared by the detector (surprise, Eq. 1) and HIM (ell(m), Definition 4.1)
    -- both reduce to summing this model's log_prob over a window of
    observed (context, action) pairs.
    """
    return float(sum(model.log_prob(context, action) for context, action in history))


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """TV distance between two discrete distributions over the same support."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    return 0.5 * float(np.abs(p - q).sum())


def model_divergence(
    m: CounterpartModel, m_prime: CounterpartModel, contexts: Sequence[Context]
) -> float:
    """d(m, m') from Definition 4.1: divergence between the predictive
    policies the two models induce, averaged over a representative set of
    contexts. Domain-agnostic given ``predict_proba`` -- no access to either
    model's internal parameters is required.
    """
    if not contexts:
        raise ValueError("model_divergence requires at least one context")
    distances = [
        total_variation_distance(m.predict_proba(c), m_prime.predict_proba(c)) for c in contexts
    ]
    return float(np.mean(distances))
