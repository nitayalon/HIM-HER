"""Hindsight Inconsistency Mitigation (Section 4.4, Definition 4.1): the
acceptance test a candidate model m' must pass before it replaces the
incumbent, whether m' came from Reuse, Revise, or Construct -- Algorithm 1
applies this test uniformly regardless of where the candidate came from,
so there is a single verifier rather than a separate, looser check per
regime.

Both conditions are checked against the full comparison set {m} u lib_t,
exactly as Definition 4.1 states. Earlier revisions of this module instead
scoped both conditions to the incumbent only, to work around a
recovery-blocking gridlock: if lib_t accumulates every *rejected* attempt
(as an earlier version of the cascade did), it fills with near-duplicates
of whatever the true counterpart actually is -- since every attempt, even
a rejected one, is fit to real data from the true process -- and the
correct model can then never clear "better than and distinct from
everything in lib_t," including its own accumulated near-clones.

The actual fix is upstream, not here: lib_t must only contain *selected*
models (Section 4.4's Reuse bullet: lib_t is "the set of previously used
models," i.e. previously an incumbent, not previously attempted).
HHCascade enforces this -- rejected candidates are logged for inspection
but never added to the library -- which is what makes the literal,
full-pool comparison in this module safe again.

ell(m) = log p(h_{t-k:t}; m), with no complexity penalty: Definition 4.1
does not include one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from himher.core.model import Action, Context, CounterpartModel, history_log_likelihood, model_divergence


def score(model: CounterpartModel, window: Sequence[tuple[Context, Action]]) -> float:
    """ell(m) = log p(h_{t-k:t}; m) (Definition 4.1)."""
    return history_log_likelihood(model, window)


@dataclass(frozen=True)
class HIMResult:
    accepted: bool
    score: float  # ell(m') for the candidate, for logging/diagnostics
    min_improvement: float  # binding value of condition (i), over the full comparison pool
    min_distinctness: float  # binding value of condition (ii), over the full comparison pool


class HIMVerifier:
    """tau, delta are the calibration parameters phi governing acceptance
    (Section 4.6). ``contexts`` is the representative set of contexts used
    to compute the distinctness divergence d(m, m') (Definition 4.1's
    "divergence between the predictive policies").
    """

    def __init__(self, tau: float, delta: float, contexts: Sequence[Context]) -> None:
        self._tau = tau
        self._delta = delta
        self._contexts = tuple(contexts)
        if not self._contexts:
            raise ValueError("contexts must be non-empty")

    def evaluate(
        self,
        candidate: CounterpartModel,
        incumbent: CounterpartModel,
        comparison_pool: Sequence[CounterpartModel],
        window: Sequence[tuple[Context, Action]],
    ) -> HIMResult:
        """``comparison_pool`` is lib_t with the incumbent and the
        candidate itself excluded. The candidate must be excluded: a
        reuse candidate is, by construction, drawn from lib_t, and
        comparing it against itself would give ell(m') - ell(m') = 0,
        which could never exceed tau >= 0 -- so Definition 4.1's "for all
        m_tilde in {m} u lib_t" is read here as implicitly excluding m'
        itself, the only reading under which reuse can ever be accepted.
        """
        comparison_set = (incumbent, *comparison_pool)
        candidate_score = score(candidate, window)

        improvements = [candidate_score - score(m, window) for m in comparison_set]
        distinctnesses = [
            model_divergence(candidate, m, self._contexts) for m in comparison_set
        ]

        min_improvement = min(improvements)
        min_distinctness = min(distinctnesses)
        accepted = (min_improvement > self._tau) and (min_distinctness >= self._delta)

        return HIMResult(
            accepted=accepted,
            score=candidate_score,
            min_improvement=min_improvement,
            min_distinctness=min_distinctness,
        )
