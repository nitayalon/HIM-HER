"""Regime selection (Section 4.4): having decided a revision is warranted,
try candidates in cost order -- Reuse (retrieve a stored model), then
Revise (interpolate between two stored models), then Construct (generate
a new model from scratch) -- matching Section 4.4's cost ordering and
Figure 3's "resource-rational agent escalates only when the cheaper regime
fails". HIM gates every candidate identically (Definition 4.1), against
the full comparison pool {incumbent} u lib_t (lib_t here contains only
previously-selected models -- see HHCascade and himher.core.him's
module docstring for why that matters).

``best_of`` is a variant to the cost-ordered rule: Reuse and Revise are
both evaluated, and the accepted candidate with the larger improvement over
the incumbent wins; Construct is tried only if neither is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence

from himher.core.him import HIMResult, HIMVerifier
from himher.core.library import ModelLibrary
from himher.core.model import Action, Context, CounterpartModel


class Regime(Enum):
    REUSE = "reuse"
    REVISE = "revise"
    CONSTRUCT = "construct"


Constructor = Callable[[Sequence[tuple[Context, Action]]], CounterpartModel]
Reviser = Callable[[ModelLibrary, Sequence[tuple[Context, Action]]], CounterpartModel | None]


@dataclass(frozen=True)
class Attempt:
    regime: Regime
    model: CounterpartModel
    result: HIMResult


@dataclass(frozen=True)
class RegimeOutcome:
    accepted_regime: Regime | None  # None if no candidate was accepted
    accepted_model: CounterpartModel | None
    attempts: tuple[Attempt, ...]  # every candidate tried, in order, win or lose


def select_regime(
    incumbent: CounterpartModel,
    library: ModelLibrary,
    window: Sequence[tuple[Context, Action]],
    verifier: HIMVerifier,
    revise: Reviser,
    construct: Constructor,
    best_of: bool = False,
) -> RegimeOutcome:
    attempts: list[Attempt] = []

    def comparison_pool(candidate: CounterpartModel) -> tuple[CounterpartModel, ...]:
        return tuple(m for m in library if m != incumbent and m != candidate)

    reuse_candidate = library.best_fit(window)
    reuse_attempt = None
    if reuse_candidate is not None and reuse_candidate != incumbent:
        reuse_result = verifier.evaluate(reuse_candidate, incumbent, comparison_pool(reuse_candidate), window)
        reuse_attempt = Attempt(Regime.REUSE, reuse_candidate, reuse_result)
        attempts.append(reuse_attempt)
        if reuse_result.accepted and not best_of:
            return RegimeOutcome(Regime.REUSE, reuse_candidate, tuple(attempts))

    revised_candidate = revise(library, window)
    revise_attempt = None
    if revised_candidate is not None and revised_candidate != incumbent:
        revise_result = verifier.evaluate(revised_candidate, incumbent, comparison_pool(revised_candidate), window)
        revise_attempt = Attempt(Regime.REVISE, revised_candidate, revise_result)
        attempts.append(revise_attempt)
        if revise_result.accepted and not best_of:
            return RegimeOutcome(Regime.REVISE, revised_candidate, tuple(attempts))

    if best_of:
        accepted = [a for a in (reuse_attempt, revise_attempt) if a is not None and a.result.accepted]
        if accepted:
            winner = max(accepted, key=lambda a: a.result.min_improvement)
            return RegimeOutcome(winner.regime, winner.model, tuple(attempts))

    constructed = construct(window)
    construct_result = verifier.evaluate(constructed, incumbent, comparison_pool(constructed), window)
    attempts.append(Attempt(Regime.CONSTRUCT, constructed, construct_result))
    if construct_result.accepted:
        return RegimeOutcome(Regime.CONSTRUCT, constructed, tuple(attempts))

    return RegimeOutcome(None, None, tuple(attempts))
