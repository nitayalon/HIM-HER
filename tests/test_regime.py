from __future__ import annotations

import pytest

from himher.core.him import HIMVerifier
from himher.core.library import ModelLibrary
from himher.core.regime import Regime, select_regime
from himher.domains.handoff.construct import construct_bernoulli_model
from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Hand
from himher.domains.handoff.revise import revise_mixture_model

CONTEXTS = (0, 1)

M_L = BernoulliHandModel(0.1, 0.1)
M_R = BernoulliHandModel(0.9, 0.9)
M_C = BernoulliHandModel(0.1, 0.9)


def _window_from(model: BernoulliHandModel, n_per_context: int = 25) -> list[tuple[int, Hand]]:
    window = []
    for context in CONTEXTS:
        theta = model.theta1 if context else model.theta0
        n_right = round(theta * n_per_context)
        window += [(context, Hand.R)] * n_right
        window += [(context, Hand.L)] * (n_per_context - n_right)
    return window


def _never_improves(window):
    # A deliberately bad constructor: returns a copy of whatever the
    # incumbent already is, so it can never satisfy HIM's improvement
    # condition. Used to exercise the "no regime accepted" path.
    return M_L


def _never_revises(library, window):
    return M_L


def test_reuse_is_preferred_when_a_library_model_fits():
    library = ModelLibrary([M_L, M_R, M_C])
    incumbent = M_L
    window = _window_from(M_R)  # recent data looks like the switched-to counterpart
    verifier = HIMVerifier(tau=0.0, delta=0.1, contexts=CONTEXTS)

    outcome = select_regime(
        incumbent,
        library,
        window,
        verifier,
        revise=revise_mixture_model,
        construct=lambda w: construct_bernoulli_model(w, 1.0),
    )

    assert outcome.accepted_regime == Regime.REUSE
    assert outcome.accepted_model == M_R
    assert len(outcome.attempts) == 1  # revise/construct should not even be tried


def test_revise_is_used_when_the_target_is_a_mixture_of_two_library_models():
    # (0.5, 0.5) is exactly the theta-space midpoint of M_L and M_R, so
    # Revise should find it and Construct should never even be tried.
    library = ModelLibrary([M_L, M_R, M_C])
    incumbent = BernoulliHandModel(0.3, 0.3)  # distinct from every library entry
    window = _window_from(BernoulliHandModel(0.5, 0.5), n_per_context=200)
    verifier = HIMVerifier(tau=0.0, delta=0.1, contexts=CONTEXTS)

    outcome = select_regime(
        incumbent,
        library,
        window,
        verifier,
        revise=revise_mixture_model,
        construct=lambda w: construct_bernoulli_model(w, 1.0),
    )

    assert outcome.accepted_regime == Regime.REVISE
    assert outcome.accepted_model.theta0 == pytest.approx(0.5, abs=0.05)
    assert outcome.accepted_model.theta1 == pytest.approx(0.5, abs=0.05)
    regimes_tried = [attempt.regime for attempt in outcome.attempts]
    assert regimes_tried == [Regime.REUSE, Regime.REVISE]  # reuse tried and rejected first


def test_construct_is_used_when_no_mixture_represents_the_target():
    # (0.2, 0.8) does not lie on the segment between any two of M_L, M_R,
    # M_C, so no theta-space mixture can represent it well -- Revise
    # should be tried and rejected, and Construct should succeed.
    # incumbent=M_C so that Reuse (M_C is the closest library fit) is a
    # no-op and the comparison moves on to Revise/Construct.
    library = ModelLibrary([M_L, M_R, M_C])
    incumbent = M_C
    window = _window_from(BernoulliHandModel(0.2, 0.8), n_per_context=200)
    verifier = HIMVerifier(tau=0.0, delta=0.1, contexts=CONTEXTS)

    outcome = select_regime(
        incumbent,
        library,
        window,
        verifier,
        revise=revise_mixture_model,
        construct=lambda w: construct_bernoulli_model(w, 1.0),
    )

    assert outcome.accepted_regime == Regime.CONSTRUCT
    assert outcome.accepted_model.theta0 == pytest.approx(0.2, abs=0.05)
    assert outcome.accepted_model.theta1 == pytest.approx(0.8, abs=0.05)
    regimes_tried = [attempt.regime for attempt in outcome.attempts]
    assert regimes_tried == [Regime.REVISE, Regime.CONSTRUCT]
    assert not outcome.attempts[0].result.accepted  # Revise was tried and rejected


def test_revise_is_skipped_when_the_library_has_fewer_than_two_models():
    library = ModelLibrary([M_L])
    incumbent = BernoulliHandModel(0.3, 0.3)
    window = _window_from(M_L, n_per_context=50)
    verifier = HIMVerifier(tau=0.0, delta=0.1, contexts=CONTEXTS)

    outcome = select_regime(
        incumbent,
        library,
        window,
        verifier,
        revise=revise_mixture_model,
        construct=lambda w: construct_bernoulli_model(w, 1.0),
    )

    regimes_tried = [attempt.regime for attempt in outcome.attempts]
    assert Regime.REVISE not in regimes_tried


def test_neither_regime_accepted_returns_none_but_reports_attempts():
    library = ModelLibrary([M_L])
    incumbent = M_L
    window = _window_from(M_L)  # data matches the incumbent: nothing should look better

    verifier = HIMVerifier(tau=0.0, delta=0.0, contexts=CONTEXTS)
    outcome = select_regime(
        incumbent, library, window, verifier, revise=_never_revises, construct=_never_improves
    )

    assert outcome.accepted_regime is None
    assert outcome.accepted_model is None
    # library only has the incumbent, so reuse/revise are never attempted;
    # only construct is unconditionally tried.
    assert len(outcome.attempts) == 1


def test_rejected_candidates_are_reported_for_the_caller_to_retain():
    library = ModelLibrary([M_L, M_R, M_C])
    incumbent = BernoulliHandModel(0.3, 0.3)
    window = _window_from(BernoulliHandModel(0.5, 0.5), n_per_context=200)
    verifier = HIMVerifier(tau=0.0, delta=0.1, contexts=CONTEXTS)

    outcome = select_regime(
        incumbent, library, window, verifier, revise=_never_revises, construct=_never_improves
    )

    assert outcome.accepted_model is None
    regimes_tried = {attempt.regime for attempt in outcome.attempts}
    assert regimes_tried == {Regime.REUSE, Regime.REVISE, Regime.CONSTRUCT}
    assert all(not attempt.result.accepted for attempt in outcome.attempts)


def test_best_of_prefers_the_larger_improvement_when_reuse_and_revise_both_pass():
    # M_L/M_R midpoint is (0.5, 0.5) exactly; a window from (0.5, 0.5) makes
    # Revise a near-perfect fit while the library's best reuse candidate is
    # only a rough fit, so with both accepted best_of must pick the larger
    # improvement regardless of cost order.
    library = ModelLibrary([M_L, M_R, M_C])
    incumbent = BernoulliHandModel(0.3, 0.3)
    window = _window_from(BernoulliHandModel(0.5, 0.5), n_per_context=200)
    verifier = HIMVerifier(tau=0.0, delta=0.05, contexts=CONTEXTS)

    cost_ordered = select_regime(
        incumbent, library, window, verifier,
        revise=revise_mixture_model, construct=lambda w: construct_bernoulli_model(w, 1.0),
    )
    best = select_regime(
        incumbent, library, window, verifier,
        revise=revise_mixture_model, construct=lambda w: construct_bernoulli_model(w, 1.0),
        best_of=True,
    )

    accepted_attempts = [a for a in best.attempts if a.result.accepted]
    assert best.accepted_model == max(
        accepted_attempts, key=lambda a: a.result.min_improvement
    ).model
    assert len(best.attempts) >= len(cost_ordered.attempts)
