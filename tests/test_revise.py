from __future__ import annotations

import pytest

from himher.core.library import ModelLibrary
from himher.core.model import history_log_likelihood
from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Hand
from himher.domains.handoff.revise import interpolate, revise_mixture_model

M_L = BernoulliHandModel(0.1, 0.1)
M_R = BernoulliHandModel(0.9, 0.9)
M_C = BernoulliHandModel(0.1, 0.9)


def _window_from(model: BernoulliHandModel, n_per_context: int = 200) -> list[tuple[int, Hand]]:
    window = []
    for context in (0, 1):
        theta = model.theta1 if context else model.theta0
        n_right = round(theta * n_per_context)
        window += [(context, Hand.R)] * n_right
        window += [(context, Hand.L)] * (n_per_context - n_right)
    return window


def test_interpolate_at_p_zero_and_one_returns_the_endpoints():
    assert interpolate(M_L, M_R, p=0.0) == M_L
    assert interpolate(M_L, M_R, p=1.0) == M_R


def test_interpolate_at_p_half_is_the_componentwise_midpoint():
    mixed = interpolate(M_L, M_C, p=0.5)
    assert mixed.theta0 == pytest.approx(0.1)
    assert mixed.theta1 == pytest.approx(0.5)


def test_revise_returns_none_for_a_library_with_fewer_than_two_models():
    library = ModelLibrary([M_L])
    assert revise_mixture_model(library, _window_from(M_L)) is None

    empty_library = ModelLibrary([])
    assert revise_mixture_model(empty_library, _window_from(M_L)) is None


def test_revise_recovers_the_exact_midpoint_when_the_target_is_a_mixture():
    # m_2 = (0.1, 0.5) from Section 5.3 is exactly the M_L/M_C midpoint.
    library = ModelLibrary([M_L, M_R, M_C])
    window = _window_from(BernoulliHandModel(0.1, 0.5))

    candidate = revise_mixture_model(library, window)

    assert candidate.theta0 == pytest.approx(0.1, abs=0.02)
    assert candidate.theta1 == pytest.approx(0.5, abs=0.02)


def test_revise_cannot_recover_a_target_off_every_segment():
    # m_1 = (0.2, 0.8) does not lie on the segment between any two
    # library models, so the best mixture should fit noticeably worse
    # than it would if the target were actually representable.
    library = ModelLibrary([M_L, M_R, M_C])
    target = BernoulliHandModel(0.2, 0.8)
    window = _window_from(target)

    candidate = revise_mixture_model(library, window)
    representable_target = BernoulliHandModel(0.1, 0.5)
    representable_window = _window_from(representable_target)
    representable_candidate = revise_mixture_model(library, representable_window)

    unrepresentable_gap = history_log_likelihood(
        target, window
    ) - history_log_likelihood(candidate, window)
    representable_gap = history_log_likelihood(
        representable_target, representable_window
    ) - history_log_likelihood(representable_candidate, representable_window)

    assert unrepresentable_gap > representable_gap


def test_revise_picks_the_best_scoring_pair_and_weight():
    # A window exactly matching m_R should make Revise converge toward
    # m_R itself (e.g. via the M_L/M_R pair at p=1), not some other pair.
    library = ModelLibrary([M_L, M_R, M_C])
    window = _window_from(M_R)

    candidate = revise_mixture_model(library, window)

    assert candidate.theta0 == pytest.approx(0.9, abs=0.02)
    assert candidate.theta1 == pytest.approx(0.9, abs=0.02)
