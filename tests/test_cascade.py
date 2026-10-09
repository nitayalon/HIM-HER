"""Integration tests for HHCascade: Algorithm 1's detect-through-verify
stages wired together. Unit-level correctness of detection, HIM, and
regime selection is already covered in their own test files; these tests
check end-to-end wiring (library/incumbent bookkeeping) using scenarios
with deterministic (non-stochastic) action sequences, so results are
exactly reproducible.
"""

from __future__ import annotations

import pytest

from himher.core.cascade import HHCascade
from himher.core.him import HIMVerifier
from himher.core.library import ModelLibrary
from himher.core.regime import Regime
from himher.domains.handoff.construct import construct_bernoulli_model
from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Hand
from himher.domains.handoff.revise import revise_mixture_model

CONTEXTS = (0, 1)

M_L = BernoulliHandModel(0.1, 0.1)
M_R = BernoulliHandModel(0.9, 0.9)
M_C = BernoulliHandModel(0.1, 0.9)


def _make_cascade(
    incumbent: BernoulliHandModel, kappa: float = 0.3, window: int = 10, tau: float = 0.5
) -> HHCascade:
    # tau=0.5 (not 0.0): with Revise in the mix, a permissive tau lets a
    # mixture candidate fit to a still-transitional window get accepted
    # before the window has fully purified, which these deterministic
    # long-running scenarios aren't designed to tolerate (confirmed by
    # direct comparison: tau=0.0 converges to an interpolated (0.83, 0.5)
    # instead of cleanly reaching m_R). tau=0.5 matches the production
    # default chosen in scripts/run_experiments.py.
    return HHCascade(
        incumbent=incumbent,
        library=ModelLibrary([M_L, M_R, M_C]),
        kappa=kappa,
        window=window,
        verifier=HIMVerifier(tau=tau, delta=0.1, contexts=CONTEXTS),
        revise=revise_mixture_model,
        construct=lambda w: construct_bernoulli_model(w, 1.0),
    )


def test_stationary_counterpart_never_triggers_detection():
    cascade = _make_cascade(incumbent=M_L)
    for _ in range(50):
        log = cascade.step(0, Hand.L)
        assert not log.detected
        assert not log.revised
    assert cascade.incumbent == M_L
    assert len(cascade.library) == 3  # unchanged


def test_within_library_switch_converges_to_the_exact_library_model_via_reuse():
    cascade = _make_cascade(incumbent=M_L)

    for _ in range(20):
        cascade.step(0, Hand.L)  # stationary phase, no detection

    saw_reuse = False
    for _ in range(100):
        log = cascade.step(0, Hand.R)  # counterpart has switched to m_R
        if log.revised:
            saw_reuse = saw_reuse or log.regime == Regime.REUSE

    assert saw_reuse
    assert cascade.incumbent == M_R


def test_old_incumbent_is_archived_into_the_library_on_acceptance():
    cascade = _make_cascade(incumbent=M_L)
    for _ in range(20):
        cascade.step(0, Hand.L)
    for _ in range(100):
        cascade.step(0, Hand.R)

    assert cascade.incumbent == M_R
    assert M_L in cascade.library  # the displaced incumbent is retained


def test_mixture_representable_switch_uses_revise():
    # (0.5, 0.5) is exactly the theta-space midpoint of M_L and M_R, so
    # this -- unlike the genuinely novel case below -- should resolve via
    # Revise, not Construct. The window is pre-seeded directly to avoid
    # the transient, partially-mixed windows a natural multi-round
    # switchover would produce.
    cascade = _make_cascade(incumbent=BernoulliHandModel(0.3, 0.3))
    pattern = [Hand.R, Hand.L, Hand.R, Hand.L, Hand.R, Hand.L, Hand.R, Hand.L, Hand.R]
    for action in pattern:
        cascade.history.append(0, action)

    log = cascade.step(0, Hand.L)  # completes a pure, 10-observation 50/50 window

    assert log.detected
    assert log.revised
    assert log.regime == Regime.REVISE
    assert cascade.incumbent.theta0 == 0.5
    assert cascade.incumbent.theta1 == 0.5


def test_outside_library_switch_to_a_fully_novel_counterpart_uses_construct():
    # (0.2, 0.8) does not lie on the segment between any two library
    # models (same scenario validated directly against select_regime in
    # test_regime.py), so Revise cannot represent it and Construct must
    # be used. incumbent=M_C so Reuse (M_C is the closest library fit) is
    # a no-op; window=400 (200 per context) is pre-seeded directly so the
    # full window is available in one step rather than accumulating
    # through many transient, partially-mixed rounds.
    cascade = _make_cascade(incumbent=M_C, window=400)
    for _ in range(160):
        cascade.history.append(0, Hand.L)
    for _ in range(39):
        cascade.history.append(0, Hand.R)  # 199 context-0 rounds so far, 39/199 ~= 0.2 R
    for _ in range(160):
        cascade.history.append(1, Hand.R)
    for _ in range(40):
        cascade.history.append(1, Hand.L)  # 200 context-1 rounds, complete: 160/200 = 0.8 R

    log = cascade.step(0, Hand.R)  # 200th context-0 round, completes the window

    assert log.detected
    assert log.revised
    assert log.regime == Regime.CONSTRUCT
    assert cascade.incumbent.theta0 == pytest.approx(0.2, abs=0.05)
    assert cascade.incumbent.theta1 == pytest.approx(0.8, abs=0.05)


def test_rejected_attempts_are_logged_but_never_enter_the_library():
    # tau impossibly high: nothing can ever be accepted. Rejected
    # candidates must be recorded for inspection (Section 4.4), but must
    # NOT be added to the library -- lib_t holds only previously-selected
    # models (see himher.core.him's module docstring for why mixing
    # rejected attempts into lib_t creates a recovery-blocking gridlock).
    cascade = HHCascade(
        incumbent=M_L,
        library=ModelLibrary([M_L, M_R, M_C]),
        kappa=0.3,
        window=10,
        verifier=HIMVerifier(tau=1e9, delta=0.0, contexts=CONTEXTS),
        revise=revise_mixture_model,
        construct=lambda w: construct_bernoulli_model(w, 1.0),
    )
    for _ in range(20):
        cascade.step(0, Hand.L)

    library_size_before = len(cascade.library)
    rejected_before = len(cascade.rejected)
    log = cascade.step(0, Hand.R)

    assert not log.revised
    assert len(cascade.library) == library_size_before
    assert len(cascade.rejected) > rejected_before
    assert cascade.incumbent == M_L  # unchanged, since nothing was accepted


def test_min_revision_fill_blocks_revisions_until_the_window_is_full():
    cascade = HHCascade(
        incumbent=M_L,
        library=ModelLibrary([M_L, M_R, M_C]),
        kappa=0.3,
        window=10,
        verifier=HIMVerifier(tau=0.5, delta=0.1, contexts=CONTEXTS),
        revise=revise_mixture_model,
        construct=lambda w: construct_bernoulli_model(w, 1.0),
        min_revision_fill=10,
    )
    early = [cascade.step(0, Hand.R) for _ in range(9)]
    assert all(not log.revised for log in early)
    assert any(log.detected for log in early)  # detection still fires, revision waits
