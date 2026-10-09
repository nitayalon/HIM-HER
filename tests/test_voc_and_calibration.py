from __future__ import annotations

from himher.core.calibration import CascadeOutcome, NoOpCalibrator
from himher.core.regime import Regime
from himher.core.voc import AlwaysProceedVOC


def test_always_proceed_voc_always_returns_true():
    voc = AlwaysProceedVOC()
    assert voc.should_revise() is True
    assert voc.should_revise() is True


def test_noop_calibrator_records_but_does_not_transform_outcomes():
    calibrator = NoOpCalibrator()
    outcome_a = CascadeOutcome(detected=True, accepted_regime=Regime.REUSE)
    outcome_b = CascadeOutcome(detected=False, accepted_regime=None)

    calibrator.update(outcome_a)
    calibrator.update(outcome_b)

    assert calibrator.history == [outcome_a, outcome_b]
