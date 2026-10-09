"""Lifelong calibration (Section 4.6): adapting phi = (kappa, k, tau,
delta) from the outcome of each completed cascade step. The paper leaves
the update rule an open problem (Section 6); the toy illustration
calibrates kappa externally instead, by sweeping fixed values across runs
(Section 5.2), rather than adapting it online. ``NoOpCalibrator``
preserves calibration's place in the cascade and records outcomes for
offline inspection, without changing any threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from himher.core.regime import Regime


@dataclass(frozen=True)
class CascadeOutcome:
    detected: bool
    accepted_regime: Regime | None


@runtime_checkable
class Calibrator(Protocol):
    def update(self, outcome: CascadeOutcome) -> None: ...


class NoOpCalibrator:
    def __init__(self) -> None:
        self.history: list[CascadeOutcome] = []

    def update(self, outcome: CascadeOutcome) -> None:
        self.history.append(outcome)
