"""The H+H cascade (Algorithm 1): one step of detection, regime
selection, and verification for a single counterpart.

This orchestrates Stages 1-4 (detect, VOC, regime-select, HIM-verify) and
the incumbent/library bookkeeping that follows acceptance or rejection.
Stage 5 (HER-based policy adaptation) is deliberately not done here: it
needs a replay buffer and an ego policy, both of which are domain-
specific, so it lives in the calling agent (see
himher.baselines.hh_agent.HHAgent), which triggers replay whenever a step
reports ``revised=True``.
"""

from __future__ import annotations

from dataclasses import dataclass

from himher.core.calibration import Calibrator, CascadeOutcome, NoOpCalibrator
from himher.core.detector import mismatch_detected, surprise_statistic
from himher.core.him import HIMVerifier
from himher.core.history import RecentHistory
from himher.core.library import ModelLibrary
from himher.core.model import Action, Context, CounterpartModel
from himher.core.regime import Attempt, Constructor, Regime, Reviser, select_regime
from himher.core.voc import AlwaysProceedVOC, VOCEstimator


@dataclass(frozen=True)
class CascadeStepLog:
    """Everything the Section 5.5 measures need to reconstruct from a
    single round: whether a mismatch was flagged, what was tried, and
    what the incumbent was before/after.
    """

    detected: bool
    statistic: float
    kappa: float
    revised: bool
    regime: Regime | None
    attempts: tuple[Attempt, ...]
    incumbent_before: CounterpartModel
    incumbent_after: CounterpartModel


class HHCascade:
    def __init__(
        self,
        incumbent: CounterpartModel,
        library: ModelLibrary,
        kappa: float,
        window: int,
        verifier: HIMVerifier,
        revise: Reviser,
        construct: Constructor,
        voc: VOCEstimator | None = None,
        calibrator: Calibrator | None = None,
        best_of: bool = False,
        min_revision_fill: int = 0,
    ) -> None:
        self.incumbent = incumbent
        self.library = library
        self.kappa = kappa
        self.window = window
        self.history = RecentHistory(window)
        self._verifier = verifier
        self._revise = revise
        self._construct = construct
        self._voc = voc or AlwaysProceedVOC()
        self._calibrator = calibrator or NoOpCalibrator()
        self._best_of = best_of
        self._min_revision_fill = min_revision_fill
        # Failed candidates are logged for inspection (Section 4.4: "rejected
        # candidates... providing mistakes for the agent to learn from") but
        # never enter the library -- lib_t holds only previously-selected
        # models, per the Reuse bullet's "set of previously used models" and
        # himher.core.him's module docstring.
        self.rejected: list[Attempt] = []

    def step(self, context: Context, counterpart_action: Action) -> CascadeStepLog:
        """One step of Algorithm 1's detect-through-verify stages, given
        the counterpart's observed action this round.
        """
        self.history.append(context, counterpart_action)
        incumbent_before = self.incumbent

        statistic = surprise_statistic(self.history, self.incumbent)
        detected = mismatch_detected(statistic, self.kappa)

        filled = len(self.history) >= self._min_revision_fill
        if not detected or not filled or not self._voc.should_revise():
            self._calibrator.update(CascadeOutcome(detected=detected, accepted_regime=None))
            return CascadeStepLog(
                detected=detected,
                statistic=statistic,
                kappa=self.kappa,
                revised=False,
                regime=None,
                attempts=(),
                incumbent_before=incumbent_before,
                incumbent_after=self.incumbent,
            )

        window = self.history.as_tuple()
        outcome = select_regime(
            self.incumbent, self.library, window, self._verifier, self._revise, self._construct,
            best_of=self._best_of,
        )

        for attempt in outcome.attempts:
            if attempt.model != outcome.accepted_model:
                self.rejected.append(attempt)

        revised = outcome.accepted_model is not None
        if revised:
            self.library.add(self.incumbent)
            self.incumbent = outcome.accepted_model

        self._calibrator.update(
            CascadeOutcome(detected=detected, accepted_regime=outcome.accepted_regime)
        )

        return CascadeStepLog(
            detected=detected,
            statistic=statistic,
            kappa=self.kappa,
            revised=revised,
            regime=outcome.accepted_regime,
            attempts=outcome.attempts,
            incumbent_before=incumbent_before,
            incumbent_after=self.incumbent,
        )
