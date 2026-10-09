"""Value-of-computation stage (Section 4.2): decides whether a detected
mismatch justifies the cost of revision. The paper leaves VOC estimation
open -- both expectations in Eq. "VOC" are not directly observable before
revision -- and Section 5.2's illustration skips this stage entirely,
going straight from detection to regime selection. ``AlwaysProceedVOC``
preserves the stage's place in the cascade (so the general orchestration
in cascade.py always calls it, matching Algorithm 1) while matching what
the illustration actually does.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class VOCEstimator(Protocol):
    def should_revise(self) -> bool: ...


class AlwaysProceedVOC:
    def should_revise(self) -> bool:
        return True
