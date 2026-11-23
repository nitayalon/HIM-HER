"""Switch schedules (Section 5.3): which counterpart policy is in effect
at a given time index. Domain-agnostic: the index is a round in the
Switching Handoff Game and a game in the Predator-Prey domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class SwitchSchedule:
    """Maps a time index to the counterpart policy in effect.

    ``segments`` is a non-empty sequence of ``(start, policy)`` pairs,
    sorted by ``start``, with the first segment starting at 0. The policy
    in effect at index t is the policy of the last segment whose
    start <= t. A single-segment schedule gives a stationary counterpart;
    multiple segments give the switch and recurring conditions.
    """

    segments: Sequence[tuple[int, Any]]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("segments must be non-empty")
        if self.segments[0][0] != 0:
            raise ValueError("the first segment must start at round 0")
        starts = [start for start, _ in self.segments]
        if starts != sorted(starts):
            raise ValueError("segments must be sorted by start_round")

    def policy_at(self, t: int) -> Any:
        active = self.segments[0][1]
        for start, policy in self.segments:
            if start > t:
                break
            active = policy
        return active

    @property
    def switch_points(self) -> tuple[int, ...]:
        return tuple(start for start, _ in self.segments[1:])
