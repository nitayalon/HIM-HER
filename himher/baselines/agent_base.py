"""Common interface shared by the H+H agent and the Bayesian baselines
(Section 5.4), so the experiment runner can treat them uniformly without
caring how each one decides its action or updates its beliefs.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from himher.domains.handoff.env import Context, Hand


@runtime_checkable
class Agent(Protocol):
    def act(self, context: Context) -> Hand: ...

    def update(
        self, context: Context, ego_action: Hand, counterpart_action: Hand, reward: float
    ) -> None: ...
