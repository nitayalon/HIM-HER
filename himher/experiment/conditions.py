"""The four experimental conditions from Section 5.3, built as
SwitchSchedules over a fixed episode length (the locked-in default is
T=300, with a switch at the episode midpoint for Within/Outside library
switch).
"""

from __future__ import annotations

from dataclasses import dataclass

from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import SwitchSchedule

EPISODE_LENGTH = 300

M_L = BernoulliHandModel(0.1, 0.1)
M_R = BernoulliHandModel(0.9, 0.9)
M_C = BernoulliHandModel(0.1, 0.9)
LIBRARY_0 = (M_L, M_R, M_C)

M_NEW_1 = BernoulliHandModel(0.1, 0.5)  # merge of (m_L, m_C) (Section 5.3)
M_NEW_2 = BernoulliHandModel(0.2, 0.8)  # completely new, off every segment (Section 5.3)


@dataclass(frozen=True)
class Condition:
    name: str
    schedule: SwitchSchedule
    episode_length: int
    true_models_in_order: tuple[BernoulliHandModel, ...]


def stationary(
    model: BernoulliHandModel = M_L, episode_length: int = EPISODE_LENGTH
) -> Condition:
    return Condition("stationary", SwitchSchedule([(0, model)]), episode_length, (model,))


def within_library_switch(episode_length: int = EPISODE_LENGTH) -> Condition:
    switch_at = episode_length // 2
    schedule = SwitchSchedule([(0, M_L), (switch_at, M_R)])
    return Condition("within_library_switch", schedule, episode_length, (M_L, M_R))


def outside_library_switch(
    m_new: BernoulliHandModel = M_NEW_1, episode_length: int = EPISODE_LENGTH
) -> Condition:
    switch_at = episode_length // 2
    schedule = SwitchSchedule([(0, M_L), (switch_at, m_new)])
    return Condition("outside_library_switch", schedule, episode_length, (M_L, m_new))


def recurring(
    m_new: BernoulliHandModel = M_NEW_2, episode_length: int = EPISODE_LENGTH
) -> Condition:
    # m_new appears, disappears, and reappears, testing whether the
    # second appearance is reused from the library instead of
    # reconstructed (Section 5.3). Defaults to M_NEW_2, the model that
    # genuinely requires construction on its first appearance.
    quarter = episode_length // 4
    schedule = SwitchSchedule(
        [(0, M_L), (quarter, m_new), (2 * quarter, M_R), (3 * quarter, m_new)]
    )
    return Condition("recurring", schedule, episode_length, (M_L, m_new, M_R, m_new))
