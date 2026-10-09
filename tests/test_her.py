from __future__ import annotations

from himher.core.her import ReplayBuffer
from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Hand
from himher.domains.handoff.policy import TabularQPolicy, Transition


def test_buffer_length_and_clear():
    buffer: ReplayBuffer[Transition] = ReplayBuffer()
    assert len(buffer) == 0

    buffer.append(Transition(0, Hand.R, Hand.R, 1.0))
    buffer.append(Transition(1, Hand.L, Hand.R, 0.0))
    assert len(buffer) == 2

    buffer.clear()
    assert len(buffer) == 0


def test_relabel_and_replay_calls_update_fn_for_every_transition_under_the_new_model():
    buffer: ReplayBuffer[Transition] = ReplayBuffer()
    transitions = [
        Transition(0, Hand.R, Hand.R, 1.0),
        Transition(1, Hand.L, Hand.L, 1.0),
        Transition(0, Hand.L, Hand.R, 0.0),
    ]
    for t in transitions:
        buffer.append(t)

    seen = []
    new_model = BernoulliHandModel(0.7, 0.3)
    buffer.relabel_and_replay(new_model, lambda t, m: seen.append((t, m)))

    assert seen == [(t, new_model) for t in transitions]


def test_relabel_and_replay_is_equivalent_to_manually_looping_update():
    # The point of relabel_and_replay is that replaying a buffer under a
    # new model produces exactly the same Q-table as manually calling
    # policy.update(transition, new_model) for each buffered transition --
    # relabeling doesn't change the transition data, only which
    # model-keyed Q-bucket it updates.
    buffer: ReplayBuffer[Transition] = ReplayBuffer()
    transitions = [
        Transition(0, Hand.R, Hand.R, 1.0),
        Transition(1, Hand.L, Hand.R, 0.0),
        Transition(0, Hand.R, Hand.L, 0.0),
    ]
    for t in transitions:
        buffer.append(t)

    new_model = BernoulliHandModel(0.6, 0.4)

    policy_via_replay = TabularQPolicy()
    buffer.relabel_and_replay(new_model, policy_via_replay.update)

    policy_via_manual_loop = TabularQPolicy()
    for t in transitions:
        policy_via_manual_loop.update(t, new_model)

    for context in (0, 1):
        for action in (Hand.L, Hand.R):
            assert policy_via_replay.q_value(
                context, new_model, action
            ) == policy_via_manual_loop.q_value(context, new_model, action)
