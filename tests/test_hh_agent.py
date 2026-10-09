"""Tests for HHAgent: wiring HHCascade and TabularQPolicy together via a
replay buffer bounded to the cascade's verification window (Remark 4.2).

Scenarios here deliberately pre-seed the cascade's history and the
agent's buffer directly rather than driving everything through act(),
because act()-driven rounds make the ego's own (initially arbitrary)
action choices part of the buffer's content, and whether replay helps at
all depends on that content being informative (containing some attempts
at the action that turns out to be correct). Pre-seeding an explicit,
known-informative window isolates the mechanism being tested -- whether
relabel-and-replay actually happens, and whether ablation actually
discards -- from that separate, genuinely empirical question of how much
replay helps on average over a real, exploration-driven episode (which
belongs to Section 5.5's measures, not a unit test).
"""

from __future__ import annotations

import numpy as np

from himher.baselines.hh_agent import HHAgent
from himher.core.cascade import HHCascade
from himher.core.him import HIMVerifier
from himher.core.library import ModelLibrary
from himher.core.regime import Regime
from himher.domains.handoff.construct import construct_bernoulli_model
from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Hand
from himher.domains.handoff.policy import TabularQPolicy, Transition
from himher.domains.handoff.revise import revise_mixture_model

CONTEXTS = (0, 1)
M_L = BernoulliHandModel(0.1, 0.1)
M_R = BernoulliHandModel(0.9, 0.9)
M_C = BernoulliHandModel(0.1, 0.9)


def _make_agent(replay: bool, epsilon: float = 0.0) -> HHAgent:
    cascade = HHCascade(
        incumbent=M_L,
        library=ModelLibrary([M_L, M_R, M_C]),
        kappa=0.3,
        window=10,
        verifier=HIMVerifier(tau=0.0, delta=0.1, contexts=CONTEXTS),
        revise=revise_mixture_model,
        construct=lambda w: construct_bernoulli_model(w, 1.0),
    )
    policy = TabularQPolicy(learning_rate=0.5, epsilon=epsilon)
    return HHAgent(cascade, policy, np.random.default_rng(0), replay=replay)


def test_buffer_never_exceeds_the_cascades_window_length():
    agent = _make_agent(replay=True)
    for _ in range(50):
        agent.update(0, Hand.L, Hand.L, 1.0)
    assert len(agent._buffer) <= agent.cascade.window


def _seed_nine_rounds_of_mixed_informative_experience(agent: HHAgent) -> None:
    # 9 rounds against the real m_R counterpart, with ego alternating
    # between the wrong guess (L) and the right one (R), seeded directly
    # into both the cascade's detection window and the replay buffer so
    # the 10th (triggering) round sees a complete, informative window.
    pattern = [(Hand.L, Hand.R), (Hand.R, Hand.R)] * 4 + [(Hand.L, Hand.R)]
    for ego, counterpart in pattern:
        reward = 1.0 if ego == counterpart else 0.0
        agent.cascade.history.append(0, counterpart)
        agent._buffer.append(Transition(0, ego, counterpart, reward))


def test_replay_gives_immediate_informed_q_values_the_moment_a_model_is_accepted():
    agent = _make_agent(replay=True)
    _seed_nine_rounds_of_mixed_informative_experience(agent)

    log_before_models = agent.cascade.incumbent
    agent.update(0, ego_action=Hand.L, counterpart_action=Hand.R, reward=0.0)

    assert agent.last_step_log.revised
    assert agent.cascade.incumbent != log_before_models
    assert agent.cascade.incumbent == M_R
    # The relabeled window contained several correct (R, R) attempts, so
    # Q(R) under the newly accepted model should already reflect that --
    # without a single *online* update having happened under m_R yet.
    assert agent._policy.q_value(0, M_R, Hand.R) > 0.5


def test_ablation_discards_the_same_experience_instead_of_replaying_it():
    agent = _make_agent(replay=False)
    _seed_nine_rounds_of_mixed_informative_experience(agent)

    agent.update(0, ego_action=Hand.L, counterpart_action=Hand.R, reward=0.0)

    assert agent.last_step_log.revised
    assert agent.cascade.incumbent == M_R
    # No replay happened, and the triggering round's own online update is
    # attributed to the pre-revision incumbent (Section 4.5's ordering),
    # so Q(m_R, R) should still be completely uninformed.
    assert agent._policy.q_value(0, M_R, Hand.R) == 0.0


def test_replay_and_ablation_agree_when_no_revision_has_happened():
    # With nothing yet to relabel, the two modes must behave identically.
    replay_agent = _make_agent(replay=True)
    ablation_agent = _make_agent(replay=False)
    for _ in range(15):
        replay_agent.update(0, Hand.L, Hand.L, 1.0)
        ablation_agent.update(0, Hand.L, Hand.L, 1.0)

    assert replay_agent.cascade.incumbent == ablation_agent.cascade.incumbent == M_L
    assert replay_agent._policy.q_value(
        0, M_L, Hand.L
    ) == ablation_agent._policy.q_value(0, M_L, Hand.L)
