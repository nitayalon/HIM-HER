from __future__ import annotations

import numpy as np
import pytest

from himher.baselines.continuous_bayes import ContinuousBayesAgent
from himher.baselines.fixed_library_bayes import FixedLibraryBayesAgent
from himher.baselines.oracle_bayes import make_oracle_bayes_agent
from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Hand

M_L = BernoulliHandModel(0.1, 0.1)
M_R = BernoulliHandModel(0.9, 0.9)
M_C = BernoulliHandModel(0.1, 0.9)


def test_fixed_library_bayes_starts_with_a_uniform_posterior():
    agent = FixedLibraryBayesAgent([M_L, M_R, M_C])
    assert agent.posterior == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_fixed_library_bayes_rejects_an_empty_hypothesis_space():
    with pytest.raises(ValueError):
        FixedLibraryBayesAgent([])


def test_fixed_library_bayes_concentrates_on_the_matching_hypothesis():
    agent = FixedLibraryBayesAgent([M_L, M_R, M_C])
    for _ in range(50):
        agent.update(context=0, ego_action=Hand.R, counterpart_action=Hand.R, reward=1.0)
        agent.update(context=1, ego_action=Hand.R, counterpart_action=Hand.R, reward=1.0)

    posterior = agent.posterior
    m_r_index = [M_L, M_R, M_C].index(M_R)
    assert posterior[m_r_index] == pytest.approx(1.0, abs=1e-6)


def test_fixed_library_bayes_acts_via_posterior_predictive_best_response():
    agent = FixedLibraryBayesAgent([M_R])  # certain of a single hypothesis
    assert agent.act(context=0) == Hand.R
    assert agent.act(context=1) == Hand.R


def test_oracle_bayes_includes_the_true_model_when_absent_from_the_library():
    agent = make_oracle_bayes_agent([M_L, M_R, M_C], true_model=BernoulliHandModel(0.2, 0.8))
    assert len(agent.posterior) == 4


def test_oracle_bayes_does_not_duplicate_the_true_model_when_already_present():
    agent = make_oracle_bayes_agent([M_L, M_R, M_C], true_model=M_R)
    assert len(agent.posterior) == 3


def test_oracle_bayes_converges_to_a_novel_true_model_the_fixed_library_never_could():
    true_model = BernoulliHandModel(0.2, 0.8)
    oracle = make_oracle_bayes_agent([M_L, M_R, M_C], true_model=true_model)
    rng = np.random.default_rng(0)

    for _ in range(500):
        for context in (0, 1):
            counterpart_action = true_model.sample(context, rng)
            oracle.update(context, Hand.L, counterpart_action, 0.0)

    # Oracle should correctly predict the true model's action at both
    # contexts (0.2 -> mostly L, 0.8 -> mostly R), which a 3-model fixed
    # library containing none of these cannot do simultaneously.
    assert oracle.act(0) == Hand.L
    assert oracle.act(1) == Hand.R


def test_continuous_bayes_defaults_to_the_prior_mean_with_no_data():
    agent = ContinuousBayesAgent(alpha=1.0)
    # prior mean 0.5 is not > 0.5, so the default action is L
    assert agent.act(0) == Hand.L
    assert agent.act(1) == Hand.L


def test_continuous_bayes_tracks_a_switch_without_any_detection_step():
    # This baseline has no window or decay -- it is a running posterior
    # over the *entire* history -- so it only tips to R once post-switch
    # evidence outweighs the accumulated pre-switch evidence, which is
    # itself the point of contrasting it with H+H's windowed detection.
    agent = ContinuousBayesAgent(alpha=1.0)
    for _ in range(30):
        agent.update(0, Hand.L, Hand.L, 1.0)
    assert agent.act(0) == Hand.L

    for _ in range(300):
        agent.update(0, Hand.L, Hand.R, 0.0)
    assert agent.act(0) == Hand.R
