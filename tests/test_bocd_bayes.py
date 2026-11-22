from __future__ import annotations

import numpy as np
import pytest

from himher.baselines.bocd_bayes import BOCDBayesAgent
from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Hand

M_L = BernoulliHandModel(0.1, 0.1)
M_R = BernoulliHandModel(0.9, 0.9)
M_C = BernoulliHandModel(0.1, 0.9)
LIBRARY = (M_L, M_R, M_C)


def _run_episode(agent: BOCDBayesAgent, true_model_at, episode_length: int, seed: int):
    rng = np.random.default_rng(seed)
    posteriors, run_length_posteriors = [], []
    for t in range(episode_length):
        context = int(rng.random() < 0.5)
        ego_action = agent.act(context)
        true_model = true_model_at(t)
        counterpart_action = true_model.sample(context, rng)
        reward = 1.0 if ego_action == counterpart_action else 0.0
        agent.update(context, ego_action, counterpart_action, reward)
        posteriors.append(agent.posterior)
        run_length_posteriors.append(agent.run_length_posterior)
    return posteriors, run_length_posteriors


def test_rejects_empty_hypotheses() -> None:
    with pytest.raises(ValueError):
        BOCDBayesAgent((), hazard=0.01)


def test_rejects_hazard_outside_open_unit_interval() -> None:
    with pytest.raises(ValueError):
        BOCDBayesAgent(LIBRARY, hazard=0.0)
    with pytest.raises(ValueError):
        BOCDBayesAgent(LIBRARY, hazard=1.0)


def test_posterior_and_run_length_posterior_are_valid_distributions() -> None:
    agent = BOCDBayesAgent(LIBRARY, hazard=1 / 150)
    posteriors, run_length_posteriors = _run_episode(agent, lambda t: M_L, 300, seed=0)
    for post in posteriors:
        assert np.all(np.isfinite(post))
        assert post.sum() == pytest.approx(1.0, abs=1e-8)
        assert np.all(post >= 0.0)
    for rlp in run_length_posteriors:
        assert np.all(np.isfinite(rlp))
        assert rlp.sum() == pytest.approx(1.0, abs=1e-8)
        assert np.all(rlp >= 0.0)


def test_stationary_counterpart_concentrates_on_the_true_type() -> None:
    agent = BOCDBayesAgent(LIBRARY, hazard=1 / 150)
    posteriors, _ = _run_episode(agent, lambda t: M_R, 300, seed=1)
    final_posterior = posteriors[-1]
    assert np.argmax(final_posterior) == LIBRARY.index(M_R)
    assert final_posterior[LIBRARY.index(M_R)] > 0.95


def test_stationary_run_length_map_grows_without_resetting() -> None:
    # On a never-changing counterpart, the MAP run length should track
    # elapsed rounds closely -- no spurious inferred changepoints.
    agent = BOCDBayesAgent(LIBRARY, hazard=1 / 150)
    _, run_length_posteriors = _run_episode(agent, lambda t: M_L, 300, seed=2)
    map_run_length_at_end = int(np.argmax(run_length_posteriors[-1]))
    assert map_run_length_at_end > 250  # close to the full 300-round episode


def test_switch_is_detected_and_run_length_tracks_elapsed_time_since_it() -> None:
    switch_round = 150
    episode_length = 300

    def true_model_at(t: int) -> BernoulliHandModel:
        return M_L if t < switch_round else M_R

    agent = BOCDBayesAgent(LIBRARY, hazard=1 / 150)
    posteriors, run_length_posteriors = _run_episode(agent, true_model_at, episode_length, seed=3)

    # Well after the switch, the posterior should have fully flipped to
    # the new true type.
    late_posterior = posteriors[-1]
    assert np.argmax(late_posterior) == LIBRARY.index(M_R)
    assert late_posterior[LIBRARY.index(M_R)] > 0.95

    # And the MAP run length should closely track true elapsed time since
    # the switch (not since the start of the episode).
    map_run_length_at_end = int(np.argmax(run_length_posteriors[-1]))
    true_elapsed_since_switch = episode_length - 1 - switch_round
    assert abs(map_run_length_at_end - true_elapsed_since_switch) <= 5


def test_act_prefers_the_action_the_posterior_favors() -> None:
    # With an overwhelmingly M_R-favoring posterior (forced via lots of
    # R-biased stationary data), act() should play R in both contexts.
    agent = BOCDBayesAgent(LIBRARY, hazard=1 / 150)
    _run_episode(agent, lambda t: M_R, 100, seed=4)
    assert agent.act(context=0) == Hand.R
    assert agent.act(context=1) == Hand.R
