from __future__ import annotations

import numpy as np
import pytest

from himher.core.him import HIMVerifier
from himher.core.library import ModelLibrary
from himher.core.regime import Regime, select_regime
from himher.domains.predator_prey.construct import construct_weight_model
from himher.domains.predator_prey.counterpart import (
    CAUTIOUS_B_MODEL,
    DEN_A_MODEL,
    DEN_B_MODEL,
    EVADE_MODEL,
    FEATURES,
    WeightedPreyModel,
    log_likelihoods,
)
from himher.domains.predator_prey.env import (
    DEN_DIST,
    T,
    cell,
    resolve_round,
    sample_start,
)
from himher.domains.predator_prey.experiment import METHODS, conditions, run_episode
from himher.domains.predator_prey.planner import plan
from himher.domains.predator_prey.revise import revise_weight_mixture

N, S, E, W, STAY = range(5)


def _sample_window(model, n, seed=0):
    """(context, action) pairs from random non-terminal contexts."""
    rng = np.random.default_rng(seed)
    window = []
    while len(window) < n:
        p, q = sample_start(rng)
        window.append(((p, q), model.sample((p, q), rng)))
    return window


# --- env -----------------------------------------------------------------
def test_same_cell_is_a_catch():
    result = resolve_round((cell(3, 1), cell(3, 3)), E, W, t=0)
    assert result.done and result.outcome == "catch"


def test_swapping_cells_is_a_catch():
    result = resolve_round((cell(3, 2), cell(3, 3)), E, W, t=0)
    assert result.outcome == "catch"


def test_entering_a_den_ends_the_game_with_no_reward():
    result = resolve_round((cell(5, 5), cell(0, 1)), N, W, t=0)
    assert result.outcome == "den"
    assert result.reward == pytest.approx(-0.01)


def test_timeout_reward_is_inverse_distance():
    result = resolve_round((cell(3, 0), cell(3, 3)), STAY, N, t=T - 1)
    assert result.outcome == "timeout"
    assert result.reward == pytest.approx(1 / (1 + 4) - 0.01)


def test_predator_counts_as_a_wall_for_den_distance():
    # Prey at (0,2), predator at (0,1): the direct route to den A is blocked.
    assert DEN_DIST[0, cell(0, 1), cell(0, 2)] == 4


# --- prey model -------------------------------------------------------------
def test_features_match_the_hand_worked_example():
    # Prey (3,3), predator (1,3); moves N, S, E, W.
    expected = [[1, -1, -1], [-1, 1, 1], [-1, 1, 1], [1, -1, 1]]
    np.testing.assert_array_equal(FEATURES[cell(1, 3), cell(3, 3)], expected)


def test_den_a_picks_a_good_move_88_percent_of_the_time():
    proba = DEN_A_MODEL.predict_proba((cell(1, 3), cell(3, 3)))
    assert proba[N] + proba[W] == pytest.approx(np.e**2 / (np.e**2 + 1))


def test_models_are_equal_and_hashable_by_value():
    assert WeightedPreyModel(0.0, 0.5, 0.25) == CAUTIOUS_B_MODEL
    assert len({WeightedPreyModel(1.0, 0.0, 0.0), DEN_A_MODEL}) == 1


def test_vectorised_likelihood_matches_log_prob():
    window = _sample_window(EVADE_MODEL, 30)
    expected = sum(EVADE_MODEL.log_prob(c, a) for c, a in window)
    assert log_likelihoods(np.array([EVADE_MODEL.w]), window)[0] == pytest.approx(expected)


# --- merge and construct ---------------------------------------------------
def test_construct_recovers_evade_from_plenty_of_data():
    assert construct_weight_model(_sample_window(EVADE_MODEL, 2000)) == EVADE_MODEL


def test_merge_cannot_express_evade_from_the_den_models():
    merged = revise_weight_mixture([DEN_A_MODEL, DEN_B_MODEL], _sample_window(EVADE_MODEL, 500))
    assert merged.w_pred == 0.0


def test_merge_finds_cautious_b_between_den_b_and_evade():
    pool = [DEN_A_MODEL, DEN_B_MODEL, EVADE_MODEL]
    assert revise_weight_mixture(pool, _sample_window(CAUTIOUS_B_MODEL, 3000)) == CAUTIOUS_B_MODEL


def test_merge_pool_includes_the_incumbent_only_when_enabled():
    # Strong-evader data: neither den model fits, so Reuse fails and Merge runs.
    window = _sample_window(WeightedPreyModel(0.0, 0.0, 3.0), 3000)
    library = ModelLibrary([DEN_A_MODEL, DEN_B_MODEL])
    verifier = HIMVerifier(tau=1.0, delta=0.1, contexts=None)
    seen = []

    def spy(pool, w):
        seen.append(tuple(pool))
        return revise_weight_mixture(pool, w)

    select_regime(EVADE_MODEL, library, window, verifier, spy, construct_weight_model)
    select_regime(
        EVADE_MODEL, library, window, verifier, spy, construct_weight_model,
        merge_includes_incumbent=True,
    )
    assert EVADE_MODEL not in seen[0]
    assert EVADE_MODEL in seen[1]


def test_him_with_window_contexts_accepts_a_clearly_better_model():
    # A one-model library leaves Merge nothing to blend, so Construct is tried.
    # (With {Den A, Den B}, Merge's near-random-walker blend can be accepted
    # first, because it fits Evade data far better than Den A does.)
    window = _sample_window(EVADE_MODEL, 200)
    verifier = HIMVerifier(tau=1.0, delta=0.1, contexts=None)
    outcome = select_regime(
        DEN_A_MODEL, ModelLibrary([DEN_A_MODEL]), window, verifier,
        revise_weight_mixture, construct_weight_model,
    )
    assert outcome.accepted_regime == Regime.CONSTRUCT
    assert outcome.accepted_model.w_pred > 0


# --- planner and runner -----------------------------------------------------
def test_learned_predator_replays_the_window_only_with_her():
    from himher.domains.predator_prey.learning import make_learned_hh_predator

    window = _sample_window(EVADE_MODEL, 60)
    weights = {}
    for replay in (True, False):
        predator = make_learned_hh_predator(np.random.default_rng(0), kappa=1.3, replay=replay)
        for t, (context, prey_action) in enumerate(window):
            action = predator.act(context, 0)
            predator.observe(context, prey_action)
            predator.learn(0, context, action, -0.01, context, False)
        new_model = predator.cascade.incumbent
        assert new_model != DEN_A_MODEL  # the cascade revised away from Den A
        weights[replay] = predator.q.weights[new_model]
        assert (predator.replay_updates > 0) == replay
    assert not np.allclose(weights[True], weights[False])


def test_library_models_start_with_their_own_responses():
    from himher.domains.predator_prey.learning import make_learned_hh_predator

    predator = make_learned_hh_predator(np.random.default_rng(0), kappa=1.3, replay=True)
    np.testing.assert_array_equal(predator.q.weights[DEN_A_MODEL], [0, 1, 0, 0])
    np.testing.assert_array_equal(predator.q.weights[DEN_B_MODEL], [0, 0, 1, 0])


def test_oracle_steps_onto_an_adjacent_prey_that_is_cornered():
    # Prey at (0,6) corner, predator at (1,6): moving N or staying both work,
    # but the planner must value the immediate catch highly.
    q = plan(EVADE_MODEL)[0, cell(1, 6), cell(0, 6)]
    assert q.max() > 0.5


def test_episode_runs_for_every_condition_and_method():
    for condition in conditions():
        for method in METHODS:
            record = run_episode(method, condition, seed=0, kappa=1.5, n_games=40)
            assert 0.0 <= record["catch_rate"] <= 1.0
            if method == "oracle":
                assert record["oracle_agreement"] == 1.0
                assert record["l2_mean"] == 0.0
