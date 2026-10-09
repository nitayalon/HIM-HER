from __future__ import annotations

from himher.core.library import ModelLibrary
from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Hand

M_L = BernoulliHandModel(0.1, 0.1)
M_R = BernoulliHandModel(0.9, 0.9)
M_C = BernoulliHandModel(0.1, 0.9)


def test_initializes_with_deduplicated_models():
    library = ModelLibrary([M_L, M_R, M_L])
    assert len(library) == 2
    assert M_L in library and M_R in library


def test_add_is_idempotent_for_an_existing_model():
    library = ModelLibrary([M_L])
    library.add(M_L)
    assert len(library) == 1


def test_add_inserts_a_new_model():
    library = ModelLibrary([M_L])
    library.add(M_R)
    assert len(library) == 2
    assert M_R in library


def test_best_fit_is_none_for_an_empty_library():
    library = ModelLibrary([])
    assert library.best_fit([(0, Hand.R)]) is None


def test_best_fit_selects_the_model_matching_the_window():
    library = ModelLibrary([M_L, M_R, M_C])
    window = [(0, Hand.R)] * 20 + [(1, Hand.R)] * 20  # looks like M_R
    assert library.best_fit(window) == M_R


def test_iteration_yields_all_models():
    library = ModelLibrary([M_L, M_R])
    assert set(library) == {M_L, M_R}


def test_max_learned_evicts_the_oldest_learned_entry_first():
    library = ModelLibrary([M_L], max_learned=2)
    a = BernoulliHandModel(0.2, 0.2)
    b = BernoulliHandModel(0.3, 0.3)
    c = BernoulliHandModel(0.4, 0.4)

    library.add(a)
    library.add(b)
    assert set(library) == {M_L, a, b}

    library.add(c)  # exceeds max_learned=2 -> evicts a (oldest learned)
    assert set(library) == {M_L, b, c}
    assert a not in library


def test_max_learned_never_evicts_core_models():
    library = ModelLibrary([M_L, M_R], max_learned=1)
    for i in range(5):
        library.add(BernoulliHandModel(0.01 * i, 0.01 * i))

    assert M_L in library
    assert M_R in library
    # core (2) + at most max_learned (1) learned entries
    assert len(library) <= 3


def test_unbounded_by_default():
    library = ModelLibrary([M_L])
    for i in range(50):
        # offset away from 0.1 so no generated model coincidentally
        # collides with M_L itself
        library.add(BernoulliHandModel(0.001 * i, 0.001 * i))
    assert len(library) == 51
