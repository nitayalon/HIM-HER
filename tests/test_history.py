from __future__ import annotations

import pytest

from himher.core.history import RecentHistory


def test_rejects_non_positive_window():
    with pytest.raises(ValueError):
        RecentHistory(window=0)


def test_empty_history_has_length_zero_and_is_not_full():
    history = RecentHistory(window=3)
    assert len(history) == 0
    assert not history.is_full
    assert history.as_tuple() == ()


def test_appends_up_to_window_size_then_evicts_oldest():
    history = RecentHistory(window=3)
    for i in range(5):
        history.append(context=i, action=f"a{i}")

    assert len(history) == 3
    assert history.is_full
    assert history.as_tuple() == ((2, "a2"), (3, "a3"), (4, "a4"))


def test_partial_window_is_not_full():
    history = RecentHistory(window=5)
    history.append(0, "a0")
    history.append(1, "a1")
    assert len(history) == 2
    assert not history.is_full
