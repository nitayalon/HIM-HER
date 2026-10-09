from __future__ import annotations

import pytest

from himher.core.diagnostics import likelihood_table
from himher.domains.handoff.counterpart import BernoulliHandModel
from himher.domains.handoff.env import Hand


def test_likelihood_table_is_empty_dict_values_for_an_empty_window():
    models = {"m_l": BernoulliHandModel(0.1, 0.1)}
    assert likelihood_table(models, []) == {"m_l": 0.0}


def test_likelihood_table_ranks_the_true_generating_model_highest():
    m_l = BernoulliHandModel(0.1, 0.1)
    m_r = BernoulliHandModel(0.9, 0.9)
    window = [(0, Hand.L)] * 18 + [(0, Hand.R)] * 2  # looks like m_l

    table = likelihood_table({"m_l": m_l, "m_r": m_r}, window)

    assert table["m_l"] > table["m_r"]


def test_likelihood_table_is_per_step_not_summed():
    m_l = BernoulliHandModel(0.1, 0.1)
    short_window = [(0, Hand.L)] * 5
    long_window = short_window * 4

    table_short = likelihood_table({"m_l": m_l}, short_window)
    table_long = likelihood_table({"m_l": m_l}, long_window)

    assert table_short["m_l"] == pytest.approx(table_long["m_l"])
