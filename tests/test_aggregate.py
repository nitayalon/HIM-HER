from __future__ import annotations

import math

import pandas as pd
import pytest

from himher.experiment.aggregate import format_cell, summarize, to_latex_table


def _records() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"condition": "known_switch", "method": "hh", "seed": 0, "return": 10.0},
            {"condition": "known_switch", "method": "hh", "seed": 1, "return": 12.0},
            {"condition": "known_switch", "method": "bayes", "seed": 0, "return": 8.0},
            {"condition": "known_switch", "method": "bayes", "seed": 1, "return": 8.0},
        ]
    )


def test_summarize_computes_mean_and_sem_per_group():
    summary = summarize(_records(), group_by=["condition", "method"], value_columns=["return"])

    hh_row = summary[summary["method"] == "hh"].iloc[0]
    assert hh_row["return"] == pytest.approx(11.0)
    assert hh_row["return_sem"] == pytest.approx(1.0)  # sem of [10, 12] with ddof=1

    bayes_row = summary[summary["method"] == "bayes"].iloc[0]
    assert bayes_row["return"] == pytest.approx(8.0)
    assert bayes_row["return_sem"] == pytest.approx(0.0)


def test_summarize_handles_all_nan_group_without_raising():
    records = pd.DataFrame(
        [
            {"condition": "c", "method": "m", "seed": 0, "accuracy": math.nan},
            {"condition": "c", "method": "m", "seed": 1, "accuracy": math.nan},
        ]
    )
    summary = summarize(records, group_by=["condition", "method"], value_columns=["accuracy"])
    assert pd.isna(summary["accuracy"].iloc[0])


def test_format_cell_renders_mean_and_sem():
    assert format_cell(1.23456, 0.01111) == "1.235 (0.011)"


def test_format_cell_renders_missing_mean_as_dashes():
    assert format_cell(math.nan, 0.01) == "--"


def test_format_cell_renders_missing_sem_as_bare_mean():
    assert format_cell(1.5, math.nan) == "1.500"


def test_to_latex_table_contains_booktabs_structure_and_values():
    summary = summarize(_records(), group_by=["condition", "method"], value_columns=["return"])
    latex = to_latex_table(
        summary,
        row_key="condition",
        column_key="method",
        value_column="return",
        caption="Cumulative return",
        label="tab:return",
    )

    assert r"\toprule" in latex
    assert r"\bottomrule" in latex
    assert r"\caption{Cumulative return}" in latex
    assert r"\label{tab:return}" in latex
    assert "11.000 (1.000)" in latex  # hh column
    assert "8.000 (0.000)" in latex  # bayes column


def test_to_latex_table_handles_a_missing_row_column_combination():
    records = pd.DataFrame(
        [
            {"condition": "a", "method": "x", "seed": 0, "v": 1.0},
            {"condition": "b", "method": "y", "seed": 0, "v": 2.0},
        ]
    )
    summary = summarize(records, group_by=["condition", "method"], value_columns=["v"])
    latex = to_latex_table(summary, "condition", "method", "v", "cap", "lab")
    assert "--" in latex
