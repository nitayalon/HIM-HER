"""Aggregation helpers for turning per-seed experiment records (one row
per condition/method/seed) into the summary tables and LaTeX output used
for Section 6 (Results).
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def summarize(
    records: pd.DataFrame, group_by: Sequence[str], value_columns: Sequence[str]
) -> pd.DataFrame:
    """Mean and standard error of the mean for each value column, grouped
    by ``group_by``. Columns that are entirely NaN within a group (e.g.
    model_accuracy for a method that never constructs) summarize to NaN
    rather than raising.
    """
    grouped = records.groupby(list(group_by), sort=False)[list(value_columns)]
    mean = grouped.mean()
    sem = grouped.sem()
    summary = mean.join(sem, rsuffix="_sem")
    return summary.reset_index()


def format_cell(mean: float, sem: float, float_format: str = "%.3f") -> str:
    """"mean (sem)", or just "--" if the mean itself is missing (e.g. a
    measure that does not apply to this method, such as model_accuracy
    for a baseline that never constructs anything)."""
    if pd.isna(mean):
        return "--"
    if pd.isna(sem):
        return float_format % mean
    return f"{float_format % mean} ({float_format % sem})"


def to_latex_table(
    summary: pd.DataFrame,
    row_key: str,
    column_key: str,
    value_column: str,
    caption: str,
    label: str,
    float_format: str = "%.3f",
) -> str:
    """A booktabs-style table with one row per distinct ``row_key`` value
    and one column per distinct ``column_key`` value, showing
    "mean (sem)" for ``value_column`` in each cell.
    """
    columns = list(dict.fromkeys(summary[column_key]))  # stable de-dup, preserves first-seen order
    rows = list(dict.fromkeys(summary[row_key]))

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        "  \\begin{tabular}{l" + "c" * len(columns) + "}",
        "    \\toprule",
        "    " + " & ".join(["Condition", *columns]) + r" \\",
        "    \\midrule",
    ]
    for row in rows:
        cells = [str(row)]
        for column in columns:
            match = summary[(summary[row_key] == row) & (summary[column_key] == column)]
            if match.empty:
                cells.append("--")
            else:
                mean = match[value_column].iloc[0]
                sem = match[f"{value_column}_sem"].iloc[0]
                cells.append(format_cell(mean, sem, float_format))
        lines.append("    " + " & ".join(cells) + r" \\")
    lines += ["    \\bottomrule", "  \\end{tabular}", r"\end{table}"]
    return "\n".join(lines)
