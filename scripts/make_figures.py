"""Generates the Results-section figures from results/main_summary.csv
and results/kappa_sweep_summary.csv, styled with the validated
categorical palette (dataviz skill, references/palette.md): a fixed hue
order assigned once per method and reused identically across every
figure, so the same method always reads as the same color.

Usage: python scripts/make_figures.py [--results results] [--out "AAMAS 2027"]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Categorical palette slots 1-5 (fixed order; see dataviz skill palette.md)
METHOD_COLORS = {
    "hh_replay": "#2a78d6",  # slot 1: blue
    "hh_ablation": "#eb6834",  # slot 2: orange
    "fixed_bayes": "#1baf7a",  # slot 3: aqua
    "oracle_bayes": "#eda100",  # slot 4: yellow
    "continuous_bayes": "#e87ba4",  # slot 5: magenta
}
METHOD_LABELS = {
    "hh_replay": "H+H (replay)",
    "hh_ablation": "H+H (ablation)",
    "fixed_bayes": "Fixed-library Bayes",
    "oracle_bayes": "Oracle Bayes",
    "continuous_bayes": "Continuous Bayes",
}
METHOD_ORDER = list(METHOD_COLORS)

CONDITION_LABELS = {
    "stationary": "Stationary",
    "within_library_switch": "Within library\nswitch",
    "outside_library_switch_m1": "Outside library\nswitch ($m_1$)",
    "outside_library_switch_m2": "Outside library\nswitch ($m_2$)",
    "recurring": "Recurring",
}

INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
DIVERGING_POSITIVE = "#2a78d6"  # blue pole
DIVERGING_NEGATIVE = "#e34948"  # red pole

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 13,
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "axes.labelsize": 14,
        "xtick.color": SECONDARY_INK,
        "ytick.color": SECONDARY_INK,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "legend.fontsize": 12,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def _clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis="y", zorder=0)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)


BASELINE_METHODS = {"fixed_bayes", "oracle_bayes", "continuous_bayes"}


def _grouped_bar(
    ax, summary: pd.DataFrame, conditions: list[str], value_col: str, sem_col: str, ylabel: str, hatch_baselines: bool = False
) -> None:
    methods = [m for m in METHOD_ORDER if m in summary["method"].unique()]
    n_groups, n_methods = len(conditions), len(methods)
    width = 0.8 / n_methods
    x = np.arange(n_groups)

    for i, method in enumerate(methods):
        values, errs = [], []
        for condition in conditions:
            row = summary[(summary.condition == condition) & (summary.method == method)]
            values.append(row[value_col].iloc[0] if not row.empty else np.nan)
            errs.append(row[sem_col].iloc[0] if not row.empty else 0.0)
        offset = (i - (n_methods - 1) / 2) * width
        is_baseline = hatch_baselines and method in BASELINE_METHODS
        ax.bar(
            x + offset,
            values,
            width=width * 0.92,
            yerr=errs,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            capsize=2,
            error_kw={"linewidth": 0.8, "ecolor": SECONDARY_INK},
            hatch="////" if is_baseline else None,
            edgecolor="white" if is_baseline else "none",
            linewidth=0.6 if is_baseline else 0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=13)
    ax.set_ylabel(ylabel, fontsize=14)
    _clean_axes(ax)


def plot_cumulative_return(summary: pd.DataFrame, out_dir: Path) -> None:
    conditions = ["stationary", "within_library_switch", "outside_library_switch_m1", "outside_library_switch_m2", "recurring"]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    _grouped_bar(ax, summary, conditions, "cumulative_return", "cumulative_return_sem", "Cumulative return (/300)")
    ax.legend(ncol=3, fontsize=12, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_return.png", dpi=200)
    plt.close(fig)


def plot_return_before_after(phase_summary: pd.DataFrame, out_dir: Path) -> None:
    r"""Per-round average return before vs.\ after the first switch in
    each condition (Stationary has no switch, so it appears only in the
    "before" panel, as the no-change reference). H+H variants are drawn
    with a solid fill; Bayesian baselines are hatched, so the two
    families of methods are visually distinguishable at a glance (same
    palette as every other figure).
    """
    before_conditions = ["stationary", "within_library_switch", "outside_library_switch_m1", "outside_library_switch_m2", "recurring"]
    after_conditions = ["within_library_switch", "outside_library_switch_m1", "outside_library_switch_m2", "recurring"]

    fig, axes = plt.subplots(
        1, 2, figsize=(19, 6.4), sharey=True, gridspec_kw={"width_ratios": [len(before_conditions), len(after_conditions)]}
    )
    _grouped_bar(axes[0], phase_summary, before_conditions, "return_before", "return_before_sem", "Mean per-round return", hatch_baselines=True)
    axes[0].set_title("Before change", fontsize=15)
    _grouped_bar(axes[1], phase_summary, after_conditions, "return_after", "return_after_sem", "", hatch_baselines=True)
    axes[1].set_title("After change", fontsize=15)

    y_max = max(
        phase_summary["return_before"].max() + phase_summary["return_before_sem"].max(),
        phase_summary["return_after"].max() + phase_summary["return_after_sem"].max(),
    )
    for ax in axes:
        ax.set_ylim(0, y_max * 1.22)
        ax.tick_params(axis="y", labelsize=13)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=13, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(out_dir / "return_before_after.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_adaptation_delay(summary: pd.DataFrame, out_dir: Path) -> None:
    conditions = ["within_library_switch", "outside_library_switch_m1", "outside_library_switch_m2", "recurring"]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    _grouped_bar(ax, summary, conditions, "adaptation_delay", "adaptation_delay_sem", "Rounds to 90% of steady state")
    ax.legend(ncol=3, fontsize=12, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    fig.savefig(out_dir / "adaptation_delay.png", dpi=200)
    plt.close(fig)


def plot_post_switch_regret(summary: pd.DataFrame, out_dir: Path) -> None:
    conditions = ["within_library_switch", "outside_library_switch_m1", "outside_library_switch_m2", "recurring"]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    _grouped_bar(ax, summary, conditions, "post_switch_regret", "post_switch_regret_sem", "Post-switch regret (cumulative)", hatch_baselines=True)
    ax.legend(ncol=3, fontsize=12, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    fig.savefig(out_dir / "post_switch_regret.png", dpi=200)
    plt.close(fig)


def plot_replay_benefit(summary: pd.DataFrame, out_dir: Path) -> None:
    conditions = ["within_library_switch", "outside_library_switch_m1", "outside_library_switch_m2", "recurring"]
    rows = summary[(summary.method == "hh_replay")]
    values, errs = [], []
    for condition in conditions:
        row = rows[rows.condition == condition]
        values.append(row["replay_benefit"].iloc[0] if not row.empty else np.nan)
        errs.append(row["replay_benefit_sem"].iloc[0] if not row.empty else 0.0)

    fig, ax = plt.subplots(figsize=(7, 5.4))
    colors = [DIVERGING_POSITIVE if v >= 0 else DIVERGING_NEGATIVE for v in values]
    x = np.arange(len(conditions))
    ax.bar(x, values, yerr=errs, color=colors, width=0.6, capsize=4, error_kw={"linewidth": 1.1, "ecolor": SECONDARY_INK})
    ax.axhline(0, color=AXIS, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=13)
    ax.set_ylabel("Replay benefit (rounds)\nablation delay − replay delay", fontsize=13)
    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "replay_benefit.png", dpi=200)
    plt.close(fig)


def plot_kappa_sweep(kappa_summary: pd.DataFrame, out_dir: Path, chosen_kappa: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.4))
    ax.errorbar(
        kappa_summary["kappa"],
        kappa_summary["false_alarm_rate"],
        yerr=kappa_summary["false_alarm_rate_sem"],
        marker="o",
        markersize=7,
        linewidth=1.8,
        color=METHOD_COLORS["hh_replay"],
        ecolor=SECONDARY_INK,
        elinewidth=1.1,
        capsize=4,
    )
    ax.axhline(0.05, color=MUTED, linewidth=0.8, linestyle="--")
    ax.text(kappa_summary["kappa"].min(), 0.065, "5% target", fontsize=12, color=MUTED)
    ax.axvline(chosen_kappa, color=DIVERGING_NEGATIVE, linewidth=0.8, linestyle=":")
    ax.set_xlabel(r"Detection threshold $\kappa$", fontsize=14)
    ax.set_ylabel("False-alarm rate (stationary)", fontsize=14)
    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "kappa_sweep.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results")
    parser.add_argument("--out", type=str, default="AAMAS 2027")
    args = parser.parse_args()

    results_dir = Path(args.results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(results_dir / "main_summary.csv")
    kappa_summary = pd.read_csv(results_dir / "kappa_sweep_summary.csv")
    chosen_kappa = kappa_summary.loc[kappa_summary["false_alarm_rate"] <= 0.05, "kappa"].min()

    plot_cumulative_return(summary, out_dir)
    plot_adaptation_delay(summary, out_dir)
    plot_post_switch_regret(summary, out_dir)
    plot_replay_benefit(summary, out_dir)
    plot_kappa_sweep(kappa_summary, out_dir, chosen_kappa)
    n_figures = 5

    phase_path = results_dir / "phase_return_summary.csv"
    if phase_path.exists():
        plot_return_before_after(pd.read_csv(phase_path), out_dir)
        n_figures += 1

    print(f"Wrote {n_figures} figures to {out_dir}/")


if __name__ == "__main__":
    main()
