"""
Generation 2 Experiment Visualization Module

Simple, modular functions for creating comprehensive analysis plots.
Each row is built separately, then combined into a single figure.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from .prepare_plot_data import LABELS


def _plot_row_performance(
    axes, row_idx, df_performance, language, category_info, style_info
):
    """Plot performance metrics row (Validity, Solved, Time)."""
    metrics = ["Validity_%", "Solved_%", "Time"]
    labels = LABELS[language]["performance"]

    for i, metric in enumerate(metrics):
        ax = axes[row_idx, i]

        for model in style_info["models"]:
            metric_data = df_performance[
                (df_performance["Model"] == model)
                & (df_performance["Metric"] == metric)
            ].copy()
            if not metric_data.empty:
                grouped = metric_data.groupby("Category")["Value"].mean().reset_index()
                grouped["x_pos"] = grouped["Category"].map(category_info["numeric"])
                grouped = grouped.sort_values("x_pos")

                ax.plot(
                    grouped["x_pos"],
                    grouped["Value"],
                    marker=style_info["markers"][model],
                    color=style_info["colors"][model],
                    linestyle=style_info["linestyles"][model],
                    markersize=8,
                    markeredgewidth=1.5,
                    markeredgecolor=style_info["colors"][model],
                    markerfacecolor=style_info["colors"][model],
                    linewidth=2.5,
                    alpha=0.7,
                    label=model,
                )

        ax.set_ylabel(labels[metric], fontsize=10)
        ax.set_title(labels[metric], fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(list(category_info["numeric"].values()))
        ax.set_xticklabels(list(category_info["labels"].values()), fontsize=8)

        if metric in ["Validity_%", "Solved_%"]:
            ax.set_ylim(0, 105)

    # Hide the 4th column (empty slot)
    axes[row_idx, 3].axis("off")


def _plot_row_risk(axes, row_idx, df_risk, language, category_info, style_info):
    """Plot risk & effort row (Risk Before, Risk Reduction, Avg NChanged All, Avg NChanged Valid)."""
    metrics = [
        "Risk_Before",
        "Risk_Reduction_%",
        "Avg_NChanged_All",
        "Avg_NChanged_Valid",
    ]
    labels = LABELS[language]["risk"]

    for i, metric in enumerate(metrics):
        ax = axes[row_idx, i]

        for model in style_info["models"]:
            metric_data = df_risk[
                (df_risk["Model"] == model) & (df_risk["Metric"] == metric)
            ].copy()
            if not metric_data.empty:
                grouped = metric_data.groupby("Category")["Value"].mean().reset_index()
                grouped["x_pos"] = grouped["Category"].map(category_info["numeric"])
                grouped = grouped.sort_values("x_pos")

                ax.plot(
                    grouped["x_pos"],
                    grouped["Value"],
                    marker=style_info["markers"][model],
                    color=style_info["colors"][model],
                    linestyle=style_info["linestyles"][model],
                    markersize=8,
                    markeredgewidth=1.5,
                    markeredgecolor=style_info["colors"][model],
                    markerfacecolor=style_info["colors"][model],
                    linewidth=2.5,
                    alpha=0.7,
                    label=model,
                )

        ax.set_ylabel(labels[metric], fontsize=10)
        ax.set_title(labels[metric], fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(list(category_info["numeric"].values()))
        ax.set_xticklabels(list(category_info["labels"].values()), fontsize=8)

        if metric in ["Risk_Before", "Risk_After"]:
            max_val = df_risk[df_risk["Metric"].isin(["Risk_Before", "Risk_After"])][
                "Value"
            ].max()
            ax.set_ylim(0, max(80, max_val * 1.1))
        elif metric == "Risk_Reduction_%":
            ax.set_ylim(0, 105)
        elif metric in ["Avg_NChanged_All", "Avg_NChanged_Valid"]:
            ax.set_ylim(1.5, 4.0)


def _plot_row_gower_all(
    axes, row_idx, df_gower_all, language, category_info, style_info
):
    """Plot Gower distance metrics for ALL CFs."""
    metrics = [
        "Avg_Gower_All",
        "Low_Gower_%_All",
        "High_Gower_%_All",
        "Median_Gower_All",
    ]
    labels = LABELS[language]["gower_all"]

    for i, metric in enumerate(metrics):
        ax = axes[row_idx, i]

        for model in style_info["models"]:
            metric_data = df_gower_all[
                (df_gower_all["Model"] == model) & (df_gower_all["Metric"] == metric)
            ].copy()
            if not metric_data.empty:
                grouped = metric_data.groupby("Category")["Value"].mean().reset_index()
                grouped["x_pos"] = grouped["Category"].map(category_info["numeric"])
                grouped = grouped.sort_values("x_pos")

                ax.plot(
                    grouped["x_pos"],
                    grouped["Value"],
                    marker=style_info["markers"][model],
                    color=style_info["colors"][model],
                    linestyle=style_info["linestyles"][model],
                    markersize=8,
                    markeredgewidth=1.5,
                    markeredgecolor=style_info["colors"][model],
                    markerfacecolor=style_info["colors"][model],
                    linewidth=2.5,
                    alpha=0.7,
                    label=model,
                )

        ax.set_ylabel(labels[metric], fontsize=10)
        ax.set_title(labels[metric], fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(list(category_info["numeric"].values()))
        ax.set_xticklabels(list(category_info["labels"].values()), fontsize=8)

        # Hardcoded limits based on metric type
        if "%" in metric:
            ax.set_ylim(-2, 25)
        else:
            ax.set_ylim(0, 0.3)


def _plot_row_gower_valid(
    axes, row_idx, df_gower_valid, language, category_info, style_info
):
    """Plot Gower distance metrics for VALID CFs only."""
    metrics = [
        "Avg_Gower_Valid",
        "Low_Gower_%_Valid",
        "High_Gower_%_Valid",
        "Min_Gower_Valid",
    ]
    labels = LABELS[language]["gower_valid"]

    for i, metric in enumerate(metrics):
        ax = axes[row_idx, i]

        for model in style_info["models"]:
            metric_data = df_gower_valid[
                (df_gower_valid["Model"] == model)
                & (df_gower_valid["Metric"] == metric)
            ].copy()
            if not metric_data.empty:
                grouped = metric_data.groupby("Category")["Value"].mean().reset_index()
                grouped["x_pos"] = grouped["Category"].map(category_info["numeric"])
                grouped = grouped.sort_values("x_pos")

                ax.plot(
                    grouped["x_pos"],
                    grouped["Value"],
                    marker=style_info["markers"][model],
                    color=style_info["colors"][model],
                    linestyle=style_info["linestyles"][model],
                    markersize=8,
                    markeredgewidth=1.5,
                    markeredgecolor=style_info["colors"][model],
                    markerfacecolor=style_info["colors"][model],
                    linewidth=2.5,
                    alpha=0.7,
                    label=model,
                )

        ax.set_ylabel(labels[metric], fontsize=10)
        ax.set_title(labels[metric], fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(list(category_info["numeric"].values()))
        ax.set_xticklabels(list(category_info["labels"].values()), fontsize=8)

        # Hardcoded limits based on metric type
        if "%" in metric:
            ax.set_ylim(-2, 25)
        else:
            ax.set_ylim(-0.01, 0.3)


def create_comprehensive_plot(
    df_dict,
    category_info,
    language="en",
    figsize=(16, 14),
    title=None,
    output_path=None,
):
    """
    Create comprehensive 4-row analysis plot.

    Args:
        df_dict: Dictionary with melted dataframes from prepare_plot_data()
        category_info: Dict with 'order', 'labels', 'numeric' for x-axis
        language: 'en' or 'sv'
        figsize: Figure size tuple
        title: Optional plot title
        output_path: Optional path to save figure

    Returns:
        fig, axes
    """
    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(4, 4, figsize=figsize)

    # Style configuration
    style_info = {
        "models": ["RandomForest", "XGBoost"],
        "colors": {"RandomForest": "#0173B2", "XGBoost": "#DE8F05"},
        "linestyles": {"RandomForest": "-", "XGBoost": "--"},
        "markers": {"RandomForest": "o", "XGBoost": "s"},
    }

    # Plot each row
    _plot_row_performance(
        axes, 0, df_dict["performance"], language, category_info, style_info
    )
    _plot_row_risk(axes, 1, df_dict["risk"], language, category_info, style_info)
    _plot_row_gower_all(
        axes, 2, df_dict["gower_all"], language, category_info, style_info
    )
    _plot_row_gower_valid(
        axes, 3, df_dict["gower_valid"], language, category_info, style_info
    )

    # Create legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=style_info["colors"]["RandomForest"],
            marker=style_info["markers"]["RandomForest"],
            markersize=8,
            markerfacecolor=style_info["colors"]["RandomForest"],
            markeredgewidth=1.5,
            label="RandomForest",
            linestyle=style_info["linestyles"]["RandomForest"],
            linewidth=2.5,
            alpha=0.7,
        ),
        Line2D(
            [0],
            [0],
            color=style_info["colors"]["XGBoost"],
            marker=style_info["markers"]["XGBoost"],
            markersize=8,
            markerfacecolor=style_info["colors"]["XGBoost"],
            markeredgewidth=1.5,
            label="XGBoost",
            linestyle=style_info["linestyles"]["XGBoost"],
            linewidth=2.5,
            alpha=0.7,
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        frameon=True,
        fontsize=9,
        ncol=1,
    )

    # Add title if provided
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
        plt.tight_layout()
        plt.subplots_adjust(top=0.955, left=0.06, right=0.95, hspace=0.3)
    else:
        # Add default Swedish title if language is Swedish
        if language == "sv":
            fig.suptitle(
                "Analysöversikt: Prestanda, Risk och Gower-avstånd",
                fontsize=13,
                fontweight="bold",
                y=0.995,
            )
            plt.tight_layout()
            plt.subplots_adjust(top=0.955, left=0.06, right=0.95, hspace=0.3)
        else:
            plt.tight_layout()
            plt.subplots_adjust(top=0.96, left=0.06, right=0.95, hspace=0.3)

    # Save if requested
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, axes


def create_performance_risk_plot(
    df_dict, category_info, language="en", figsize=(16, 7), title=None, output_path=None
):
    """
    Create 2-row plot: Performance + Risk (Rows 1-2).

    Args:
        df_dict: Dictionary with melted dataframes from prepare_plot_data()
        category_info: Dict with 'order', 'labels', 'numeric' for x-axis
        language: 'en' or 'sv'
        figsize: Figure size tuple
        title: Optional plot title
        output_path: Optional path to save figure

    Returns:
        fig, axes
    """
    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(2, 4, figsize=figsize)

    # Style configuration
    style_info = {
        "models": ["RandomForest", "XGBoost"],
        "colors": {"RandomForest": "#0173B2", "XGBoost": "#DE8F05"},
        "linestyles": {"RandomForest": "-", "XGBoost": "--"},
        "markers": {"RandomForest": "o", "XGBoost": "s"},
    }

    # Plot rows
    _plot_row_performance(
        axes, 0, df_dict["performance"], language, category_info, style_info
    )
    _plot_row_risk(axes, 1, df_dict["risk"], language, category_info, style_info)

    # Create legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=style_info["colors"]["RandomForest"],
            marker=style_info["markers"]["RandomForest"],
            markersize=8,
            markerfacecolor=style_info["colors"]["RandomForest"],
            markeredgewidth=1.5,
            label="RandomForest",
            linestyle=style_info["linestyles"]["RandomForest"],
            linewidth=2.5,
            alpha=0.7,
        ),
        Line2D(
            [0],
            [0],
            color=style_info["colors"]["XGBoost"],
            marker=style_info["markers"]["XGBoost"],
            markersize=8,
            markerfacecolor=style_info["colors"]["XGBoost"],
            markeredgewidth=1.5,
            label="XGBoost",
            linestyle=style_info["linestyles"]["XGBoost"],
            linewidth=2.5,
            alpha=0.7,
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        frameon=True,
        fontsize=9,
        ncol=1,
    )

    # Add title if provided
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, left=0.06, right=0.95, hspace=0.3)
    else:
        plt.tight_layout()
        plt.subplots_adjust(top=0.96, left=0.06, right=0.95, hspace=0.3)

    # Save if requested
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, axes


def create_gower_plot(
    df_dict, category_info, language="en", figsize=(16, 7), title=None, output_path=None
):
    """
    Create 2-row plot: Gower ALL + Gower VALID (Rows 3-4).

    Args:
        df_dict: Dictionary with melted dataframes from prepare_plot_data()
        category_info: Dict with 'order', 'labels', 'numeric' for x-axis
        language: 'en' or 'sv'
        figsize: Figure size tuple
        title: Optional plot title
        output_path: Optional path to save figure

    Returns:
        fig, axes
    """
    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(2, 4, figsize=figsize)

    # Style configuration
    style_info = {
        "models": ["RandomForest", "XGBoost"],
        "colors": {"RandomForest": "#0173B2", "XGBoost": "#DE8F05"},
        "linestyles": {"RandomForest": "-", "XGBoost": "--"},
        "markers": {"RandomForest": "o", "XGBoost": "s"},
    }

    # Plot rows
    _plot_row_gower_all(
        axes, 0, df_dict["gower_all"], language, category_info, style_info
    )
    _plot_row_gower_valid(
        axes, 1, df_dict["gower_valid"], language, category_info, style_info
    )

    # Create legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=style_info["colors"]["RandomForest"],
            marker=style_info["markers"]["RandomForest"],
            markersize=8,
            markerfacecolor=style_info["colors"]["RandomForest"],
            markeredgewidth=1.5,
            label="RandomForest",
            linestyle=style_info["linestyles"]["RandomForest"],
            linewidth=2.5,
            alpha=0.7,
        ),
        Line2D(
            [0],
            [0],
            color=style_info["colors"]["XGBoost"],
            marker=style_info["markers"]["XGBoost"],
            markersize=8,
            markerfacecolor=style_info["colors"]["XGBoost"],
            markeredgewidth=1.5,
            label="XGBoost",
            linestyle=style_info["linestyles"]["XGBoost"],
            linewidth=2.5,
            alpha=0.7,
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        frameon=True,
        fontsize=9,
        ncol=1,
    )

    # Add title if provided
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, left=0.06, right=0.95, hspace=0.3)
    else:
        plt.tight_layout()
        plt.subplots_adjust(top=0.96, left=0.06, right=0.95, hspace=0.3)

    # Save if requested
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, axes
