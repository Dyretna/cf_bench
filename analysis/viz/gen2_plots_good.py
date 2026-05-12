"""
Generation 2 Visualization for Baseline/Optimized Experiments

Plots threshold (0.1, 0.5, 0.9) on X-axis with three model configurations:
Baseline RF, Baseline XGB, Optimized XGB
"""

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Same labels
from .prepare_plot_data import LABELS


def prepare_plot_data_threshold(df):
    """
    Prepare melted dataframes for threshold-based plotting.
    Creates Model_Config column combining Category + Model.
    """
    # Create combined model configuration
    df = df.copy()
    df["Model_Config"] = df["Category"] + " " + df["Model"]

    # Performance metrics (Row 1)
    df_performance = df.melt(
        id_vars=["Model_Config", "Threshold"],
        value_vars=["Validity_%", "Solved_%", "Time"],
        var_name="Metric",
        value_name="Value",
    )

    # Risk & Effort metrics (Row 2)
    df_risk = df.melt(
        id_vars=["Model_Config", "Threshold"],
        value_vars=[
            "Risk_Before",
            "Risk_Reduction_%",
            "Avg_NChanged_All",
            "Avg_NChanged_Valid",
        ],
        var_name="Metric",
        value_name="Value",
    )

    # Gower ALL metrics
    df_gower_all = df.melt(
        id_vars=["Model_Config", "Threshold"],
        value_vars=[
            "Avg_Gower_All",
            "Low_Gower_%_All",
            "High_Gower_%_All",
            "Median_Gower_All",
        ],
        var_name="Metric",
        value_name="Value",
    )

    # Gower VALID metrics
    df_gower_valid = df.melt(
        id_vars=["Model_Config", "Threshold"],
        value_vars=[
            "Avg_Gower_Valid",
            "Low_Gower_%_Valid",
            "High_Gower_%_Valid",
            "Min_Gower_Valid",
        ],
        var_name="Metric",
        value_name="Value",
    )

    return {
        "performance": df_performance,
        "risk": df_risk,
        "gower_all": df_gower_all,
        "gower_valid": df_gower_valid,
    }


def _plot_row_performance(
    axes, row_idx, df_performance, language, threshold_order, style_info
):
    """Plot performance metrics row (Validity, Solved, Time)."""
    metrics = ["Validity_%", "Solved_%", "Time"]
    labels = LABELS[language]["performance"]

    for i, metric in enumerate(metrics):
        ax = axes[row_idx, i]

        for model_config in style_info["model_configs"]:
            metric_data = df_performance[
                (df_performance["Model_Config"] == model_config)
                & (df_performance["Metric"] == metric)
            ].copy()
            if not metric_data.empty:
                metric_data = metric_data.sort_values("Threshold")

                ax.plot(
                    metric_data["Threshold"],
                    metric_data["Value"],
                    marker=style_info["markers"][model_config],
                    color=style_info["colors"][model_config],
                    linestyle=style_info["linestyles"][model_config],
                    markersize=8,
                    markeredgewidth=1.5,
                    markeredgecolor=style_info["colors"][model_config],
                    markerfacecolor=style_info["colors"][model_config],
                    linewidth=2.5,
                    alpha=0.7,
                    label=model_config,
                )

        ax.set_ylabel(labels[metric], fontsize=10)
        ax.set_title(labels[metric], fontsize=11, fontweight="bold")
        ax.set_xlabel("Threshold", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(threshold_order)
        ax.set_xticklabels([str(t) for t in threshold_order])

        if metric in ["Validity_%", "Solved_%"]:
            ax.set_ylim(0, 105)

    # Hide the 4th column (empty slot)
    axes[row_idx, 3].axis("off")


def _plot_row_risk(axes, row_idx, df_risk, language, threshold_order, style_info):
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

        for model_config in style_info["model_configs"]:
            metric_data = df_risk[
                (df_risk["Model_Config"] == model_config)
                & (df_risk["Metric"] == metric)
            ].copy()
            if not metric_data.empty:
                metric_data = metric_data.sort_values("Threshold")

                ax.plot(
                    metric_data["Threshold"],
                    metric_data["Value"],
                    marker=style_info["markers"][model_config],
                    color=style_info["colors"][model_config],
                    linestyle=style_info["linestyles"][model_config],
                    markersize=8,
                    markeredgewidth=1.5,
                    markeredgecolor=style_info["colors"][model_config],
                    markerfacecolor=style_info["colors"][model_config],
                    linewidth=2.5,
                    alpha=0.7,
                    label=model_config,
                )

        ax.set_ylabel(labels[metric], fontsize=10)
        ax.set_title(labels[metric], fontsize=11, fontweight="bold")
        ax.set_xlabel("Threshold", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(threshold_order)
        ax.set_xticklabels([str(t) for t in threshold_order])

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
    axes, row_idx, df_gower_all, language, threshold_order, style_info
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

        for model_config in style_info["model_configs"]:
            metric_data = df_gower_all[
                (df_gower_all["Model_Config"] == model_config)
                & (df_gower_all["Metric"] == metric)
            ].copy()
            if not metric_data.empty:
                metric_data = metric_data.sort_values("Threshold")

                ax.plot(
                    metric_data["Threshold"],
                    metric_data["Value"],
                    marker=style_info["markers"][model_config],
                    color=style_info["colors"][model_config],
                    linestyle=style_info["linestyles"][model_config],
                    markersize=8,
                    markeredgewidth=1.5,
                    markeredgecolor=style_info["colors"][model_config],
                    markerfacecolor=style_info["colors"][model_config],
                    linewidth=2.5,
                    alpha=0.7,
                    label=model_config,
                )

        ax.set_ylabel(labels[metric], fontsize=10)
        ax.set_title(labels[metric], fontsize=11, fontweight="bold")
        ax.set_xlabel("Threshold", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(threshold_order)
        ax.set_xticklabels([str(t) for t in threshold_order])

        # Hardcoded limits
        if "%" in metric:
            ax.set_ylim(-2, 25)
        else:
            ax.set_ylim(0, 0.3)


def _plot_row_gower_valid(
    axes, row_idx, df_gower_valid, language, threshold_order, style_info
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

        for model_config in style_info["model_configs"]:
            metric_data = df_gower_valid[
                (df_gower_valid["Model_Config"] == model_config)
                & (df_gower_valid["Metric"] == metric)
            ].copy()
            if not metric_data.empty:
                metric_data = metric_data.sort_values("Threshold")

                ax.plot(
                    metric_data["Threshold"],
                    metric_data["Value"],
                    marker=style_info["markers"][model_config],
                    color=style_info["colors"][model_config],
                    linestyle=style_info["linestyles"][model_config],
                    markersize=8,
                    markeredgewidth=1.5,
                    markeredgecolor=style_info["colors"][model_config],
                    markerfacecolor=style_info["colors"][model_config],
                    linewidth=2.5,
                    alpha=0.7,
                    label=model_config,
                )

        ax.set_ylabel(labels[metric], fontsize=10)
        ax.set_title(labels[metric], fontsize=11, fontweight="bold")
        ax.set_xlabel("Threshold", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_xticks(threshold_order)
        ax.set_xticklabels([str(t) for t in threshold_order])

        # Hardcoded limits
        if "%" in metric:
            ax.set_ylim(-2, 25)
        else:
            ax.set_ylim(-0.01, 0.3)


def create_comprehensive_plot_good(
    df, language="en", figsize=(16, 14), title=None, output_path=None
):
    """
    Create comprehensive 4-row analysis plot for baseline/optimized experiments.

    X-axis: Thresholds (0.1, 0.5, 0.9)
    Lines: 3 model configurations (Baseline RF, Baseline XGB, Optimized XGB)

    Args:
        df: DataFrame with processed metrics (needs Category, Model, Threshold columns)
        language: 'en' or 'sv'
        figsize: Figure size tuple
        title: Optional plot title
        output_path: Optional path to save figure

    Returns:
        fig, axes
    """
    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(4, 4, figsize=figsize)

    # Prepare threshold-based data
    df_dict = prepare_plot_data_threshold(df)

    # Threshold order
    threshold_order = [0.1, 0.5, 0.9]

    # Style configuration for 3 model configurations
    style_info = {
        "model_configs": [
            "Baseline RandomForest",
            "Baseline XGBoost",
            "Optimized XGBoost XGBoost",
        ],
        "colors": {
            "Baseline RandomForest": "#0173B2",
            "Baseline XGBoost": "#DE8F05",
            "Optimized XGBoost XGBoost": "#029E73",
        },
        "linestyles": {
            "Baseline RandomForest": "-",
            "Baseline XGBoost": "-",
            "Optimized XGBoost XGBoost": "--",
        },
        "markers": {
            "Baseline RandomForest": "o",
            "Baseline XGBoost": "s",
            "Optimized XGBoost XGBoost": "^",
        },
    }

    # Plot each row
    _plot_row_performance(
        axes, 0, df_dict["performance"], language, threshold_order, style_info
    )
    _plot_row_risk(axes, 1, df_dict["risk"], language, threshold_order, style_info)
    _plot_row_gower_all(
        axes, 2, df_dict["gower_all"], language, threshold_order, style_info
    )
    _plot_row_gower_valid(
        axes, 3, df_dict["gower_valid"], language, threshold_order, style_info
    )

    # Create legend
    legend_labels = {
        "sv": {
            "Baseline RandomForest": "Baslinje RF",
            "Baseline XGBoost": "Baslinje XGB",
            "Optimized XGBoost XGBoost": "Optimerad XGB",
        },
        "en": {
            "Baseline RandomForest": "Baseline RF",
            "Baseline XGBoost": "Baseline XGB",
            "Optimized XGBoost XGBoost": "Optimized XGB",
        },
    }

    legend_elements = [
        Line2D(
            [0],
            [0],
            color=style_info["colors"]["Baseline RandomForest"],
            marker=style_info["markers"]["Baseline RandomForest"],
            markersize=8,
            markerfacecolor=style_info["colors"]["Baseline RandomForest"],
            markeredgewidth=1.5,
            label=legend_labels[language]["Baseline RandomForest"],
            linestyle=style_info["linestyles"]["Baseline RandomForest"],
            linewidth=2.5,
            alpha=0.7,
        ),
        Line2D(
            [0],
            [0],
            color=style_info["colors"]["Baseline XGBoost"],
            marker=style_info["markers"]["Baseline XGBoost"],
            markersize=8,
            markerfacecolor=style_info["colors"]["Baseline XGBoost"],
            markeredgewidth=1.5,
            label=legend_labels[language]["Baseline XGBoost"],
            linestyle=style_info["linestyles"]["Baseline XGBoost"],
            linewidth=2.5,
            alpha=0.7,
        ),
        Line2D(
            [0],
            [0],
            color=style_info["colors"]["Optimized XGBoost XGBoost"],
            marker=style_info["markers"]["Optimized XGBoost XGBoost"],
            markersize=8,
            markerfacecolor=style_info["colors"]["Optimized XGBoost XGBoost"],
            markeredgewidth=1.5,
            label=legend_labels[language]["Optimized XGBoost XGBoost"],
            linestyle=style_info["linestyles"]["Optimized XGBoost XGBoost"],
            linewidth=2.5,
            alpha=0.7,
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        fontsize=9,
        ncol=3,
    )

    # Add title if provided
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
        plt.tight_layout()
        plt.subplots_adjust(top=0.955, left=0.08, right=0.98, hspace=0.35)
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


def create_performance_risk_plot_good(
    df, language="en", figsize=(16, 7), title=None, output_path=None
):
    """
    Create 2-row plot: Performance + Risk (Rows 1-2).

    Args:
        df: DataFrame with processed metrics (needs Category, Model, Threshold columns)
        language: 'en' or 'sv'
        figsize: Figure size tuple
        title: Optional plot title
        output_path: Optional path to save figure

    Returns:
        fig, axes
    """
    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(2, 4, figsize=figsize)

    # Prepare threshold-based data
    df_dict = prepare_plot_data_threshold(df)

    # Threshold order
    threshold_order = [0.1, 0.5, 0.9]

    # Style configuration for 3 model configurations
    style_info = {
        "model_configs": [
            "Baseline RandomForest",
            "Baseline XGBoost",
            "Optimized XGBoost XGBoost",
        ],
        "colors": {
            "Baseline RandomForest": "#0173B2",
            "Baseline XGBoost": "#DE8F05",
            "Optimized XGBoost XGBoost": "#029E73",
        },
        "linestyles": {
            "Baseline RandomForest": "-",
            "Baseline XGBoost": "-",
            "Optimized XGBoost XGBoost": "--",
        },
        "markers": {
            "Baseline RandomForest": "o",
            "Baseline XGBoost": "s",
            "Optimized XGBoost XGBoost": "^",
        },
    }

    # Plot rows
    _plot_row_performance(
        axes, 0, df_dict["performance"], language, threshold_order, style_info
    )
    _plot_row_risk(axes, 1, df_dict["risk"], language, threshold_order, style_info)

    # Create legend
    legend_labels = {
        "sv": {
            "Baseline RandomForest": "Baslinje RF",
            "Baseline XGBoost": "Baslinje XGB",
            "Optimized XGBoost XGBoost": "Optimerad XGB",
        },
        "en": {
            "Baseline RandomForest": "Baseline RF",
            "Baseline XGBoost": "Baseline XGB",
            "Optimized XGBoost XGBoost": "Optimized XGB",
        },
    }

    legend_elements = [
        Line2D(
            [0],
            [0],
            color=style_info["colors"]["Baseline RandomForest"],
            marker=style_info["markers"]["Baseline RandomForest"],
            markersize=8,
            markerfacecolor=style_info["colors"]["Baseline RandomForest"],
            markeredgewidth=1.5,
            label=legend_labels[language]["Baseline RandomForest"],
            linestyle=style_info["linestyles"]["Baseline RandomForest"],
            linewidth=2.5,
            alpha=0.7,
        ),
        Line2D(
            [0],
            [0],
            color=style_info["colors"]["Baseline XGBoost"],
            marker=style_info["markers"]["Baseline XGBoost"],
            markersize=8,
            markerfacecolor=style_info["colors"]["Baseline XGBoost"],
            markeredgewidth=1.5,
            label=legend_labels[language]["Baseline XGBoost"],
            linestyle=style_info["linestyles"]["Baseline XGBoost"],
            linewidth=2.5,
            alpha=0.7,
        ),
        Line2D(
            [0],
            [0],
            color=style_info["colors"]["Optimized XGBoost XGBoost"],
            marker=style_info["markers"]["Optimized XGBoost XGBoost"],
            markersize=8,
            markerfacecolor=style_info["colors"]["Optimized XGBoost XGBoost"],
            markeredgewidth=1.5,
            label=legend_labels[language]["Optimized XGBoost XGBoost"],
            linestyle=style_info["linestyles"]["Optimized XGBoost XGBoost"],
            linewidth=2.5,
            alpha=0.7,
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        fontsize=9,
        ncol=3,
    )

    # Add title if provided
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, left=0.08, right=0.98, hspace=0.35)
    else:
        plt.tight_layout()
        plt.subplots_adjust(top=0.96, left=0.08, right=0.98, hspace=0.35)

    # Save if requested
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, axes


def create_gower_plot_good(
    df, language="en", figsize=(16, 7), title=None, output_path=None
):
    """
    Create 2-row plot: Gower ALL + Gower VALID (Rows 3-4).

    Args:
        df: DataFrame with processed metrics (needs Category, Model, Threshold columns)
        language: 'en' or 'sv'
        figsize: Figure size tuple
        title: Optional plot title
        output_path: Optional path to save figure

    Returns:
        fig, axes
    """
    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(2, 4, figsize=figsize)

    # Prepare threshold-based data
    df_dict = prepare_plot_data_threshold(df)

    # Threshold order
    threshold_order = [0.1, 0.5, 0.9]

    # Style configuration for 3 model configurations
    style_info = {
        "model_configs": [
            "Baseline RandomForest",
            "Baseline XGBoost",
            "Optimized XGBoost XGBoost",
        ],
        "colors": {
            "Baseline RandomForest": "#0173B2",
            "Baseline XGBoost": "#DE8F05",
            "Optimized XGBoost XGBoost": "#029E73",
        },
        "linestyles": {
            "Baseline RandomForest": "-",
            "Baseline XGBoost": "-",
            "Optimized XGBoost XGBoost": "--",
        },
        "markers": {
            "Baseline RandomForest": "o",
            "Baseline XGBoost": "s",
            "Optimized XGBoost XGBoost": "^",
        },
    }

    # Plot rows
    _plot_row_gower_all(
        axes, 0, df_dict["gower_all"], language, threshold_order, style_info
    )
    _plot_row_gower_valid(
        axes, 1, df_dict["gower_valid"], language, threshold_order, style_info
    )

    # Create legend
    legend_labels = {
        "sv": {
            "Baseline RandomForest": "Baslinje RF",
            "Baseline XGBoost": "Baslinje XGB",
            "Optimized XGBoost XGBoost": "Optimerad XGB",
        },
        "en": {
            "Baseline RandomForest": "Baseline RF",
            "Baseline XGBoost": "Baseline XGB",
            "Optimized XGBoost XGBoost": "Optimized XGB",
        },
    }

    legend_elements = [
        Line2D(
            [0],
            [0],
            color=style_info["colors"]["Baseline RandomForest"],
            marker=style_info["markers"]["Baseline RandomForest"],
            markersize=8,
            markerfacecolor=style_info["colors"]["Baseline RandomForest"],
            markeredgewidth=1.5,
            label=legend_labels[language]["Baseline RandomForest"],
            linestyle=style_info["linestyles"]["Baseline RandomForest"],
            linewidth=2.5,
            alpha=0.7,
        ),
        Line2D(
            [0],
            [0],
            color=style_info["colors"]["Baseline XGBoost"],
            marker=style_info["markers"]["Baseline XGBoost"],
            markersize=8,
            markerfacecolor=style_info["colors"]["Baseline XGBoost"],
            markeredgewidth=1.5,
            label=legend_labels[language]["Baseline XGBoost"],
            linestyle=style_info["linestyles"]["Baseline XGBoost"],
            linewidth=2.5,
            alpha=0.7,
        ),
        Line2D(
            [0],
            [0],
            color=style_info["colors"]["Optimized XGBoost XGBoost"],
            marker=style_info["markers"]["Optimized XGBoost XGBoost"],
            markersize=8,
            markerfacecolor=style_info["colors"]["Optimized XGBoost XGBoost"],
            markeredgewidth=1.5,
            label=legend_labels[language]["Optimized XGBoost XGBoost"],
            linestyle=style_info["linestyles"]["Optimized XGBoost XGBoost"],
            linewidth=2.5,
            alpha=0.7,
        ),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        fontsize=9,
        ncol=3,
    )

    # Add title if provided
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, left=0.08, right=0.98, hspace=0.35)
    else:
        plt.tight_layout()
        plt.subplots_adjust(top=0.96, left=0.08, right=0.98, hspace=0.35)

    # Save if requested
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig, axes
