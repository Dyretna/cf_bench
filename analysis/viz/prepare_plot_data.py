# Import threshold constants
from ..reporter.summary import GOWER_HIGH_THRESHOLD, GOWER_LOW_THRESHOLD

# Language mappings
LABELS = {
    "sv": {
        "performance": {
            "Validity_%": "Validitet (%)",
            "Solved_%": "Lösta observationer (%)",
            "Time": "Körtid (s)",
            "Avg_NChanged_All": "Medel antal ändringar (Alla)",
        },
        "risk": {
            "Risk_Before": "Risk före (%)",
            "Risk_After": "Risk efter (%)",
            "Risk_Reduction_%": "Riskreduktion (%)",
            "Avg_NChanged_All": "Medel antal ändringar (Alla)",
            "Avg_NChanged_Valid": "Medel antal ändringar (Giltiga)",
        },
        "gower_all": {
            "Avg_Gower_All": "Medel Gower (Alla)",
            "Low_Gower_%_All": f"Låg Gower <{GOWER_LOW_THRESHOLD} % (Alla)",
            "High_Gower_%_All": f"Hög Gower >{GOWER_HIGH_THRESHOLD} % (Alla)",
            "Median_Gower_All": "Median Gower (Alla)",
        },
        "gower_valid": {
            "Avg_Gower_Valid": "Medel Gower (Giltiga)",
            "Low_Gower_%_Valid": f"Låg Gower <{GOWER_LOW_THRESHOLD} % (Giltiga)",
            "High_Gower_%_Valid": f"Hög Gower >{GOWER_HIGH_THRESHOLD} % (Giltiga)",
            "Min_Gower_Valid": "Min Gower (Giltiga)",
        },
    },
    "en": {
        "performance": {
            "Validity_%": "Validity (%)",
            "Solved_%": "Solved Observations (%)",
            "Time": "Time (s)",
            "Avg_NChanged_All": "Avg # Changes (All)",
        },
        "risk": {
            "Risk_Before": "Risk Before (%)",
            "Risk_After": "Risk After (%)",
            "Risk_Reduction_%": "Risk Reduction (%)",
            "Avg_NChanged_All": "Avg # Changes (All)",
            "Avg_NChanged_Valid": "Avg # Changes (Valid)",
        },
        "gower_all": {
            "Avg_Gower_All": "Avg Gower (All)",
            "Low_Gower_%_All": f"Low Gower <{GOWER_LOW_THRESHOLD} % (All)",
            "High_Gower_%_All": f"High Gower >{GOWER_HIGH_THRESHOLD} % (All)",
            "Median_Gower_All": "Median Gower (All)",
        },
        "gower_valid": {
            "Avg_Gower_Valid": "Avg Gower (Valid)",
            "Low_Gower_%_Valid": f"Low Gower <{GOWER_LOW_THRESHOLD} % (Valid)",
            "High_Gower_%_Valid": f"High Gower >{GOWER_HIGH_THRESHOLD} % (Valid)",
            "Min_Gower_Valid": "Min Gower (Valid)",
        },
    },
}


def prepare_plot_data(df):
    """
    Prepare melted dataframes for plotting.

    Args:
        df: DataFrame with processed metrics

    Returns:
        dict: Dictionary with melted dataframes for each row type
    """
    # Performance metrics (Row 1)
    df_performance = df.melt(
        id_vars=["Category", "Model", "Threshold"],
        value_vars=["Validity_%", "Solved_%", "Time"],
        var_name="Metric",
        value_name="Value",
    )

    # Risk & Effort metrics (Row 2)
    df_risk = df.melt(
        id_vars=["Category", "Model", "Threshold"],
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
        id_vars=["Category", "Model", "Threshold"],
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
        id_vars=["Category", "Model", "Threshold"],
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
