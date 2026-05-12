"""Centralized experiment paths - simple nested dictionary."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
CF_OUTPUTS = Path(os.getenv("CF_OUTPUTS"))

# Base directories
PRED_VS_THRESH = CF_OUTPUTS / "predictors_vs_threshold"
GEN1_EXP = CF_OUTPUTS / "gen_1_experiments"


# ------------------------------------------------------------------------------
# gen 2 experiments
# ------------------------------------------------------------------------------

# Baseline experiments
BASELINE = {
    "xgb": {
        0.1: PRED_VS_THRESH / "baseline" / "XGBoost_thres0.1_2026-05-11",
        0.5: PRED_VS_THRESH / "baseline" / "XGBoost_thres0.5_2026-05-11",
        0.9: PRED_VS_THRESH / "baseline" / "XGBoost_thres0.9_2026-05-11",
    },
    "rf": {
        0.1: PRED_VS_THRESH / "baseline" / "RandomForest_thres0.1_2026-05-11",
        0.5: PRED_VS_THRESH / "baseline" / "RandomForest_thres0.5_2026-05-11",
        0.9: PRED_VS_THRESH / "baseline" / "RandomForest_thres0.9_2026-05-11",
    },
}

# XGBoost optimized
XGB_OPTIMIZED = {
    0.1: PRED_VS_THRESH / "xgb_optimized" / "XGBoost_thres0.1_2026-05-11",
    0.5: PRED_VS_THRESH / "xgb_optimized" / "XGBoost_thres0.5_2026-05-11",
    0.9: PRED_VS_THRESH / "xgb_optimized" / "XGBoost_thres0.9_2026-05-11",
}

# Weighted predictors
WEIGHTED = {
    "xgb": {
        0.5: PRED_VS_THRESH / "weighted" / "XGBoost_thres0.5_2026-05-11",
    },
    "rf": {
        0.5: PRED_VS_THRESH / "weighted" / "RandomForest_thres0.5_2026-05-11",
    },
}

# SMOTE experiments
SMOTE = {
    "base": {
        "xgb": {
            0.5: PRED_VS_THRESH
            / "SMOTE"
            / "base_predictors"
            / "XGBoost_thres0.5_2026-05-11",
            0.9: PRED_VS_THRESH
            / "SMOTE"
            / "base_predictors"
            / "XGBoost_thres0.9_2026-05-11",
        },
        "rf": {
            0.5: PRED_VS_THRESH
            / "SMOTE"
            / "base_predictors"
            / "RandomForest_thres0.5_2026-05-11",
            0.9: PRED_VS_THRESH
            / "SMOTE"
            / "base_predictors"
            / "RandomForest_thres0.9_2026-05-11",
        },
    },
    "gridsearched": {
        "xgb": {
            0.5: PRED_VS_THRESH
            / "SMOTE"
            / "gridsearched_predictors"
            / "XGBoost_thres0.5_2026-05-11",
        },
        "rf": {
            0.5: PRED_VS_THRESH
            / "SMOTE"
            / "gridsearched_predictors"
            / "RandomForest_thres0.5_2026-05-11",
        },
    },
}
# ------------------------------------------------------------------------------
# gen 1 experiments
# ------------------------------------------------------------------------------

GEN1 = {
    "genetic": {
        "xgb_highthres": GEN1_EXP / "genetic_exp" / "XGB_highthres_2026-04-17",
        "xgb_prange_highthres": GEN1_EXP
        / "genetic_exp"
        / "XGB_prange_highthres_2026-04-17",
        "xgb_prange_lowthres": GEN1_EXP
        / "genetic_exp"
        / "XGB_prange_lowthres_2026-04-17",
        "rf_highthres": GEN1_EXP / "genetic_exp" / "RF_highthres_2026-04-21",
        "rf_lowthres": GEN1_EXP / "genetic_exp" / "RF_lowthres_2026-04-21",
        "rf_prange_highthres": GEN1_EXP
        / "genetic_exp"
        / "RF_prange_highthres_2026-04-29",
        "rf_prange_lowthres": GEN1_EXP
        / "genetic_exp"
        / "RF_prange_lowthres_2026-04-21",
    },
    "random": {
        "xgb_highthres": GEN1_EXP / "random_search_exp" / "XGB_highthres_2026-04-17",
        "xgb_prange_highthres": GEN1_EXP
        / "random_search_exp"
        / "XGB_prange_highthres_2026-04-17",
        "rf_highthres": GEN1_EXP / "random_search_exp" / "RF_highthres_2026-04-21",
        "rf_lowthres": GEN1_EXP / "random_search_exp" / "RF_lowthres_2026-04-21",
        "rf_prange_highthres": GEN1_EXP
        / "random_search_exp"
        / "RF_prange_highthres_2026-04-29",
    },
}

# ------------------------------------------------------------------------------
# Grouping Paths
# ------------------------------------------------------------------------------

# Predictors vs Threshold - all experiments grouped together
PRED_VS_THRESH_EXPERIMENTS = {
    "baseline": BASELINE,
    "xgb_optimized": XGB_OPTIMIZED,
    "weighted": WEIGHTED,
    "smote": SMOTE,
}

# Main paths dictionary - assembled from the above
PATHS = {
    "pred_vs_thresh": PRED_VS_THRESH_EXPERIMENTS,
    "gen1": GEN1,
}


# Usage examples:
# from paths import PATHS, BASELINE, XGB_OPTIMIZED, GEN1, PRED_VS_THRESH_EXPERIMENTS

# Using main PATHS dict (original structure intact):
# df = pd.read_csv(PATHS["pred_vs_thresh"]["baseline"]["xgb"][0.5] / "annotated.csv")
# df = pd.read_csv(PATHS["pred_vs_thresh"]["xgb_optimized"][0.1] / "annotated.csv")
# df = pd.read_csv(PATHS["gen1"]["genetic"]["rf_highthres"] / "annotated.csv")
