"""Centralized experiment paths - simple nested dictionary."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
CF_OUTPUTS = Path(os.getenv("CF_OUTPUTS"))

# All experiment paths organized by type
PATHS = {
    "baseline": {
        "xgb": {
            0.1: CF_OUTPUTS
            / "predictors_vs_threshold/baseline/XGBoost_thres0.1_2026-05-11",
            0.5: CF_OUTPUTS
            / "predictors_vs_threshold/baseline/XGBoost_thres0.5_2026-05-11",
            0.9: CF_OUTPUTS
            / "predictors_vs_threshold/baseline/XGBoost_thres0.9_2026-05-11",
        },
        "rf": {
            0.1: CF_OUTPUTS
            / "predictors_vs_threshold/baseline/RandomForest_thres0.1_2026-05-11",
            0.5: CF_OUTPUTS
            / "predictors_vs_threshold/baseline/RandomForest_thres0.5_2026-05-11",
            0.9: CF_OUTPUTS
            / "predictors_vs_threshold/baseline/RandomForest_thres0.9_2026-05-11",
        },
    },
    "xgb_optimized": {
        0.1: CF_OUTPUTS
        / "predictors_vs_threshold/xgb_optimized/XGBoost_thres0.1_2026-05-11",
        0.5: CF_OUTPUTS
        / "predictors_vs_threshold/xgb_optimized/XGBoost_thres0.5_2026-05-11",
        0.9: CF_OUTPUTS
        / "predictors_vs_threshold/xgb_optimized/XGBoost_thres0.9_2026-05-11",
    },
    "weighted": {
        "xgb": {
            0.5: CF_OUTPUTS
            / "predictors_vs_threshold/weighted/XGBoost_thres0.5_2026-05-11",
        },
        "rf": {
            0.5: CF_OUTPUTS
            / "predictors_vs_threshold/weighted/RandomForest_thres0.5_2026-05-11",
        },
    },
    "smote": {
        "base": {
            "xgb": {
                0.5: CF_OUTPUTS
                / "predictors_vs_threshold/SMOTE/base_predictors/XGBoost_thres0.5_2026-05-11",
                0.9: CF_OUTPUTS
                / "predictors_vs_threshold/SMOTE/base_predictors/XGBoost_thres0.9_2026-05-11",
            },
            "rf": {
                0.5: CF_OUTPUTS
                / "predictors_vs_threshold/SMOTE/base_predictors/RandomForest_thres0.5_2026-05-11",
                0.9: CF_OUTPUTS
                / "predictors_vs_threshold/SMOTE/base_predictors/RandomForest_thres0.9_2026-05-11",
            },
        },
        "gridsearched": {
            "xgb": {
                0.5: CF_OUTPUTS
                / "predictors_vs_threshold/SMOTE/gridsearched_predictors/XGBoost_thres0.5_2026-05-11",
            },
            "rf": {
                0.5: CF_OUTPUTS
                / "predictors_vs_threshold/SMOTE/gridsearched_predictors/RandomForest_thres0.5_2026-05-11",
            },
        },
    },
    "gen1": {
        "genetic": {
            "xgb_highthres": CF_OUTPUTS
            / "gen_1_experiments/genetic_exp/XGB_highthres_2026-04-17",
            "xgb_prange_highthres": CF_OUTPUTS
            / "gen_1_experiments/genetic_exp/XGB_prange_highthres_2026-04-17",
            "xgb_prange_lowthres": CF_OUTPUTS
            / "gen_1_experiments/genetic_exp/XGB_prange_lowthres_2026-04-17",
            "rf_highthres": CF_OUTPUTS
            / "gen_1_experiments/genetic_exp/RF_highthres_2026-04-21",
            "rf_lowthres": CF_OUTPUTS
            / "gen_1_experiments/genetic_exp/RF_lowthres_2026-04-21",
            "rf_prange_highthres": CF_OUTPUTS
            / "gen_1_experiments/genetic_exp/RF_prange_highthres_2026-04-29",
            "rf_prange_lowthres": CF_OUTPUTS
            / "gen_1_experiments/genetic_exp/RF_prange_lowthres_2026-04-21",
        },
        "random": {
            "xgb_highthres": CF_OUTPUTS
            / "gen_1_experiments/random_search_exp/XGB_highthres_2026-04-17",
            "xgb_prange_highthres": CF_OUTPUTS
            / "gen_1_experiments/random_search_exp/XGB_prange_highthres_2026-04-17",
            "rf_highthres": CF_OUTPUTS
            / "gen_1_experiments/random_search_exp/RF_highthres_2026-04-21",
            "rf_lowthres": CF_OUTPUTS
            / "gen_1_experiments/random_search_exp/RF_lowthres_2026-04-21",
            "rf_prange_highthres": CF_OUTPUTS
            / "gen_1_experiments/random_search_exp/RF_prange_highthres_2026-04-29",
        },
    },
}


# Usage examples:
# from paths import PATHS
# df = pd.read_csv(PATHS["baseline"]["xgb"][0.5] / "annotated.csv")
# df = pd.read_csv(PATHS["xgb_optimized"][0.1] / "annotated.csv")
# df = pd.read_csv(PATHS["gen1"]["genetic"]["rf_highthres"] / "annotated.csv")
