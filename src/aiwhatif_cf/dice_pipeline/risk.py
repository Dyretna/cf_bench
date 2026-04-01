"""
risk.py — Risk evaluation utilities for the counterfactual explanation pipeline.

This module contains the RiskEvaluator class, which is responsible for computing
and annotating risk-related metrics for both original instances and generated
counterfactuals.

The goal of this module is to centralize all risk logic in one place so that:
  - new developers can easily understand how risk is computed
  - the pipeline remains clean and modular
  - risk logic can evolve independently of CF generation or distance metrics

Typical workflow:
  1. A model predicts the risk of the original instance.
  2. Counterfactuals are generated elsewhere in the pipeline.
  3. RiskEvaluator annotates the CFs with:
       - predicted risk
       - baseline risk
       - half-target threshold
       - boolean flags indicating whether the CF meets the target


This is a common pattern in counterfactual analysis: we generate CFs,
evaluate their predicted outcomes, and then check whether they achieve
a meaningful improvement relative to the original individual.

"""

from typing import List

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin


class RiskEvaluator:
    """
    Evaluates and annotates risk for original instances and counterfactuals.

    This class encapsulates all logic related to risk scoring in a
    counterfactual explanation pipeline. It is responsible for computing:

      - the predicted risk of the original instance (baseline risk)
      - predicted risks for all generated counterfactuals
      - a half-target threshold (50% reduction from baseline)
      - boolean indicators showing whether each counterfactual meets the target

    The class is designed to be:
      - pipeline-friendly
      - testable
      - extendable (e.g., new risk targets or metrics can be added later)

    Attributes
    ----------
    model : ClassifierMixin
        The trained classification model used to compute predicted risks.
        Must implement `predict_proba`.
    feature_cols : List[str]
        The feature columns the model expects as input. These columns are used
        consistently for both the original instance and all counterfactuals.
    """

    def __init__(
        self,
        model: ClassifierMixin,
        feature_cols: List[str],
        target_factor: float = 0.5,
    ):
        """
        Initialize the RiskEvaluator with a model and feature column names.

        Parameters
        ----------
        model : ClassifierMixin
            Any sklearn-compatible classifier with a `predict_proba` method.
        feature_cols : List[str]
            The feature columns used by the model.
        target_factor : float
            A multiplicative factor that determines the desired target risk level
            relative to the individual's original predicted risk.
        """
        self.model = model
        self.feature_cols = feature_cols
        self.target_factor = target_factor

    def compute_original_risk(self, query_instance: pd.DataFrame) -> float:
        """
        Compute the model's predicted probability for the positive class
        for the given single-row query instance.
        """
        return float(
            self.model.predict_proba(query_instance[self.feature_cols])[:, 1][0]
        )

    def compute_cf_risks(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute predicted risks for all rows in the counterfactual DataFrame.
        """
        return self.model.predict_proba(df[self.feature_cols])[:, 1]

    def annotate(
        self, query_instances: pd.DataFrame, counterfactuals: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add risk-related columns to a counterfactual DataFrame.

        The method returns a *new* DataFrame (copy).

        Parameters
        ----------
        query_instance : pd.DataFrame
            A single-row DataFrame representing the individual we want to improve.
        df : pd.DataFrame
            A DataFrame containing counterfactual examples.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with added columns:
                - original_risk
                - target_risk
                - predicted_risk
                - meets_target_risk
        """
        counterfactuals = counterfactuals.copy()

        # Assign CF IDs if missing
        if "cf_id" not in counterfactuals.columns:
            counterfactuals["cf_id"] = [f"cf_{i}" for i in range(len(counterfactuals))]
        # Assign CF query index if missing
        if "query_index" not in counterfactuals.columns:
            counterfactuals["query_index"] = int(query_instances.index[0])

        # Compute risks
        original_prob = self.compute_original_risk(query_instances)
        target_risk = original_prob * self.target_factor

        counterfactuals["original_risk"] = original_prob
        counterfactuals["target_risk"] = target_risk
        counterfactuals["predicted_risk"] = self.compute_cf_risks(counterfactuals)
        counterfactuals["meets_target_risk"] = (
            counterfactuals["predicted_risk"] <= target_risk
        )

        first_cols = ["query_index", "cf_id"]
        other_cols = [c for c in counterfactuals.columns if c not in first_cols]
        counterfactuals = counterfactuals[first_cols + other_cols]

        return counterfactuals
