"""
risk.py — Risk evaluation utilities for the counterfactual explanation pipeline.

This module contains the RiskEvaluator classes, which are responsible for
computing and annotating risk-related metrics for both original instances and
generated counterfactuals.

The goal of this module is to centralize all risk logic in one place so that:
  - new developers can easily understand how risk is computed
  - the pipeline remains clean and modular
  - risk logic can evolve independently of CF generation or distance metrics

Typical workflow:
  1. A model predicts the risk of the original instance.
  2. Counterfactuals are generated elsewhere in the pipeline.
  3. A RiskEvaluator annotates the CFs with:
       - predicted risk
       - baseline risk
       - target threshold
       - boolean flags indicating whether the CF meets the target
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseRiskEvaluator(ABC):
    """
    Base class for risk evaluators used in counterfactual analysis.

    Subclasses must implement:
        - _predict_proba(self, X): returns a 1D array of probabilities
    """

    def __init__(
        self,
        model: object,
        feature_cols: List[str],
        target_factor: float = 0.5,
    ) -> None:
        self.model = model
        self.feature_cols = feature_cols
        self.target_factor = target_factor

    def compute_original_risk(self, query_instance: pd.DataFrame) -> float:
        """Compute predicted risk for a single-row query instance."""
        return float(self._predict_proba(query_instance)[0])

    def compute_cf_risks(self, df: pd.DataFrame) -> np.ndarray:
        """Compute predicted risks for all counterfactual rows."""
        return self._predict_proba(df)

    def annotate(
        self,
        query_instances: pd.DataFrame,
        counterfactuals: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Annotate counterfactuals with:
            - original_risk
            - target_risk
            - predicted_risk
            - meets_target_risk
        """
        counterfactuals = counterfactuals.copy()

        if "cf_id" not in counterfactuals.columns:
            counterfactuals["cf_id"] = [f"cf_{i}" for i in range(len(counterfactuals))]

        if "query_index" not in counterfactuals.columns:
            counterfactuals["query_index"] = int(query_instances.index[0])

        original_prob: float = self.compute_original_risk(query_instances)
        target_risk: float = original_prob * self.target_factor

        counterfactuals["original_risk"] = original_prob
        counterfactuals["target_risk"] = target_risk
        counterfactuals["predicted_risk"] = self.compute_cf_risks(counterfactuals)
        counterfactuals["meets_target_risk"] = (
            counterfactuals["predicted_risk"] <= target_risk
        )

        first_cols = ["query_index", "cf_id"]
        other_cols = [c for c in counterfactuals.columns if c not in first_cols]
        return counterfactuals[first_cols + other_cols]

    @abstractmethod
    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return a 1D array of predicted probabilities."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Sklearn evaluator
# ---------------------------------------------------------------------------


class SKlearnRiskEvaluator(BaseRiskEvaluator):
    """Risk evaluator for sklearn classifiers using predict_proba."""

    def __init__(
        self,
        model: ClassifierMixin,
        feature_cols: List[str],
        target_factor: float = 0.5,
    ) -> None:
        super().__init__(model, feature_cols, target_factor)

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.feature_cols])[:, 1]


# ---------------------------------------------------------------------------
# TensorFlow / Keras evaluator
# ---------------------------------------------------------------------------


class TF2RiskEvaluator(BaseRiskEvaluator):
    """Risk evaluator for TensorFlow/Keras models using model.predict."""

    def __init__(
        self,
        model: object,  # tf.keras.Model, but we avoid hard dependency
        feature_cols: List[str],
        target_factor: float = 0.5,
    ) -> None:
        super().__init__(model, feature_cols, target_factor)

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X[self.feature_cols], verbose=0)

        # Convert shape (n, 1) → (n,)
        if preds.ndim == 2 and preds.shape[1] == 1:
            preds = preds[:, 0]

        return preds


def build_risk_evaluator(
    backend: str,
    model: object,
    feature_cols: List[str],
    target_factor: float = 0.5,
) -> BaseRiskEvaluator:
    """
    Factory function that returns the appropriate RiskEvaluator subclass
    based on the model backend.

    Parameters
    ----------
    backend : str
        Either "sklearn" or "TF2".
    model : object
        The trained predictive model.
    feature_cols : List[str]
        Feature columns used by the model.
    target_factor : float
        Multiplicative factor defining the target risk threshold.

    Returns
    -------
    BaseRiskEvaluator
        An instance of SKlearnRiskEvaluator or TF2RiskEvaluator.
    """
    backend = backend.lower()

    if backend == "sklearn":
        return SKlearnRiskEvaluator(
            model=model,
            feature_cols=feature_cols,
            target_factor=target_factor,
        )

    if backend in ("tf2", "tensorflow"):
        return TF2RiskEvaluator(
            model=model,
            feature_cols=feature_cols,
            target_factor=target_factor,
        )

    raise ValueError(f"Unsupported backend for RiskEvaluator: {backend}")
