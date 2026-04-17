"""
Model prediction utilities for performance evaluation.

Handles backend-specific prediction logic (TF2 vs sklearn).
"""


class ModelPredictor:
    """Handles model predictions with backend-specific logic."""

    def __init__(self, backend: str):
        self.backend = backend

    def predict(self, model, df, feature_cols, target_col):
        """
        Compute predictions and extract ground truth.

        Args:
            model: Trained model (Keras or sklearn)
            df: DataFrame with features and target (numeric dtypes expected)
            feature_cols: List of feature column names
            target_col: Target column name

        Returns:
            tuple: (y_true, y_pred) as arrays
        """
        X = df[feature_cols]
        y_true = df[target_col]

        # Get raw predictions
        y_pred = model.predict(X)

        # Apply threshold for Keras models
        if self.backend == "TF2":
            y_pred = (y_pred >= 0.5).astype(int)

        return y_true, y_pred
