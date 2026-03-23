from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from .visualizations import MultiClassifierVisualizations


class PredictionService:
    def __init__(self, models: dict):
        """
        models: dict[target_name -> trained_model]
        """
        self.models = models
        self.predictions = {}  # cache: target -> y_pred

    def predict_all(self, X):
        """
        Compute predictions for all targets and store them.
        """
        self.predictions = {
            target: model.predict(X) for target, model in self.models.items()
        }
        return self.predictions

    def _ensure_predictions(self, X):
        """
        Internal helper: compute predictions if not already cached.
        """
        if not self.predictions:
            self.predict_all(X)

    def accuracy_all(self, X, y_dict):
        self._ensure_predictions(X)
        return {
            target: accuracy_score(y_dict[target], self.predictions[target])
            for target in self.models
        }

    def roc_auc_all(self, X, y_dict):
        self._ensure_predictions(X)
        results = {}
        for target, model in self.models.items():
            if hasattr(model, "predict_proba") and len(model.classes_) == 2:
                y_true = y_dict[target]
                y_prob = model.predict_proba(X)[:, 1]
                results[target] = roc_auc_score(y_true, y_prob)
            else:
                results[target] = None
        return results

    def classification_reports(self, X, y_dict):
        self._ensure_predictions(X)
        return {
            target: classification_report(y_dict[target], self.predictions[target])
            for target in self.models
        }

    def plot_all_confusion_matrices(self, X, y_dict):
        self._ensure_predictions(X)
        return MultiClassifierVisualizations.plot_all_confusion_matrices(
            self.predictions, y_dict, self.models
        )

    def plot_confusion_matrix(self, X, y_dict, target):
        self._ensure_predictions(X)
        return MultiClassifierVisualizations.plot_confusion_matrix(
            self.predictions, y_dict, target
        )

    def plot_confusion_matrix_grid(self, X, y_dict):
        self._ensure_predictions(X)
        return MultiClassifierVisualizations.plot_confusion_matrix_grid(
            self.predictions, y_dict, self.models
        )

    def summary(self, X, y_dict):
        """
        Return a compact summary of key metrics.
        """
        self._ensure_predictions(X)

        summary = {}
        for target in self.models:
            acc = accuracy_score(y_dict[target], self.predictions[target])

            roc = None
            model = self.models[target]
            if hasattr(model, "predict_proba") and len(model.classes_) == 2:
                y_prob = model.predict_proba(X)[:, 1]
                roc = roc_auc_score(y_dict[target], y_prob)

            summary[target] = {"accuracy": acc, "roc_auc": roc}

        return summary
