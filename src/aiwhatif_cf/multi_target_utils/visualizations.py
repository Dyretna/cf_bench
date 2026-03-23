# visualizations.py
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


class MultiClassifierVisualizations:
    @staticmethod
    def plot_confusion_matrix(predictions, y_dict, target):
        y_true = y_dict[target]
        y_pred = predictions[target]

        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
        ax.set_title(f"Confusion Matrix: {target}")
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_all_confusion_matrices(predictions, y_dict, models):
        """
        Returns dict[target -> figure]
        """
        figs = {}
        for target in models:
            figs[target] = MultiClassifierVisualizations.plot_confusion_matrix(
                predictions, y_dict, target
            )
        return figs

    @staticmethod
    def plot_confusion_matrix_grid(predictions, y_dict, models):
        n = len(models)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

        if n == 1:
            axes = [axes]

        for ax, target in zip(axes, models.keys()):
            y_true = y_dict[target]
            y_pred = predictions[target]

            ConfusionMatrixDisplay.from_predictions(
                y_true, y_pred, ax=ax, colorbar=False
            )
            ax.set_title(f"{target}")

        fig.suptitle("All Confusion Matrices", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        return fig
