"""
visualization.py
================

Generate publication-quality visualizations for model evaluation.

Responsibilities
----------------
1. Metric comparison charts
2. ROC curves
3. Precision-Recall curves
4. Confusion matrices
5. Feature importance plots
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import ConfusionMatrixDisplay

from .config import (
    FIGURE_DPI,
    FIGURES_DIR,
)
from .exceptions import VisualizationError
from .logger import get_logger

logger = get_logger(__name__)


class ModelVisualizer:
    """
    Generate and save all project visualizations.
    """

    def __init__(self) -> None:

        FIGURES_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ==========================================================
    # Public API
    # ==========================================================

    def generate_all(
        self,
        results: pd.DataFrame,
        confusion_matrices: dict,
        roc_curves: dict,
        pr_curves: dict,
        trained_models: dict,
    ) -> None:
        """
        Generate every visualization.
        """

        logger.info("Generating visualizations...")

        try:

            self.metric_bar_charts(results)

            self.roc_curve_plot(roc_curves)

            self.precision_recall_plot(pr_curves)

            self.confusion_matrix_grid(confusion_matrices)

            self.feature_importance(trained_models)

            logger.info("All visualizations generated successfully.")

        except Exception as exc:

            logger.exception("Visualization generation failed.")

            raise VisualizationError(
                "Unable to generate visualizations."
            ) from exc

    # ==========================================================
    # Metric Comparison
    # ==========================================================

    def metric_bar_charts(
        self,
        results: pd.DataFrame,
    ) -> None:

        metrics = [

            "Accuracy",

            "Precision",

            "Recall",

            "F1 Score",

            "ROC AUC",

        ]

        for metric in metrics:

            plt.figure(figsize=(10, 5))

            plt.bar(
                results["Model"],
                results[metric],
            )

            plt.title(metric)

            plt.ylabel(metric)

            plt.xticks(rotation=45)

            plt.tight_layout()

            plt.savefig(
                FIGURES_DIR / f"{metric.lower().replace(' ', '_')}.png",
                dpi=FIGURE_DPI,
            )

            plt.close()

    # ==========================================================
    # ROC Curve
    # ==========================================================

    def roc_curve_plot(
        self,
        roc_curves: dict,
    ) -> None:

        plt.figure(figsize=(8, 6))

        for model_name, (fpr, tpr) in roc_curves.items():

            plt.plot(
                fpr,
                tpr,
                label=model_name,
            )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
        )

        plt.xlabel("False Positive Rate")

        plt.ylabel("True Positive Rate")

        plt.title("ROC Curve Comparison")

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / "roc_curves.png",
            dpi=FIGURE_DPI,
        )

        plt.close()

    # ==========================================================
    # Precision Recall Curve
    # ==========================================================

    def precision_recall_plot(
        self,
        pr_curves: dict,
    ) -> None:

        plt.figure(figsize=(8, 6))

        for model_name, (precision, recall) in pr_curves.items():

            plt.plot(
                recall,
                precision,
                label=model_name,
            )

        plt.xlabel("Recall")

        plt.ylabel("Precision")

        plt.title("Precision-Recall Curve Comparison")

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / "precision_recall_curves.png",
            dpi=FIGURE_DPI,
        )

        plt.close()

    # ==========================================================
    # Confusion Matrices
    # ==========================================================

    def confusion_matrix_grid(
        self,
        confusion_matrices: dict,
    ) -> None:

        total = len(confusion_matrices)

        cols = 2

        rows = int(np.ceil(total / cols))

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(12, rows * 5),
        )

        axes = np.array(axes).reshape(-1)

        for ax, (model_name, matrix) in zip(
            axes,
            confusion_matrices.items(),
        ):

            ConfusionMatrixDisplay(
                confusion_matrix=matrix,
            ).plot(
                ax=ax,
                colorbar=False,
            )

            ax.set_title(model_name)

        for ax in axes[total:]:

            ax.axis("off")

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / "confusion_matrices.png",
            dpi=FIGURE_DPI,
        )

        plt.close()

    # ==========================================================
    # Feature Importance
    # ==========================================================

    def feature_importance(
        self,
        trained_models: dict,
    ) -> None:

        for model_name, pipeline in trained_models.items():

            estimator = pipeline.named_steps["model"]

            if not hasattr(estimator, "feature_importances_"):

                continue

            preprocessor = pipeline.named_steps["preprocessor"]

            try:

                feature_names = (
                    preprocessor.get_feature_names_out()
                )

            except Exception:

                feature_names = np.arange(
                    len(estimator.feature_importances_)
                ).astype(str)

            importances = estimator.feature_importances_

            indices = np.argsort(importances)[::-1][:20]

            plt.figure(figsize=(12, 6))

            plt.bar(
                range(len(indices)),
                importances[indices],
            )

            plt.xticks(
                range(len(indices)),
                feature_names[indices],
                rotation=90,
            )

            plt.ylabel("Importance")

            plt.title(
                f"{model_name} Feature Importance"
            )

            plt.tight_layout()

            plt.savefig(
                FIGURES_DIR
                / f"{model_name.lower().replace(' ', '_')}_feature_importance.png",
                dpi=FIGURE_DPI,
            )

            plt.close()