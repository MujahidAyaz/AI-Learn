"""
evaluator.py
============

Evaluate trained machine learning models.

Responsibilities
----------------
1. Evaluate all trained models
2. Calculate classification metrics
3. Store confusion matrices
4. Store ROC curves
5. Store Precision-Recall curves
6. Export comparison table
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import (
    FILE_ENCODING,
    TABLES_DIR,
)
from .exceptions import ModelEvaluationError
from .logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluate trained machine learning models.
    """

    def __init__(self) -> None:

        self.results: list[dict[str, Any]] = []

        self.confusion_matrices: dict[str, np.ndarray] = {}

        self.roc_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        self.pr_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # ==========================================================
    # Public API
    # ==========================================================

    def evaluate_all(
        self,
        trained_models: dict[str, Any],
        X_test,
        y_test,
    ) -> pd.DataFrame:
        """
        Evaluate all trained models.

        Parameters
        ----------
        trained_models : dict
            Dictionary of trained pipelines.

        X_test : pd.DataFrame

        y_test : pd.Series

        Returns
        -------
        pd.DataFrame
            Sorted evaluation results.
        """

        logger.info("Starting model evaluation...")

        self.results.clear()

        try:

            for model_name, model in trained_models.items():

                logger.info(
                    "Evaluating %s...",
                    model_name,
                )

                metrics = self._evaluate_single_model(
                    model_name=model_name,
                    model=model,
                    X_test=X_test,
                    y_test=y_test,
                )

                self.results.append(metrics)

            results_df = (
                pd.DataFrame(self.results)
                .sort_values(
                    by="F1 Score",
                    ascending=False,
                )
                .reset_index(drop=True)
            )

            self._save_results(results_df)

            logger.info(
                "Evaluation completed successfully."
            )

            return results_df

        except Exception as exc:

            logger.exception(
                "Model evaluation failed."
            )

            raise ModelEvaluationError(
                "Unable to evaluate trained models."
            ) from exc

    # ==========================================================
    # Private Methods
    # ==========================================================

    def _evaluate_single_model(
        self,
        model_name: str,
        model,
        X_test,
        y_test,
    ) -> dict[str, Any]:
        """
        Evaluate a single trained model.
        """

        y_pred = model.predict(X_test)

        y_prob = self._predict_probability(
            model,
            X_test,
        )

        self.confusion_matrices[model_name] = (
            confusion_matrix(
                y_test,
                y_pred,
            )
        )

        fpr, tpr, _ = roc_curve(
            y_test,
            y_prob,
        )

        self.roc_curves[model_name] = (
            fpr,
            tpr,
        )

        precision, recall, _ = (
            precision_recall_curve(
                y_test,
                y_prob,
            )
        )

        self.pr_curves[model_name] = (
            precision,
            recall,
        )

        return {

            "Model": model_name,

            "Accuracy": accuracy_score(
                y_test,
                y_pred,
            ),

            "Balanced Accuracy": balanced_accuracy_score(
                y_test,
                y_pred,
            ),

            "Precision": precision_score(
                y_test,
                y_pred,
                zero_division=0,
            ),

            "Recall": recall_score(
                y_test,
                y_pred,
                zero_division=0,
            ),

            "F1 Score": f1_score(
                y_test,
                y_pred,
                zero_division=0,
            ),

            "ROC AUC": roc_auc_score(
                y_test,
                y_prob,
            ),

            "PR AUC": average_precision_score(
                y_test,
                y_prob,
            ),

            "MCC": matthews_corrcoef(
                y_test,
                y_pred,
            ),

        }

    # ----------------------------------------------------------

    @staticmethod
    def _predict_probability(
        model,
        X_test,
    ) -> np.ndarray:
        """
        Return prediction probabilities.

        Falls back to decision_function when
        predict_proba is unavailable.
        """

        if hasattr(model, "predict_proba"):

            return model.predict_proba(
                X_test
            )[:, 1]

        scores = model.decision_function(
            X_test
        )

        scores = (
            scores - scores.min()
        ) / (
            scores.max() - scores.min()
        )

        return scores

    # ----------------------------------------------------------

    @staticmethod
    def _save_results(
        results_df: pd.DataFrame,
    ) -> None:
        """
        Save evaluation results.
        """

        TABLES_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        results_df.to_csv(
            TABLES_DIR / "model_comparison.csv",
            index=False,
            encoding=FILE_ENCODING,
        )

    # ==========================================================
    # Getters
    # ==========================================================

    def get_confusion_matrices(
        self,
    ) -> dict[str, np.ndarray]:

        return self.confusion_matrices

    def get_roc_curves(
        self,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:

        return self.roc_curves

    def get_pr_curves(
        self,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:

        return self.pr_curves