"""
trainer.py
==========

Train, tune and persist all machine learning models.

Responsibilities
----------------
1. Build preprocessing pipelines
2. Train baseline models
3. Perform hyperparameter tuning
4. Save trained models
5. Record training time
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .config import (
    CV_FOLDS,
    GRID_SEARCH_VERBOSE,
    MODEL_EXTENSION,
    MODELS_DIR,
    N_JOBS,
)
from .exceptions import (
    ModelSaveError,
    ModelTrainingError,
)
from .logger import get_logger
from .models import ModelConfig, ModelFactory

logger = get_logger(__name__)


class ModelTrainer:
    """
    Train and save all machine learning models.
    """

    def __init__(
        self,
        scaled_preprocessor,
        tree_preprocessor,
    ) -> None:

        self.scaled_preprocessor = scaled_preprocessor
        self.tree_preprocessor = tree_preprocessor

        self.factory = ModelFactory()

        self.trained_models: dict[str, Pipeline] = {}

        self.training_times: dict[str, float] = {}

        MODELS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ==========================================================
    # Public API
    # ==========================================================

    def train_all(
        self,
        X_train,
        y_train,
    ) -> dict[str, Pipeline]:
        """
        Train every configured model.
        """

        logger.info("Starting model training...")

        models = self.factory.get_models()

        for model_name, config in models.items():

            logger.info("=" * 70)
            logger.info("Training %s", model_name)

            start = time.perf_counter()

            try:

                pipeline = self._build_pipeline(config)

                if config.tune and config.param_grid:

                    pipeline = self._perform_grid_search(
                        pipeline=pipeline,
                        config=config,
                        X_train=X_train,
                        y_train=y_train,
                    )

                else:

                    pipeline.fit(
                        X_train,
                        y_train,
                    )

                elapsed = time.perf_counter() - start

                self.training_times[model_name] = elapsed

                self.trained_models[model_name] = pipeline

                self._save_model(
                    model_name,
                    pipeline,
                )

                logger.info(
                    "%s completed in %.2f seconds.",
                    model_name,
                    elapsed,
                )

            except Exception as exc:

                logger.exception(
                    "Training failed for %s.",
                    model_name,
                )

                raise ModelTrainingError(
                    f"Training failed for '{model_name}'."
                ) from exc

        logger.info("All models trained successfully.")

        return self.trained_models

    # ==========================================================
    # Private Methods
    # ==========================================================

    def _build_pipeline(
        self,
        config: ModelConfig,
    ) -> Pipeline:
        """
        Build preprocessing + estimator pipeline.
        """

        preprocessor = (
            self.scaled_preprocessor
            if config.requires_scaling
            else self.tree_preprocessor
        )

        return Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    config.estimator,
                ),
            ]
        )

    # ----------------------------------------------------------

    def _perform_grid_search(
        self,
        pipeline: Pipeline,
        config: ModelConfig,
        X_train,
        y_train,
    ) -> Pipeline:
        """
        Perform GridSearchCV and return the best estimator.
        """

        logger.info("Running GridSearchCV...")

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=config.param_grid,
            scoring="accuracy",
            cv=CV_FOLDS,
            n_jobs=N_JOBS,
            verbose=GRID_SEARCH_VERBOSE,
            refit=True,
        )

        search.fit(
            X_train,
            y_train,
        )

        logger.info(
            "Best Score: %.4f",
            search.best_score_,
        )

        logger.info(
            "Best Parameters: %s",
            search.best_params_,
        )

        return search.best_estimator_

    # ----------------------------------------------------------

    def _save_model(
        self,
        model_name: str,
        model: Any,
    ) -> None:
        """
        Save a trained model.
        """

        filename = (
            model_name.lower()
            .replace(" ", "_")
            + MODEL_EXTENSION
        )

        path: Path = MODELS_DIR / filename

        try:

            joblib.dump(
                model,
                path,
            )

            logger.info(
                "Model saved to %s",
                path,
            )

        except Exception as exc:

            logger.exception(
                "Failed to save model '%s'.",
                model_name,
            )

            raise ModelSaveError(
                f"Unable to save model '{model_name}'."
            ) from exc

    # ==========================================================
    # Getters
    # ==========================================================

    def get_training_times(
        self,
    ) -> dict[str, float]:
        """
        Return training times for all models.
        """

        return self.training_times

    def get_trained_models(
        self,
    ) -> dict[str, Pipeline]:
        """
        Return trained models.
        """

        return self.trained_models