"""
models.py
=========

Factory responsible for creating all machine learning models used
throughout the project.

Each model is returned together with its metadata, allowing the trainer
to remain completely independent from model-specific implementation
details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import (
    LOGISTIC_MAX_ITER,
    N_JOBS,
    RANDOM_STATE,
    RF_ESTIMATORS,
    XGB_ESTIMATORS,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
)
from .exceptions import ModelCreationError
from .logger import get_logger

logger = get_logger(__name__)


# ==========================================================
# Model Configuration
# ==========================================================

@dataclass(slots=True, frozen=True)
class ModelConfig:
    """
    Configuration container for a machine learning model.
    """

    estimator: Any

    requires_scaling: bool

    tune: bool = False

    param_grid: dict[str, list[Any]] = field(
        default_factory=dict
    )


# ==========================================================
# Factory
# ==========================================================

class ModelFactory:
    """
    Factory class responsible for creating all machine learning models.
    """

    def get_models(self) -> dict[str, ModelConfig]:
        """
        Return all supported models.

        Returns
        -------
        dict[str, ModelConfig]
        """

        try:

            models = {

                # ==============================================
                # Logistic Regression
                # ==============================================

                "Logistic Regression": ModelConfig(

                    estimator=LogisticRegression(

                        max_iter=LOGISTIC_MAX_ITER,

                        random_state=RANDOM_STATE,

                    ),

                    requires_scaling=True,

                    tune=True,

                    param_grid={

                        "model__C": [

                            0.01,

                            0.1,

                            1,

                            10,

                        ],

                        "model__penalty": [

                            "l2",

                        ],

                    },

                ),

                # ==============================================
                # K-Nearest Neighbors
                # ==============================================

                "KNN": ModelConfig(

                    estimator=KNeighborsClassifier(),

                    requires_scaling=True,

                    tune=True,

                    param_grid={

                        "model__n_neighbors": [

                            3,

                            5,

                            7,

                            9,

                        ],

                        "model__weights": [

                            "uniform",

                            "distance",

                        ],

                    },

                ),

                # ==============================================
                # Support Vector Machine
                # ==============================================

                "SVM": ModelConfig(

                    estimator=SVC(

                        probability=True,

                        random_state=RANDOM_STATE,

                    ),

                    requires_scaling=True,

                    tune=False,

                ),

                # ==============================================
                # Gaussian Naive Bayes
                # ==============================================

                "Naive Bayes": ModelConfig(

                    estimator=GaussianNB(),

                    requires_scaling=True,

                    tune=False,

                ),

                # ==============================================
                # Decision Tree
                # ==============================================

                "Decision Tree": ModelConfig(

                    estimator=DecisionTreeClassifier(

                        random_state=RANDOM_STATE,

                    ),

                    requires_scaling=False,

                    tune=True,

                    param_grid={

                        "model__max_depth": [

                            None,

                            5,

                            10,

                            20,

                        ],

                        "model__min_samples_split": [

                            2,

                            5,

                            10,

                        ],

                    },

                ),

                # ==============================================
                # Random Forest
                # ==============================================

                "Random Forest": ModelConfig(

                    estimator=RandomForestClassifier(

                        n_estimators=RF_ESTIMATORS,

                        random_state=RANDOM_STATE,

                        n_jobs=N_JOBS,

                    ),

                    requires_scaling=False,

                    tune=True,

                    param_grid={

                        "model__n_estimators": [

                            100,

                            200,

                            300,

                        ],

                        "model__max_depth": [

                            None,

                            10,

                            20,

                        ],

                        "model__min_samples_split": [

                            2,

                            5,

                        ],

                    },

                ),

                # ==============================================
                # XGBoost
                # ==============================================

                "XGBoost": ModelConfig(

                    estimator=XGBClassifier(

                        n_estimators=XGB_ESTIMATORS,

                        learning_rate=XGB_LEARNING_RATE,

                        max_depth=XGB_MAX_DEPTH,

                        random_state=RANDOM_STATE,

                        eval_metric="logloss",

                        n_jobs=N_JOBS,

                        use_label_encoder=False,

                    ),

                    requires_scaling=False,

                    tune=True,

                    param_grid={

                        "model__n_estimators": [

                            100,

                            200,

                            300,

                        ],

                        "model__learning_rate": [

                            0.05,

                            0.1,

                            0.2,

                        ],

                        "model__max_depth": [

                            3,

                            6,

                            9,

                        ],

                    },

                ),

            }

            logger.info(
                "Initialized %d machine learning models.",
                len(models),
            )

            return models

        except Exception as exc:

            logger.exception(
                "Failed to create machine learning models."
            )

            raise ModelCreationError(
                "Unable to initialize machine learning models."
            ) from exc