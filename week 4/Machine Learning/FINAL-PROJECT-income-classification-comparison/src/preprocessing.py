"""
preprocessing.py
================

Preprocessing pipeline for the Adult Income Classification project.

Responsibilities
----------------
1. Validate the dataset
2. Split features and target
3. Encode target labels
4. Identify numerical and categorical features
5. Build preprocessing pipelines
6. Provide feature metadata for downstream modules
"""

from __future__ import annotations

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from .config import (
    RANDOM_STATE,
    TARGET_COLUMN,
    TARGET_MAPPING,
    TEST_SIZE,
)
from .exceptions import DataPreprocessingError
from .logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Prepare the Adult Income dataset for machine learning.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Raw dataset.
        """

        self.df = df.copy()

        self.categorical_features: list[str] = []
        self.numerical_features: list[str] = []

    # ==========================================================
    # Public API
    # ==========================================================

    def prepare_data(
        self,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
    ]:
        """
        Execute the preprocessing workflow.

        Returns
        -------
        tuple
            X_train,
            X_test,
            y_train,
            y_test
        """

        logger.info("Starting data preprocessing...")

        self._validate_dataset()

        X, y = self._split_features_target()

        y = self._encode_target(y)

        self._identify_feature_types(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        logger.info(
            "Training samples: %d | Test samples: %d",
            len(X_train),
            len(X_test),
        )

        return (
            X_train,
            X_test,
            y_train,
            y_test,
        )

    # ----------------------------------------------------------

    def get_scaled_preprocessor(
        self,
    ) -> ColumnTransformer:
        """
        Preprocessor for models requiring feature scaling.
        """

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        return ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_pipeline,
                    self.numerical_features,
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    self.categorical_features,
                ),
            ],
            remainder="drop",
        )

    # ----------------------------------------------------------

    def get_tree_preprocessor(
        self,
    ) -> ColumnTransformer:
        """
        Preprocessor for tree-based models.
        """

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                    ),
                ),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        return ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_pipeline,
                    self.numerical_features,
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    self.categorical_features,
                ),
            ],
            remainder="drop",
        )

    # ----------------------------------------------------------

    def get_feature_lists(
        self,
    ) -> tuple[list[str], list[str]]:
        """
        Return numerical and categorical feature names.
        """

        return (
            self.numerical_features,
            self.categorical_features,
        )

    # ==========================================================
    # Private Methods
    # ==========================================================

    def _validate_dataset(self) -> None:
        """
        Validate the input dataset.
        """

        if self.df.empty:

            raise DataPreprocessingError(
                "Input dataset is empty."
            )

        if TARGET_COLUMN not in self.df.columns:

            raise DataPreprocessingError(
                f"Target column '{TARGET_COLUMN}' not found."
            )

    # ----------------------------------------------------------

    def _split_features_target(
        self,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Split features and target.
        """

        X = self.df.drop(columns=TARGET_COLUMN)

        y = self.df[TARGET_COLUMN]

        return X, y

    # ----------------------------------------------------------

    @staticmethod
    def _encode_target(
        y: pd.Series,
    ) -> pd.Series:
        """
        Encode income labels into binary values.
        """

        y = (
            y.astype(str)
            .str.strip()
            .map(TARGET_MAPPING)
        )

        if y.isna().any():

            raise DataPreprocessingError(
                "Unknown target labels detected."
            )

        return y.astype("int64")

    # ----------------------------------------------------------

    def _identify_feature_types(
        self,
        X: pd.DataFrame,
    ) -> None:
        """
        Identify categorical and numerical features.
        """

        self.categorical_features = (
            X.select_dtypes(
                include=["object", "category"],
            )
            .columns
            .tolist()
        )

        self.numerical_features = (
            X.select_dtypes(
                exclude=["object", "category"],
            )
            .columns
            .tolist()
        )

        logger.info(
            "Detected %d numerical and %d categorical features.",
            len(self.numerical_features),
            len(self.categorical_features),
        )