"""
data_loader.py
==============

Download, cache, load and validate the Adult Income dataset.

Responsibilities
----------------
1. Download the dataset from the UCI repository.
2. Cache the dataset locally.
3. Load the cached dataset.
4. Clean column names.
5. Validate dataset integrity.
6. Return a pandas DataFrame.

This module intentionally performs NO preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import (
    COLUMN_NAMES,
    DATASET_URL,
    FILE_ENCODING,
    MISSING_VALUE_SYMBOL,
    RAW_DATA_DIR,
    RAW_DATA_FILENAME,
    TARGET_COLUMN,
)
from .exceptions import (
    DatasetLoadError,
    DatasetValidationError,
)
from .logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class DataLoader:
    """
    Adult Income dataset loader.

    Parameters
    ----------
    use_cache : bool, default=True
        Load the cached dataset if available.

    file_name : str, default=RAW_DATA_FILENAME
        Name of the cached dataset.
    """

    use_cache: bool = True
    file_name: str = RAW_DATA_FILENAME

    @property
    def dataset_path(self) -> Path:
        """Return the dataset cache path."""
        return RAW_DATA_DIR / self.file_name

    # ======================================================
    # Public API
    # ======================================================

    def load(self) -> pd.DataFrame:
        """
        Load and validate the dataset.

        Returns
        -------
        pd.DataFrame
            Validated dataset.
        """

        logger.info("Loading dataset...")

        RAW_DATA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        if self.use_cache and self.dataset_path.exists():

            logger.info(
                "Loading dataset from cache: %s",
                self.dataset_path,
            )

            df = self._load_from_cache()

        else:

            logger.info(
                "Downloading dataset from UCI repository..."
            )

            df = self._download()

            self._save_cache(df)

        df = self._clean_columns(df)

        self._validate(df)

        logger.info(
            "Dataset loaded successfully (%d rows, %d columns).",
            df.shape[0],
            df.shape[1],
        )

        return df

    # ======================================================
    # Private Methods
    # ======================================================

    def _load_from_cache(self) -> pd.DataFrame:
        """
        Load the cached dataset.

        Returns
        -------
        pd.DataFrame
        """

        try:

            return pd.read_csv(
                self.dataset_path,
                encoding=FILE_ENCODING,
            )

        except Exception as exc:

            logger.exception(
                "Failed to load cached dataset."
            )

            raise DatasetLoadError(
                f"Unable to load cached dataset: "
                f"{self.dataset_path}"
            ) from exc

    # ------------------------------------------------------

    def _download(self) -> pd.DataFrame:
        """
        Download the dataset from UCI.

        Returns
        -------
        pd.DataFrame
        """

        try:

            return pd.read_csv(
                DATASET_URL,
                names=COLUMN_NAMES,
                skipinitialspace=True,
                na_values=MISSING_VALUE_SYMBOL,
                encoding=FILE_ENCODING,
            )

        except Exception as exc:

            logger.exception(
                "Failed to download dataset."
            )

            raise DatasetLoadError(
                "Unable to download dataset."
            ) from exc

    # ------------------------------------------------------

    def _save_cache(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Save the dataset locally.
        """

        try:

            df.to_csv(
                self.dataset_path,
                index=False,
                encoding=FILE_ENCODING,
            )

            logger.info(
                "Dataset cached at %s",
                self.dataset_path,
            )

        except Exception:

            logger.exception(
                "Unable to cache dataset."
            )

    # ------------------------------------------------------

    @staticmethod
    def _clean_columns(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Standardize column names.

        Returns
        -------
        pd.DataFrame
        """

        df = df.copy()

        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
        )

        return df

    # ------------------------------------------------------

    @staticmethod
    def _validate(
        df: pd.DataFrame,
    ) -> None:
        """
        Validate dataset integrity.

        Raises
        ------
        DatasetValidationError
        """

        if df.empty:

            raise DatasetValidationError(
                "Dataset is empty."
            )

        if df.shape[1] != len(COLUMN_NAMES):

            raise DatasetValidationError(
                f"Expected {len(COLUMN_NAMES)} columns "
                f"but found {df.shape[1]}."
            )

        missing_columns = (
            set(COLUMN_NAMES)
            - set(df.columns)
        )

        if missing_columns:

            raise DatasetValidationError(
                f"Missing columns: "
                f"{sorted(missing_columns)}"
            )

        duplicate_columns = (
            df.columns[df.columns.duplicated()]
            .tolist()
        )

        if duplicate_columns:

            raise DatasetValidationError(
                "Duplicate columns found: "
                f"{duplicate_columns}"
            )

        if TARGET_COLUMN not in df.columns:

            raise DatasetValidationError(
                f"Target column '{TARGET_COLUMN}' "
                "is missing."
            )

        if df.shape[0] < 100:

            raise DatasetValidationError(
                "Dataset appears incomplete."
            )

        duplicate_rows = df.duplicated().sum()

        if duplicate_rows:

            logger.warning(
                "%d duplicate rows detected.",
                duplicate_rows,
            )

        logger.info(
            "Dataset validation completed successfully."
        )