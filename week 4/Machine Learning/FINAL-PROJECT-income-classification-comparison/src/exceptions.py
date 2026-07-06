"""
exceptions.py
=============

Custom exception hierarchy for the Adult Income Classification project.

Using project-specific exceptions improves readability, debugging,
and enables precise exception handling throughout the pipeline.
"""

from __future__ import annotations


class ProjectError(Exception):
    """
    Base exception for the entire project.

    All custom exceptions should inherit from this class.
    """


# ==========================================================
# Configuration
# ==========================================================

class ConfigurationError(ProjectError):
    """
    Raised when the project configuration is invalid.
    """


# ==========================================================
# Data
# ==========================================================

class DatasetLoadError(ProjectError):
    """
    Raised when the dataset cannot be downloaded or loaded.
    """


class DatasetValidationError(ProjectError):
    """
    Raised when dataset validation fails.
    """


class DataPreprocessingError(ProjectError):
    """
    Raised when data preprocessing fails.
    """


# ==========================================================
# Models
# ==========================================================

class ModelCreationError(ProjectError):
    """
    Raised when model creation fails.
    """


class ModelTrainingError(ProjectError):
    """
    Raised when model training fails.
    """


class ModelEvaluationError(ProjectError):
    """
    Raised when model evaluation fails.
    """


# ==========================================================
# Persistence
# ==========================================================

class ModelSaveError(ProjectError):
    """
    Raised when saving a trained model fails.
    """


class ModelLoadError(ProjectError):
    """
    Raised when loading a saved model fails.
    """


# ==========================================================
# Visualization
# ==========================================================

class VisualizationError(ProjectError):
    """
    Raised when generating visualizations fails.
    """


# ==========================================================
# Pipeline
# ==========================================================

class PipelineError(ProjectError):
    """
    Raised when the end-to-end machine learning pipeline fails.
    """