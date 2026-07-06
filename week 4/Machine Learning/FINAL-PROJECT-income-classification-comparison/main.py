"""
main.py
=======

Entry point for the Adult Income Classification project.

Workflow
--------
1. Configure logging
2. Load dataset
3. Preprocess data
4. Train models
5. Evaluate models
6. Generate visualizations
"""

from __future__ import annotations

import sys

from src.data_loader import DataLoader
from src.evaluator import ModelEvaluator
from src.exceptions import ProjectError
from src.logger import configure_logger, get_logger
from src.preprocessing import DataPreprocessor
from src.trainer import ModelTrainer
from src.visualization import ModelVisualizer

logger = get_logger(__name__)


def main() -> None:
    """
    Execute the complete machine learning pipeline.
    """

    configure_logger()

    logger.info("=" * 80)
    logger.info("Adult Income Classification Project Started")
    logger.info("=" * 80)

    try:

        # ======================================================
        # Load Dataset
        # ======================================================

        loader = DataLoader()

        df = loader.load()

        # ======================================================
        # Preprocess Dataset
        # ======================================================

        preprocessor = DataPreprocessor(df)

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = preprocessor.prepare_data()

        # ======================================================
        # Train Models
        # ======================================================

        trainer = ModelTrainer(
            scaled_preprocessor=preprocessor.get_scaled_preprocessor(),
            tree_preprocessor=preprocessor.get_tree_preprocessor(),
        )

        trained_models = trainer.train_all(
            X_train,
            y_train,
        )

        # ======================================================
        # Evaluate Models
        # ======================================================

        evaluator = ModelEvaluator()

        results = evaluator.evaluate_all(
            trained_models=trained_models,
            X_test=X_test,
            y_test=y_test,
        )

        # ======================================================
        # Generate Visualizations
        # ======================================================

        visualizer = ModelVisualizer()

        visualizer.generate_all(
            results=results,
            confusion_matrices=evaluator.get_confusion_matrices(),
            roc_curves=evaluator.get_roc_curves(),
            pr_curves=evaluator.get_pr_curves(),
            trained_models=trained_models,
        )

        # ======================================================
        # Display Results
        # ======================================================

        print("\n" + "=" * 80)
        print("PROJECT COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(results)

        logger.info("=" * 80)
        logger.info("Pipeline completed successfully.")
        logger.info("=" * 80)

    except ProjectError as exc:

        logger.exception("Project failed.")

        print(f"\nERROR: {exc}")

        sys.exit(1)

    except Exception as exc:

        logger.exception("Unexpected error.")

        print(f"\nUnexpected Error: {exc}")

        sys.exit(1)


if __name__ == "__main__":

    main()