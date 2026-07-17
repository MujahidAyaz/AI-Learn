"""
============================================================
Early Stopping
============================================================

Stops training when validation accuracy
does not improve for several epochs.

============================================================
"""


class EarlyStopping:
    """
    Early stopping utility.
    """

    def __init__(
        self,
        patience=5,
        min_delta=0.0,
    ):

        self.patience = patience

        self.min_delta = min_delta

        self.best_score = None

        self.counter = 0

        self.early_stop = False

    def __call__(self, validation_accuracy):

        if self.best_score is None:

            self.best_score = validation_accuracy

            return

        if validation_accuracy > self.best_score + self.min_delta:

            self.best_score = validation_accuracy

            self.counter = 0

        else:

            self.counter += 1

            print(
                f"EarlyStopping Counter: "
                f"{self.counter}/{self.patience}"
            )

            if self.counter >= self.patience:

                self.early_stop = True