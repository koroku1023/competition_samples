from typing import List
import numpy as np
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    classification_report,
)


class ModelEvaluator:

    def __init__(
        self,
        pred_probs_train: List[float] | np.ndarray,
        pred_probs_valid: List[float] | np.ndarray,
        y_train: List[int] | np.ndarray,
        y_valid: List[int] | np.ndarray,
        threshold: float = 0.5,
    ):

        self.pred_probs_train = pred_probs_train
        self.pred_probs_valid = pred_probs_valid
        self.y_train = y_train
        self.y_valid = y_valid
        self.threshold = threshold

        self._cal_predictions()

    def _cal_predictions(self):

        self.predictions_train = (
            self.pred_probs_train > self.threshold
        ).astype(int)
        self.predictions_valid = (
            self.pred_probs_valid > self.threshold
        ).astype(int)

    def f1_score(
        self,
        average: str = "binary",
        verbose: bool = True,
        return_result: bool = True,
    ):

        self.f1_train = f1_score(
            self.y_train, self.predictions_train, average=average
        )
        self.f1_valid = f1_score(
            self.y_valid, self.predictions_valid, average=average
        )
        if verbose:
            print(
                f"F1 Score (Training): {self.f1_train:.5f}  (Validation): {self.f1_valid:.5f}"
            )

        if return_result:
            return self.f1_train, self.f1_valid

    def balanced_accuracy_score(
        self,
        verbose: bool = True,
        return_result: bool = True,
    ):

        self.bal_acc_train = balanced_accuracy_score(
            self.y_train, self.predictions_train
        )
        self.bal_acc_valid = balanced_accuracy_score(
            self.y_valid, self.predictions_valid
        )
        if verbose:
            print(
                f"Train F1 Score (Training): {self.bal_acc_train:.5f}  (Validation): {self.bal_acc_valid:.5f}"
            )

        if return_result:
            return self.bal_acc_train, self.bal_acc_valid

    def classification_report(
        self,
        train: bool = False,
    ):

        if train:
            print("Train Classification Report")
            print(
                classification_report(self.y_train_f, self.predictions_train)
            )
        print("Valid Classification Report")
        print(classification_report(self.y_valid, self.predictions_valid))
