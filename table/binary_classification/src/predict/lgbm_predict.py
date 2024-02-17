from typing import List, Any
import pandas as pd
import numpy as np


def predictor(
    X: pd.DataFrame,
    models: List[Any],
    threshold: float = 0.5,
    is_base: bool = False,
    cols: List[str] = None,
):

    pred_probs = np.zeros(len(X))
    for i, model in enumerate(models):
        # 予測
        if cols:
            X = X[cols[i]]
        pred_probs += model.predict(X)

    pred_probs /= len(models)
    predictions = (pred_probs > threshold).astype(int)

    if is_base is True:  # stackingのbaseモデル用
        return predictions, pred_probs
    return predictions
