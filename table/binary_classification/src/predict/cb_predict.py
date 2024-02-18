from typing import List, Any
import pandas as pd
import numpy as np
import catboost as cb


def predictor(
    X: pd.DataFrame,
    models: List[Any],
    threshold: float = 0.5,
    is_base: bool = False,
):

    categorical_features = [
        col
        for col in X.columns
        if X[col].dtype == "object" or X[col].dtype.name == "category"
    ]
    cb_dataset = cb.Pool(
        X, cat_features=categorical_features, feature_names=list(X.columns)
    )
    pred_probs = np.zeros(len(X))
    for i, model in enumerate(models):
        # 予測
        pred_probs += model.predict(cb_dataset)

    pred_probs /= len(models)
    predictions = (pred_probs > threshold).astype(int)

    if is_base is True:  # stackingのbaseモデル用
        return predictions, pred_probs
    return predictions
