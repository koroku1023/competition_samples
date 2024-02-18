from typing import List, Dict, Any
import sys

sys.path.append(
    "/Users/koroku/program/competition/signate/202402_samples/table/binary_classification"
)

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

from src.eval.eval import ModelEvaluator


def trainer(
    X: pd.DataFrame,
    y: pd.Series,
    params: Dict[str, Any],
    n_splits: int = 5,
    is_base: bool = False,
):

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores_train = []
    balanced_accuracies_train = []
    scores_valid = []
    balanced_accuracies_valid = []
    models = []
    oof = np.zeros(len(y))

    for i, (train_index, valid_index) in enumerate(kf.split(X, y)):
        print("--------------------------")
        print(f"Fold {i+1}")
        print("--------------------------")

        # データセットの作成
        X_train_f, X_valid_f = X.iloc[train_index], X.iloc[valid_index]
        y_train_f, y_valid_f = y.iloc[train_index], y.iloc[valid_index]

        xgb_train = xgb.DMatrix(
            X_train_f, y_train_f, feature_names=list(X_train_f.columns)
        )
        xgb_valid = xgb.DMatrix(
            X_valid_f, y_valid_f, feature_names=list(X_valid_f.columns)
        )

        # 学習
        model = xgb.train(
            params=params,
            dtrain=xgb_train,
            evals=[(xgb_train, "train"), (xgb_valid, "eval")],
            verbose_eval=False,
            early_stopping_rounds=50,
        )
        models.append(model)

        # 予測
        pred_probs_train = model.predict(xgb_train)
        pred_probs_valid = model.predict(xgb_valid)

        # stacking用にoofに予測確率を入れる
        if is_base is True:
            oof[valid_index] = pred_probs_valid

        # 評価
        evaluator = ModelEvaluator(
            pred_probs_train, pred_probs_valid, y_train_f, y_valid_f
        )
        f1_train, f1_valid = evaluator.f1_score()
        bal_acc_train, bal_acc_valid = evaluator.balanced_accuracy_score()
        evaluator.classification_report(train=False)

        # 評価値を保存
        scores_train.append(f1_train)
        scores_valid.append(f1_valid)
        balanced_accuracies_train.append(bal_acc_train)
        balanced_accuracies_valid.append(bal_acc_valid)

    # 最終評価
    f1_mean_train, f1_mean_valid = np.mean(scores_train), np.mean(scores_valid)
    bal_acc_mean_train, bal_acc_mean_valid = np.mean(
        balanced_accuracies_train
    ), np.mean(balanced_accuracies_valid)
    print(
        f"Average F1 Score (Training): {f1_mean_train}  (Validation): {f1_mean_valid}"
    )
    print(
        f"Average Balanced Accuracy (Training): {bal_acc_mean_train}  (Validation): {bal_acc_mean_valid}"
    )

    if is_base is True:
        return models, f1_mean_valid, oof
    else:
        return models, f1_mean_valid
