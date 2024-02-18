import os
import random
import sys

sys.path.append(
    "/Users/koroku/program/competition/signate/202402_samples/table/binary_classification"
)

import pandas as pd
import numpy as np

from src.preprocessing.tools import convert_column_type
from src.preprocessing.encoder import ordinal_encoder
from src.preprocessing.missing_value import default_mv_processor
from src.training.lgbm_training import trainer
from src.predict.lgbm_predict import predictor
from src.save.save_model import save_model

RAW_DATA_DIR = "data/raw"
STACKING_MODEL_SVE_DIR = "model/stacking"
OOF_DATA_DIR = "data/oof"
SUBMISSION_DATA_DIR = "data/submission"

OBJECT_VARIABLE = "Survived"

CONVERSION_DICT = {"Pclass": str, "SibSp": str, "Parch": str}


# 再現性を出すために必要な関数
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(42)


def main():

    # データの読み込み
    df_train = pd.read_csv(
        os.path.join(RAW_DATA_DIR, "train.csv"), index_col=0
    )
    df_train = df_train[OBJECT_VARIABLE]
    df_test = pd.DataFrame()
    df_submit = pd.read_csv(
        os.path.join(RAW_DATA_DIR, "sample_submission.csv"),
        index_col=0,
        header=None,
    )

    # baseモデルの出力値を読み込み
    paths_oof_train = [
        os.path.join(OOF_DATA_DIR, "train_lgbm_base.csv"),
        os.path.join(OOF_DATA_DIR, "train_xgb_base.csv"),
        os.path.join(OOF_DATA_DIR, "train_cb_base.csv"),
    ]
    paths_oof_test = [
        os.path.join(OOF_DATA_DIR, "test_lgbm_base.csv"),
        os.path.join(OOF_DATA_DIR, "test_xgb_base.csv"),
        os.path.join(OOF_DATA_DIR, "test_cb_base.csv"),
    ]
    for path_oof_train, path_oof_test in zip(paths_oof_train, paths_oof_test):
        df_oof_train = pd.read_csv(path_oof_train, index_col=0)
        df_oof_test = pd.read_csv(path_oof_test, index_col=0)

        # 元データとoofを連結
        df_train = pd.concat([df_train, df_oof_train], axis=1)
        df_test = pd.concat([df_test, df_oof_test], axis=1)

    # 特徴量と正解ラベルに分割
    y_train = df_train[OBJECT_VARIABLE]
    X_train = df_train.drop([OBJECT_VARIABLE], axis=1)
    X_test = df_test

    # 学習
    pred_probs_test = np.zeros(
        len(X_test)
    )  # 最終的なラベル1の予測確率を入れる
    for base_num in list(range(5)):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.01,
            "max_depth": 6,
            "feature_fraction": 0.30,
            "num_iterations": 1000,
            "seed": 42 + base_num,
            "num_threads": 5,
            "verbose": -1,
        }
        models, f1, oof = trainer(X_train, y_train, params, is_base=True)

        # モデルを保存
        for fold_num, model in enumerate(models):
            dir_name = f"lgbm_stacking_lgbm_xgb_cb_base"
            if not os.path.exists(
                os.path.join(STACKING_MODEL_SVE_DIR, dir_name)
            ):
                os.makedirs(os.path.join(STACKING_MODEL_SVE_DIR, dir_name))
            model_name = f"lgbm_stacking_lgbm_xgb_cb_base{base_num+1}_fold{fold_num+1}_{f1:.5f}.pkl"
            model_path = os.path.join(
                STACKING_MODEL_SVE_DIR, dir_name, model_name
            )
            save_model(model, model_path)

        # 予測
        pred, pred_probs = predictor(X_test, models, is_base=True)
        pred_probs_test += pred_probs

    pred_probs_test /= 5.0
    predictions_test = (pred_probs_test > 0.5).astype(int)

    # 提出用のcsvファイルを作成
    df_submit[1] = predictions_test
    df_submit.to_csv(
        os.path.join(
            SUBMISSION_DATA_DIR,
            f"lgbm_stacking_lgbm_xgb_cb_base_{f1:5f}.csv",
        ),
        header=None,
    )


if __name__ == "__main__":
    main()
