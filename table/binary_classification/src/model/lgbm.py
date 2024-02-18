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
CV_MODEL_SVE_DIR = "model/cv"
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
    df_test = pd.read_csv(os.path.join(RAW_DATA_DIR, "test.csv"), index_col=0)
    df_submit = pd.read_csv(
        os.path.join(RAW_DATA_DIR, "sample_submission.csv"),
        index_col=0,
        header=None,
    )

    # データの型を変換する
    df_train = convert_column_type(df_train, CONVERSION_DICT)
    df_test = convert_column_type(df_test, CONVERSION_DICT)

    # 欠損値処理
    missing_cols_train = df_train.columns[df_train.isnull().any()].tolist()
    missing_cols_test = df_test.columns[df_test.isnull().any()].tolist()
    df_train = default_mv_processor(df_train, missing_cols_train)
    df_test = default_mv_processor(df_test, missing_cols_test)

    # 特徴量と正解ラベルに分割
    y_train = df_train[OBJECT_VARIABLE]
    X_train = df_train.drop([OBJECT_VARIABLE], axis=1)
    X_test = df_test

    # エンコーディング
    cols = X_test.select_dtypes(include="object").columns.tolist()
    X_train, X_test = ordinal_encoder(X_train, X_test, cols)

    # 学習
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.01,
        "max_depth": 6,
        "feature_fraction": 0.30,
        "num_iterations": 1000,
        "seed": 42,
        "num_threads": 5,
        "verbose": -1,
    }
    models, f1 = trainer(X_train, y_train, params)

    # モデルを保存
    for i, model in enumerate(models):
        dir_name = f"lgbm"
        if not os.path.exists(os.path.join(CV_MODEL_SVE_DIR, dir_name)):
            os.makedirs(os.path.join(CV_MODEL_SVE_DIR, dir_name))
        model_name = f"lgbm_fold{i+1}_{f1:.5f}.pkl"
        model_path = os.path.join(CV_MODEL_SVE_DIR, dir_name, model_name)
        save_model(model, model_path)

    # 予測
    pred = predictor(X_test, models)

    # 提出用のcsvファイルを作成
    df_submit[1] = pred
    df_submit.to_csv(
        os.path.join(SUBMISSION_DATA_DIR, f"lgbm_{f1:5f}.csv"), header=None
    )


if __name__ == "__main__":
    main()
