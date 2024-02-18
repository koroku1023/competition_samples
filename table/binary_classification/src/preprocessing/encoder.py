from typing import List
import pandas as pd
import category_encoders as ce


def ordinal_encoder(
    X_train: pd.DataFrame, X_test: pd.DataFrame, cols: List[str]
):

    oe = ce.OrdinalEncoder(cols=cols, return_df=False)
    X_train[cols] = oe.fit_transform(X_train[cols])
    X_test[cols] = oe.transform(X_test[cols])

    return X_train, X_test


def count_encoder(
    X_train: pd.DataFrame, X_test: pd.DataFrame, cols: List[str]
):

    coe = ce.CountEncoder(cols=cols, return_df=False)
    X_train[cols] = coe.fit_transform(X_train[cols])
    X_test[cols] = coe.transform(X_test[cols])

    return X_train, X_test


def one_hot_encoder(
    X_train: pd.DataFrame, X_test: pd.DataFrame, cols: List[str]
):

    ohe = ce.OneHotEncoder(
        cols=cols,
        use_cat_names=True,
        return_df=True,
    )

    df_encoded_train = ohe.fit_transform(X_train[cols])
    df_encoded_test = ohe.transform(X_test[cols])

    X_train.drop(cols, axis=1, inplace=True)
    X_test.drop(cols, axis=1, inplace=True)

    X_train = pd.concat([X_train, df_encoded_train], axis=1)
    X_test = pd.concat([X_test, df_encoded_test], axis=1)

    return X_train, X_test
