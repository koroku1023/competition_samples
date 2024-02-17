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
