from typing import List
import pandas as pd


def default_mv_processor(df: pd.DataFrame, cols: List[int]) -> pd.DataFrame:

    for col in cols:
        if df[col].dtype == "object":
            mode_value = (
                df[col].mode().iloc[0] if not df[col].mode().empty else ""
            )
            df[col] = df[col].fillna(mode_value)
        elif df[col].dtype == "int" or df[col].dtype == "float":
            mean_value = df[col].mean()
            df[col] = df[col].fillna(mean_value)
    return df
