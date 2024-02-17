from typing import Dict
import pandas as pd


def convert_column_type(df: pd.DataFrame, conversion_dict: Dict):

    for column, new_type in conversion_dict.items():
        df[column] = df[column].astype(new_type)

    return df
