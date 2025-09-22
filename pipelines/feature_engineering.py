# pipelines/feature_engineering.py
import pandas as pd
import logging
from typing import Dict

log = logging.getLogger(__name__)

def feature_engineer_data(df: pd.DataFrame) -> Dict:
    """
    Performs feature engineering and separates features (X) and target (y).
    Returns dict with 'X' and 'y' serialized for Redis storage.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None.")

    df_fe = df.copy()

    # ----- Create new features -----
    if set(['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration']).issubset(df_fe.columns):
        df_fe['total_duration'] = df_fe[['Administrative_Duration','Informational_Duration','ProductRelated_Duration']].sum(axis=1)
    if set(['Administrative','Informational','ProductRelated']).issubset(df_fe.columns):
        df_fe['total_pages_visited'] = df_fe[['Administrative','Informational','ProductRelated']].sum(axis=1)

    # ----- Split Features & Target -----
    if "Revenue" not in df_fe.columns:
        raise ValueError("Target column 'Revenue' not found in dataframe.")

    X = pd.get_dummies(df_fe.drop('Revenue', axis=1), drop_first=True)
    y = df_fe['Revenue'].astype(int)

    return {
        "X": X.to_dict(orient='split'),
        "y": y.tolist()
    }
