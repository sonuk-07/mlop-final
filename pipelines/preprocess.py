# pipelines/preprocess.py

import pandas as pd
import logging
from typing import Dict

log = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> Dict:
    """
    Cleans and preprocesses the input DataFrame for ML pipelines.

    Args:
        df (pd.DataFrame): DataFrame from previous validation step.

    Returns:
        dict: Cleaned DataFrame serialized as dict (orient='split') for XCom/Redis.

    Raises:
        ValueError: If input DataFrame is None or empty.
    """
    if df is None or df.empty:
        log.error("Input DataFrame is empty or None. Cannot perform cleaning.")
        raise ValueError("Input DataFrame is empty or None")

    log.info("Starting data cleaning and preprocessing.")

    # ---------- 🧹 Cleaning ----------
    df_clean = df.drop_duplicates().copy()
    log.info(f"Removed duplicates. DataFrame shape is now {df_clean.shape}.")

    # ---------- ⚙️ Preprocessing ----------
    # Ordered categorical for Month
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    if "Month" in df_clean.columns:
        df_clean['Month'] = pd.Categorical(df_clean['Month'],
                                           categories=month_order,
                                           ordered=True)
        df_clean = pd.get_dummies(df_clean, columns=['Month'], prefix='Month', drop_first=True)
        log.info("One-hot encoded 'Month' column.")

    if "VisitorType" in df_clean.columns:
        df_clean = pd.get_dummies(df_clean, columns=['VisitorType'], prefix='VisitorType', drop_first=True)
        log.info("One-hot encoded 'VisitorType' column.")

    # Convert boolean columns to int
    for col in ['Weekend', 'Revenue']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(int)
            log.info(f"Converted '{col}' to integer type.")

    log.info("Data cleaning and preprocessing complete.")

    # Return as dict (orient='split') for Redis/XCom
    return df_clean.to_dict(orient='split')
