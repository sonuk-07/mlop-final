import pandas as pd
import logging

log = logging.getLogger(__name__)

def clean_data(df: pd.DataFrame) -> dict:
    """
    Clean and preprocess input DataFrame for ML pipelines.
    Returns dict (orient='split') for Redis/XCom.
    """
    if df is None or df.empty:
        log.error("Input DataFrame is empty or None")
        raise ValueError("Input DataFrame is empty or None")

    df_clean = df.drop_duplicates().copy()

    # Handle Month column
    month_order = ['Jan','Feb','Mar','Apr','May','June','Jul','Aug','Sep','Oct','Nov','Dec']
    if 'Month' in df_clean.columns:
        df_clean['Month'] = pd.Categorical(df_clean['Month'], categories=month_order, ordered=True)
        df_clean = pd.get_dummies(df_clean, columns=['Month'], prefix='Month', drop_first=True)

    # VisitorType
    if 'VisitorType' in df_clean.columns:
        df_clean = pd.get_dummies(df_clean, columns=['VisitorType'], prefix='VisitorType', drop_first=True)
    else:
        df_clean['VisitorType_Returning_Visitor'] = 0  # safe default

    # Convert boolean to int
    for col in ['Weekend','Revenue']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(int)

    return df_clean.to_dict(orient='split')
