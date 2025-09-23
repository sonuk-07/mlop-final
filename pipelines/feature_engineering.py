import pandas as pd
import logging

log = logging.getLogger(__name__)

def feature_engineer_data(df: pd.DataFrame) -> dict:
    """
    Create behavioral features and separate X/y.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None")

    df_fe = df.copy()

    # Create features safely
    df_fe['total_duration'] = df_fe.get('Administrative_Duration',0) + df_fe.get('Informational_Duration',0) + df_fe.get('ProductRelated_Duration',0)
    df_fe['total_pages_visited'] = df_fe.get('Administrative',0) + df_fe.get('Informational',0) + df_fe.get('ProductRelated',0)
    df_fe['ratio_product_duration'] = df_fe.get('ProductRelated_Duration',0) / df_fe['total_duration'].replace(0,1)
    df_fe['bounce_per_page'] = df_fe.get('BounceRates',0) / df_fe['total_pages_visited'].replace(0,1)
    df_fe['returning_visitor_flag'] = df_fe.get('VisitorType_Returning_Visitor',0)
    df_fe['special_day_interaction'] = df_fe.get('SpecialDay',0) * df_fe.get('PageValues',0)

    # Separate X and y
    if 'Revenue' not in df_fe.columns:
        raise ValueError("Target 'Revenue' not found")
    X = df_fe.drop('Revenue', axis=1)
    y = df_fe['Revenue'].astype(int)

    return {'X': X.to_dict(orient='split'), 'y': y.tolist()}
