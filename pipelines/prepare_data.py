# pipelines/prepare_data.py
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import logging

log = logging.getLogger(__name__)

def prepare_data(X_dict: dict, y_list: list, test_size=0.3, random_state=42, resample=True) -> dict:
    """
    Split dataset into train/test sets and optionally apply SMOTE.
    Returns Redis-friendly dicts.
    """
    X = pd.DataFrame(**X_dict)
    y = pd.Series(y_list)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    if resample:
        sm = SMOTE(random_state=random_state)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        log.info(f"SMOTE applied. New training class distribution: {Counter(y_train)}")

    return {
        "X_train": X_train.to_dict(orient='split'),
        "X_test": X_test.to_dict(orient='split'),
        "y_train": y_train.tolist(),
        "y_test": y_test.tolist()
    }
