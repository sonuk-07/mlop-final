# pipelines/train_model.py
import pandas as pd
from catboost import CatBoostClassifier
import pickle
import logging
from pipelines.modeling import evaluate_model

log = logging.getLogger(__name__)

def train_catboost_model(X_train_dict, y_train_list, X_test_dict, y_test_list, hyperparams: dict, artifact_dir="/opt/airflow/artifacts") -> dict:
    """
    Train CatBoost model, evaluate, and save model + metrics.
    """
    X_train = pd.DataFrame(**X_train_dict)
    X_test = pd.DataFrame(**X_test_dict)
    y_train = pd.Series(y_train_list)
    y_test = pd.Series(y_test_list)

    model = CatBoostClassifier(**hyperparams, random_state=42, verbose=0)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

    metrics = evaluate_model("CatBoost", model, X_train, y_train, X_test, y_test, artifact_dir=artifact_dir)

    # Save model
    model_path = f"{artifact_dir}/catboost_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    log.info(f"Model trained and saved at {model_path}")
    return {"model_path": model_path, "metrics": metrics}
