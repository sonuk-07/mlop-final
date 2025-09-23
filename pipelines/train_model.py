import os
import logging
import pickle
import pandas as pd
from catboost import CatBoostClassifier
import mlflow
import mlflow.catboost

from pipelines.modeling import evaluate_model
from pipelines.config import ARTIFACT_DIR, MLFLOW_TRACKING_URI, MLFLOW_ARTIFACT_ROOT

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_catboost_model(
    X_train_dict, y_train_list, X_test_dict, y_test_list, hyperparams=None, threshold=0.3
):
    """
    Train CatBoost, evaluate, save artifacts, and log everything to MLflow.
    """

    # Convert dicts/lists to DataFrames/Series
    X_train = pd.DataFrame(**X_train_dict)
    X_test = pd.DataFrame(**X_test_dict)
    y_train = pd.Series(y_train_list)
    y_test = pd.Series(y_test_list)

    # Ensure artifact directories exist
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # ------------------------
    # MLflow setup - with error handling
    # ------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # Don't set MLFLOW_DEFAULT_ARTIFACT_ROOT if it causes permission issues
    # os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = f"file://{MLFLOW_ARTIFACT_ROOT}"
    
    try:
        mlflow.set_experiment("shoppers_mlops_pipeline")
    except Exception as e:
        log.warning(f"Could not set MLflow experiment: {e}")

    # ------------------------
    # Compute class weights
    # ------------------------
    num_neg = sum(y_train == 0)
    num_pos = sum(y_train == 1)
    class_weights = [1, max(num_neg / num_pos, 1)]

    with mlflow.start_run(run_name="CatBoost_Training") as run:
        log.info("Training CatBoost model...")

        # Hyperparameters
        if hyperparams is None:
            hyperparams = {
                "iterations": 500,
                "depth": 6,
                "learning_rate": 0.05,
                "l2_leaf_reg": 5
            }
        mlflow.log_params(hyperparams)

        # Train CatBoost
        model = CatBoostClassifier(
            **hyperparams,
            random_state=42,
            verbose=0,
            class_weights=class_weights
        )
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

        # Predict & threshold
        y_prob_test = model.predict_proba(X_test)[:, 1]
        y_pred_test = (y_prob_test > threshold).astype(int)

        # ------------------------
        # Evaluate and save plots
        # ------------------------
        metrics = evaluate_model(
            "CatBoost", model, X_train, y_train, X_test, y_test, artifact_dir=ARTIFACT_DIR
        )
        mlflow.log_metrics({"train_accuracy": metrics["train_accuracy"],
                            "test_accuracy": metrics["test_accuracy"],
                            "roc_auc": metrics["roc_auc"]})

        # ------------------------
        # Save model locally
        # ------------------------
        model_path = os.path.join(ARTIFACT_DIR, "catboost_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        log.info(f"Saved model locally: {model_path}")

        # Log model & artifacts to MLflow with error handling
        try:
            mlflow.log_artifact(model_path, artifact_path="models")
            for plot_key in ["confusion_matrix_plot", "roc_curve_plot", "feature_importance_plot"]:
                plot_path = metrics.get(plot_key)
                if plot_path and os.path.exists(plot_path):
                    mlflow.log_artifact(plot_path, artifact_path="evaluation_plots")

            mlflow.catboost.log_model(model, artifact_path="catboost_model",
                                      registered_model_name="CatBoostClassifierModel")
        except PermissionError as e:
            log.warning(f"Could not log artifacts to MLflow due to permission error: {e}")
            log.info("Model and metrics saved locally, but not logged to MLflow")

        log.info(f"Training completed. MLflow Run ID: {run.info.run_id}")

    return {"model_path": model_path, "metrics": metrics, "mlflow_run_id": run.info.run_id}