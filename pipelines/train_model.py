import pandas as pd
from catboost import CatBoostClassifier
import pickle
import logging
import mlflow
import mlflow.catboost
import os
from pipelines.modeling import evaluate_model
from pipelines.config import ARTIFACT_DIR

log = logging.getLogger(__name__)

def train_catboost_model(X_train_dict, y_train_list, X_test_dict, y_test_list, hyperparams: dict, ARTIFACT_DIR) -> dict:
    X_train = pd.DataFrame(**X_train_dict)
    X_test = pd.DataFrame(**X_test_dict)
    y_train = pd.Series(y_train_list)
    y_test = pd.Series(y_test_list)

    # Ensure local artifacts directory exists
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    
    # Configure MLflow with proper paths
    mlflow_artifact_root = "/home/sonu/mlop_final/mlflow_data/artifacts"
    os.makedirs(mlflow_artifact_root, exist_ok=True)
    
    # Set MLflow configuration
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Set environment variable for artifact root - this is crucial for avoiding permission issues
    os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = f"file://{mlflow_artifact_root}"
    
    try:
        mlflow.set_experiment("shoppers_mlops_pipeline")
    except Exception as e:
        log.warning(f"Could not set experiment: {e}. Using default experiment.")

    # Start MLflow run
    with mlflow.start_run(run_name="CatBoost_Training") as run:
        log.info("Starting CatBoost model training...")

        # Log hyperparameters
        mlflow.log_params(hyperparams)

        # Train model
        model = CatBoostClassifier(**hyperparams, random_state=42, verbose=0)
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

        # Evaluate model
        log.info("Evaluating model...")
        metrics = evaluate_model("CatBoost", model, X_train, y_train, X_test, y_test, artifact_dir=ARTIFACT_DIR)

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        log.info(f"Model metrics: {metrics}")

        # Save model locally first
        model_path = f"{ARTIFACT_DIR}/catboost_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        log.info(f"Model saved locally at: {model_path}")
        
        # Log model artifact to MLflow
        try:
            if os.path.exists(model_path):
                mlflow.log_artifact(model_path, artifact_path="models")
                log.info("Model artifact logged to MLflow successfully")
            else:
                log.error(f"Model file not found at {model_path}")
        except Exception as e:
            log.error(f"Failed to log model artifact: {e}")

        # Log plots if they exist
        cm_path = f"{ARTIFACT_DIR}/CatBoost_confusion_matrix.png"
        roc_path = f"{ARTIFACT_DIR}/CatBoost_roc_curve.png"
        
        try:
            if os.path.exists(cm_path):
                mlflow.log_artifact(cm_path, artifact_path="plots")
                log.info("Confusion matrix plot logged to MLflow")
            else:
                log.warning(f"Confusion matrix plot not found at {cm_path}")
        except Exception as e:
            log.error(f"Failed to log confusion matrix: {e}")
            
        try:
            if os.path.exists(roc_path):
                mlflow.log_artifact(roc_path, artifact_path="plots")
                log.info("ROC curve plot logged to MLflow")
            else:
                log.warning(f"ROC curve plot not found at {roc_path}")
        except Exception as e:
            log.error(f"Failed to log ROC curve: {e}")

        log.info(f"Model training completed. MLflow run ID: {run.info.run_id}")

    return {
        "model_path": model_path, 
        "metrics": metrics, 
        "mlflow_run_id": run.info.run_id
    }