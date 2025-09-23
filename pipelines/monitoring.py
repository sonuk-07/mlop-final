import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import redis
import pickle
import logging
from pathlib import Path
from email.message import EmailMessage
import smtplib
import json

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# Pipeline imports
PROJECT_DIR = os.getenv("PROJECT_DIR", "/home/sonu/mlop_final")
sys.path.insert(0, PROJECT_DIR)

from pipelines.ingest_data import ingest_data
from pipelines.validate_data import validate_data
from pipelines.preprocess import clean_data
from pipelines.feature_engineering import feature_engineer_data
from pipelines.prepare_data import prepare_data
from pipelines.hyperparameter import prepare_hyperparameters
from pipelines.train_model import train_catboost_model
from pipelines.config import CSV_PATH, REDIS_HOST, REDIS_PORT, RKEY_RAW, ARTIFACT_DIR

# Evidently imports
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import CatTargetDriftTab, DataDriftTab
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently import ColumnMapping
from sqlalchemy import create_engine

log = logging.getLogger(__name__)

# ---------------------------
# CONFIG
# ---------------------------
DB_URL = "mysql+pymysql://sonu:Yunachan10@127.0.0.1:3306/shoppers_db"
REFERENCE_TABLE = "shoppers_data"
CURRENT_TABLE = "shoppers_predictions"
TARGET_COL = "Revenue"
MAX_ROWS = 1000

# Reports folder structure
REPORT_DIR = Path(PROJECT_DIR) / "reports/evidently"
CONCEPT_DRIFT_DIR = REPORT_DIR / "concept_drift"
DATA_DRIFT_DIR = REPORT_DIR / "data_drift"
CONCEPT_DRIFT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DRIFT_DIR.mkdir(parents=True, exist_ok=True)

# Email config
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "youremail@gmail.com")
APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "your-app-password")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL", "recipient@gmail.com")

engine = create_engine(DB_URL)

# ---------------------------
# Redis helpers
# ---------------------------
def get_redis():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    r.ping()
    return r

def df_to_bytes(df: pd.DataFrame) -> bytes:
    return pickle.dumps(df)

def bytes_to_df(b: bytes) -> pd.DataFrame:
    return pickle.loads(b)

# ---------------------------
# ML pipeline tasks
# ---------------------------
def ingest_and_cache():
    ingest_data()
    df = pd.read_csv(CSV_PATH)
    r = get_redis()
    r.set(RKEY_RAW, df_to_bytes(df))
    print(f"✅ Stored {df.shape[0]} rows into Redis with key '{RKEY_RAW}'")

def validate_from_redis():
    r = get_redis()
    raw_bytes = r.get(RKEY_RAW)
    if raw_bytes is None:
        raise ValueError("❌ No raw data found in Redis.")
    df = bytes_to_df(raw_bytes)
    validated_dict = validate_data(df)
    r.set("pipeline:validated_data", pickle.dumps(validated_dict))
    print("✅ Validated data stored in Redis")
    return validated_dict

def preprocess_from_redis():
    r = get_redis()
    validated_bytes = r.get("pipeline:validated_data")
    if validated_bytes is None:
        raise ValueError("❌ No validated data in Redis.")
    validated_dict = pickle.loads(validated_bytes)
    df_validated = pd.DataFrame(
        validated_dict['data'],
        columns=validated_dict['columns'],
        index=validated_dict['index']
    )
    cleaned_dict = clean_data(df_validated)
    r.set("pipeline:cleaned_data", pickle.dumps(cleaned_dict))
    print("✅ Preprocessed data stored in Redis")
    return cleaned_dict

def feature_engineering_from_redis():
    r = get_redis()
    cleaned_bytes = r.get("pipeline:cleaned_data")
    if cleaned_bytes is None:
        raise ValueError("❌ No cleaned data in Redis.")
    df_cleaned = pd.DataFrame(**pickle.loads(cleaned_bytes))
    fe_dict = feature_engineer_data(df_cleaned)
    r.set("pipeline:fe_data", pickle.dumps(fe_dict))
    print("✅ Feature engineered data stored in Redis")
    return fe_dict

def prepare_data_from_redis():
    r = get_redis()
    fe_bytes = r.get("pipeline:fe_data")
    if fe_bytes is None:
        raise ValueError("❌ No feature engineered data in Redis.")
    fe_dict = pickle.loads(fe_bytes)
    prep_dict = prepare_data(fe_dict['X'], fe_dict['y'], resample=True)
    r.set("pipeline:prepared_data", pickle.dumps(prep_dict))
    print("✅ Prepared data stored in Redis")
    return prep_dict

def hyperparameter_from_redis():
    r = get_redis()
    hyper_dict = prepare_hyperparameters()
    r.set("pipeline:hyperparameters", pickle.dumps(hyper_dict))
    print(f"✅ Hyperparameters stored in Redis: {hyper_dict}")
    return hyper_dict

def train_model_from_redis():
    r = get_redis()
    prep_bytes = r.get("pipeline:prepared_data")
    hyper_bytes = r.get("pipeline:hyperparameters")
    if prep_bytes is None or hyper_bytes is None:
        raise ValueError("❌ Prepared data or hyperparameters not found in Redis.")
    prep_dict = pickle.loads(prep_bytes)
    hyper_dict = pickle.loads(hyper_bytes)

    train_dict = train_catboost_model(
        prep_dict['X_train'], prep_dict['y_train'],
        prep_dict['X_test'], prep_dict['y_test'],
        hyper_dict,
        threshold=0.3
    )
    r.set("pipeline:trained_model", pickle.dumps(train_dict))
    print(f"✅ Model trained, logged in MLflow, stored in Redis (run_id={train_dict['mlflow_run_id']})")
    return {"mlflow_run_id": train_dict['mlflow_run_id']}

# ---------------------------
# Evidently monitoring tasks
# ---------------------------
def send_email_alert(report_path, report_type="Data/Concept Drift"):
    msg = EmailMessage()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    msg["Subject"] = f"{report_type} Detected!"
    msg.set_content(f"{report_type} detected!\nReport: {report_path}")

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)
        print("✅ Email alert sent!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")


def monitor_concept_drift_task():
    ref_df = pd.read_sql(f"SELECT * FROM {REFERENCE_TABLE}", con=engine)
    cur_df = pd.read_sql(f"SELECT * FROM {CURRENT_TABLE}", con=engine)
    if ref_df.empty or cur_df.empty:
        print("Skipping concept drift: empty dataset")
        return

    ref_df = ref_df.sample(n=MAX_ROWS, random_state=42) if len(ref_df) > MAX_ROWS else ref_df
    cur_df = cur_df.sample(n=MAX_ROWS, random_state=42) if len(cur_df) > MAX_ROWS else cur_df

    feature_cols = [col for col in ref_df.columns if col != TARGET_COL]
    ref_data = ref_df[feature_cols + [TARGET_COL]]
    cur_data = cur_df[feature_cols + ["prediction"]].copy()
    cur_data.rename(columns={"prediction": TARGET_COL}, inplace=True)

    column_mapping = ColumnMapping()
    column_mapping.target = TARGET_COL

    dashboard = Dashboard(tabs=[CatTargetDriftTab()])
    dashboard.calculate(reference_data=ref_data, current_data=cur_data, column_mapping=column_mapping)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    report_path = CONCEPT_DRIFT_DIR / f"concept_drift_report_{ts}.html"
    dashboard.save(report_path)
    print(f"✅ Concept drift report saved at {report_path}")
    send_email_alert(report_path, report_type="Concept Drift")


def monitor_data_drift_task():
    ref_df = pd.read_sql(f"SELECT * FROM {REFERENCE_TABLE}", con=engine)
    cur_df = pd.read_sql(f"SELECT * FROM {CURRENT_TABLE}", con=engine)
    if ref_df.empty or cur_df.empty:
        print("Skipping data drift: empty dataset")
        return

    ref_df = ref_df.sample(n=MAX_ROWS, random_state=42) if len(ref_df) > MAX_ROWS else ref_df
    cur_df = cur_df.sample(n=MAX_ROWS, random_state=42) if len(cur_df) > MAX_ROWS else cur_df

    feature_cols = [col for col in ref_df.columns if col != TARGET_COL]
    ref_data = ref_df[feature_cols]
    cur_data = cur_df[feature_cols]

    dashboard = Dashboard(tabs=[DataDriftTab()])
    dashboard.calculate(ref_data, cur_data)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    report_path = DATA_DRIFT_DIR / f"data_drift_report_{ts}.html"
    dashboard.save(report_path)
    print(f"✅ Data drift report saved at {report_path}")

    profile = Profile(sections=[DataDriftProfileSection()])
    profile.calculate(ref_data, cur_data)
    profile_data = json.loads(profile.json())
    metrics_dict = profile_data.get("data_drift", {}).get("data", {}).get("metrics", {})

    drift_detected = any(metrics.get("drift_detected", False) for metrics in metrics_dict.values() if isinstance(metrics, dict))
    if drift_detected:
        print("⚠️ Data drift detected! Sending email alert...")
        send_email_alert(report_path, report_type="Data Drift")
    else:
        print("✅ No data drift detected.")

# ---------------------------
# DAG definition
# ---------------------------
default_args = {
    "owner": "mlops_user",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="shoppers_full_pipeline",
    default_args=default_args,
    description="Full ML pipeline with Docker, Redis, MariaDB, and Evidently monitoring",
    schedule="@daily",
    start_date=datetime(2025, 9, 1),
    catchup=False,
    tags=["mlops", "pipeline", "redis", "monitoring"],
) as dag:

    # ----- Bash Operators -----
    install_requirements = BashOperator(
        task_id="install_requirements",
        bash_command=f"pip install -r {PROJECT_DIR}/requirements.txt",
    )

    docker_compose_up = BashOperator(
        task_id='docker_compose_up',
        bash_command=f"cd {PROJECT_DIR} && docker compose up -d"
    )

    create_db_user = BashOperator(
        task_id="create_db_user",
        bash_command=f"""
        docker exec -i mcs_container mariadb -uroot -proot -e "
        CREATE DATABASE IF NOT EXISTS shoppers_db;
        CREATE USER IF NOT EXISTS 'sonu'@'%' IDENTIFIED BY 'Yunachan10';
        GRANT ALL PRIVILEGES ON shoppers_db.* TO 'sonu'@'%';
        FLUSH PRIVILEGES;"
        """
    )

    # ----- Python Operators -----
    ingest_task = PythonOperator(task_id="ingest_data_task", python_callable=ingest_and_cache)
    validate_task = PythonOperator(task_id="validate_data_task", python_callable=validate_from_redis)
    preprocess_task = PythonOperator(task_id="preprocess_data_task", python_callable=preprocess_from_redis)
    feature_task = PythonOperator(task_id="feature_engineering_task", python_callable=feature_engineering_from_redis)
    prepare_task = PythonOperator(task_id="prepare_data_task", python_callable=prepare_data_from_redis)
    hyper_task = PythonOperator(task_id="hyperparameter_task", python_callable=hyperparameter_from_redis)
    train_task = PythonOperator(task_id="train_model_task", python_callable=train_model_from_redis)

    concept_drift_task = PythonOperator(task_id="concept_drift_task", python_callable=monitor_concept_drift_task)
    data_drift_task = PythonOperator(task_id="data_drift_task", python_callable=monitor_data_drift_task)

    # ----- DAG Dependencies -----
    install_requirements >> docker_compose_up >> create_db_user
    create_db_user >> ingest_task >> validate_task >> preprocess_task
    preprocess_task >> feature_task >> prepare_task >> hyper_task >> train_task
    train_task >> [concept_drift_task, data_drift_task]
