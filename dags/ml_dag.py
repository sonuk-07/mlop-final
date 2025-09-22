import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import redis
import pickle
import logging

from pipelines.ingest_data import ingest_data
from pipelines.validate_data import validate_data
from pipelines.preprocess import clean_data
from pipelines.feature_engineering import feature_engineer_data
from pipelines.prepare_data import prepare_data
from pipelines.hyperparameter import prepare_hyperparameters
from pipelines.train_model import train_catboost_model

from pipelines.config import CSV_PATH, REDIS_HOST, REDIS_PORT, RKEY_RAW

log = logging.getLogger(__name__)

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
# 1️⃣ Ingest data and cache
# ---------------------------
def ingest_and_cache():
    ingest_data()  # Insert into MariaDB
    df = pd.read_csv(CSV_PATH)
    r = get_redis()
    r.set(RKEY_RAW, df_to_bytes(df))
    print(f"✅ Stored {df.shape[0]} rows into Redis with key '{RKEY_RAW}'")

# ---------------------------
# 2️⃣ Validate data
# ---------------------------
def validate_from_redis():
    r = get_redis()
    raw_bytes = r.get(RKEY_RAW)
    if raw_bytes is None:
        raise ValueError("❌ No raw data found in Redis.")
    df = bytes_to_df(raw_bytes)
    validated_dict = validate_data(df)  # returns dict

    # Save validated data to Redis
    r.set("pipeline:validated_data", pickle.dumps(validated_dict))
    print("✅ Validated data stored in Redis")
    return validated_dict

# ---------------------------
# 3️⃣ Preprocess data
# ---------------------------
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

# ---------------------------
# 4️⃣ Feature engineering
# ---------------------------
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

# ---------------------------
# 5️⃣ Data preparation (train/test split + SMOTE)
# ---------------------------
def prepare_data_from_redis():
    r = get_redis()
    fe_bytes = r.get("pipeline:fe_data")
    if fe_bytes is None:
        raise ValueError("❌ No feature engineered data in Redis.")
    fe_dict = pickle.loads(fe_bytes)

    prep_dict = prepare_data(fe_dict['X'], fe_dict['y'])
    r.set("pipeline:prepared_data", pickle.dumps(prep_dict))
    print("✅ Prepared data stored in Redis")
    return prep_dict

# ---------------------------
# 6️⃣ Hyperparameter tuning
# ---------------------------
def hyperparameter_from_redis():
    r = get_redis()
    hyper_dict = prepare_hyperparameters()
    r.set("pipeline:hyperparameters", pickle.dumps(hyper_dict))
    print(f"✅ Hyperparameters stored in Redis: {hyper_dict}")
    return hyper_dict

# ---------------------------
# 7️⃣ Train model
# ---------------------------
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
        hyper_dict
    )

    r.set("pipeline:trained_model", pickle.dumps(train_dict))
    print(f"✅ Model trained and stored in Redis")
    return train_dict

# ---------------------------
# DAG definition
# ---------------------------
default_args = {
    "owner": "sonu",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="shoppers_full_pipeline",
    default_args=default_args,
    description="Full ML pipeline using Redis for intermediate storage",
    schedule="@daily",
    start_date=datetime(2025, 9, 1),
    catchup=False,
    tags=["mlops", "pipeline", "redis"],
) as dag:

    ingest_task = PythonOperator(task_id="ingest_data_task", python_callable=ingest_and_cache)
    validate_task = PythonOperator(task_id="validate_data_task", python_callable=validate_from_redis)
    preprocess_task = PythonOperator(task_id="preprocess_data_task", python_callable=preprocess_from_redis)
    feature_task = PythonOperator(task_id="feature_engineering_task", python_callable=feature_engineering_from_redis)
    prepare_task = PythonOperator(task_id="prepare_data_task", python_callable=prepare_data_from_redis)
    hyper_task = PythonOperator(task_id="hyperparameter_task", python_callable=hyperparameter_from_redis)
    train_task = PythonOperator(task_id="train_model_task", python_callable=train_model_from_redis)

    # DAG dependencies
    ingest_task >> validate_task >> preprocess_task >> feature_task >> prepare_task >> hyper_task >> train_task
