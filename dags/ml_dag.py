import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import redis
import pickle
import logging

PROJECT_DIR = os.getenv("PROJECT_DIR", "/home/sonu/mlop_final")
sys.path.insert(0, PROJECT_DIR)

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

from pipelines.ingest_data import ingest_data
from pipelines.validate_data import validate_data
from pipelines.preprocess import clean_data
from pipelines.feature_engineering import feature_engineer_data
from pipelines.prepare_data import prepare_data
from pipelines.hyperparameter import prepare_hyperparameters
from pipelines.train_model import train_catboost_model
from pipelines.config import CSV_PATH, REDIS_HOST, REDIS_PORT, RKEY_RAW, ARTIFACT_DIR

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
# ML Pipeline Python tasks
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
    prep_dict = prepare_data(fe_dict['X'], fe_dict['y'])
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
        ARTIFACT_DIR=ARTIFACT_DIR
    )

    r.set("pipeline:trained_model", pickle.dumps(train_dict))
    print(f"✅ Model trained, logged in MLflow, and stored in Redis (run_id={train_dict['mlflow_run_id']})")
    return train_dict


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
    description="Full ML pipeline with Docker Compose, Redis, and MariaDB",
    schedule="@daily",
    start_date=datetime(2025, 9, 1),
    catchup=False,
    tags=["mlops", "pipeline", "redis"],
) as dag:

    # ---------------------------
    # 1️⃣ Install requirements
    # ---------------------------
    install_requirements = BashOperator(
        task_id="install_requirements",
        bash_command=f"pip install -r {PROJECT_DIR}/requirements.txt",
    )

    # ---------------------------
    # 2️⃣ Docker Compose up
    # ---------------------------
    docker_compose_up = BashOperator(
        task_id='docker_compose_up',
        bash_command=f"""
        cd {PROJECT_DIR}

        start_container() {{
            local name=$1
            if [ "$(docker ps -q -f name=$name)" ]; then
                echo "$name is already running. Skipping..."
            else
                if docker compose ps | grep -q $name; then
                    echo "Starting $name..."
                    docker compose up -d $name || echo "⚠️ Failed to start $name, but continuing..."
                else
                    echo "⚠️ $name service not defined in docker-compose.yml. Skipping..."
                fi
            fi
        }}

        start_container redis
        start_container mcs_container
        start_container mlflow
        start_container fastapi
        start_container streamlit

        """
    )

    # ---------------------------
    # 3️⃣ Create MariaDB DB & user
    # ---------------------------
    create_db_user = BashOperator(
        task_id="create_db_user",
        bash_command="""
        echo "Waiting for MariaDB to be ready..."
        until docker exec mcs_container mariadb -uroot -proot &> /dev/null; do
            echo "MariaDB not ready yet... retrying in 3s"
            sleep 3
        done
        echo "MariaDB is ready. Creating database and user..."
        docker exec -i mcs_container mariadb -uroot -proot -e "
        CREATE DATABASE IF NOT EXISTS shoppers_db;
        CREATE USER IF NOT EXISTS 'sonu'@'%' IDENTIFIED BY 'Yunachan10';
        GRANT ALL PRIVILEGES ON shoppers_db.* TO 'sonu'@'%';
        CREATE USER IF NOT EXISTS 'sonu'@'localhost' IDENTIFIED BY 'Yunachan10';
        GRANT ALL PRIVILEGES ON shoppers_db.* TO 'sonu'@'localhost';
        FLUSH PRIVILEGES;"
        """
    )

    # ---------------------------
    # 4️⃣ ML pipeline tasks
    # ---------------------------
    ingest_task = PythonOperator(task_id="ingest_data_task", python_callable=ingest_and_cache)
    validate_task = PythonOperator(task_id="validate_data_task", python_callable=validate_from_redis)
    preprocess_task = PythonOperator(task_id="preprocess_data_task", python_callable=preprocess_from_redis)
    feature_task = PythonOperator(task_id="feature_engineering_task", python_callable=feature_engineering_from_redis)
    prepare_task = PythonOperator(task_id="prepare_data_task", python_callable=prepare_data_from_redis)
    hyper_task = PythonOperator(task_id="hyperparameter_task", python_callable=hyperparameter_from_redis)
    train_task = PythonOperator(task_id="train_model_task", python_callable=train_model_from_redis)

    # ---------------------------
    # DAG dependencies
    # ---------------------------
    install_requirements >> docker_compose_up >> create_db_user
    create_db_user >> ingest_task >> validate_task >> preprocess_task
    preprocess_task >> feature_task >> prepare_task >> hyper_task >> train_task
