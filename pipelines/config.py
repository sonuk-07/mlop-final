import os

# Data paths
CSV_PATH = "/home/sonu/mlop_final/data/online_shoppers_intention.csv"

# Database configuration
DB_CONN = "mariadb+pymysql://sonu:Yunachan10@127.0.0.1:3306/shoppers_db"
TABLE_NAME = "shoppers_data"

# Redis configuration
REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
RKEY_RAW = "pipeline:raw_data"

# Artifact directory for local storage
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "/home/sonu/mlop_final/artifacts")

# MLflow configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_ARTIFACT_ROOT = "/home/sonu/mlop_final/mlflow_data/artifacts"

# Ensure directories exist
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(MLFLOW_ARTIFACT_ROOT, exist_ok=True)