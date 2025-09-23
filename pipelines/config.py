import os

PROJECT_DIR = os.getenv("PROJECT_DIR", "/home/sonu/mlop_final")
CSV_PATH = os.path.join(PROJECT_DIR, "data/online_shoppers_intention.csv")
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", os.path.join(PROJECT_DIR, "artifacts"))
os.makedirs(ARTIFACT_DIR, exist_ok=True)

REPORT_DIR = os.path.join(PROJECT_DIR, "reports/evidently")
os.makedirs(REPORT_DIR, exist_ok=True)

# Database configuration
DB_CONN = os.getenv("DB_CONN")  # Read from .env
REFERENCE_TABLE = "shoppers_data"
CURRENT_TABLE = "shoppers_predictions"
TARGET_COL = "Revenue"
TABLE_NAME = "shoppers_data"

# Redis configuration
REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379
RKEY_RAW = "pipeline:raw_data"

# MLflow configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_ARTIFACT_ROOT = f"file://{ARTIFACT_DIR}"

# Email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = os.getenv("SENDER_EMAIL")  # Read from .env
APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")  # Read from .env
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")  # Read from .env

# Other
MAX_ROWS = 1000
