# pipelines/ingest_data.py

import os
import sys
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from pipelines.config import CSV_PATH, DB_CONN, TABLE_NAME

def ingest_data():
    """
    Ingests data from CSV into MariaDB only.
    """
    print("--- Starting data ingestion process ---")

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV file not found at {CSV_PATH}", file=sys.stderr)
        return

    try:
        df = pd.read_csv(CSV_PATH)
        print(f"Read {df.shape[0]} rows from {CSV_PATH}")
    except Exception as e:
        print(f"ERROR: Failed to read CSV. {e}", file=sys.stderr)
        return

    if df.empty:
        print("WARNING: CSV is empty. Skipping DB ingestion.", file=sys.stderr)
        return

    try:
        engine = create_engine(DB_CONN)
        df.to_sql(TABLE_NAME, con=engine, if_exists="replace", index=False)
        print(f"✅ Ingested {df.shape[0]} rows into table '{TABLE_NAME}'")
    except SQLAlchemyError as e:
        print(f"ERROR: Database ingestion failed. {e}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Unexpected DB ingestion error. {e}", file=sys.stderr)
