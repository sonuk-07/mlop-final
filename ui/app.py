import streamlit as st
import requests
import pandas as pd
from sqlalchemy import create_engine
import os
import logging
import redis
import pickle
import hashlib
import json

# -------------------------
# Logging
# -------------------------
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -------------------------
# Redis config
# -------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

# -------------------------
# Streamlit setup
# -------------------------
st.set_page_config(page_title="Online Shoppers Revenue Prediction", layout="wide")
st.title("Online Shoppers Revenue Prediction")

# -------------------------
# Input fields
# -------------------------
Administrative = st.number_input("Administrative", 0, 100, 0)
Administrative_Duration = st.number_input("Administrative_Duration", 0.0, 1000.0, 0.0)
Informational = st.number_input("Informational", 0, 100, 0)
Informational_Duration = st.number_input("Informational_Duration", 0.0, 1000.0, 0.0)
ProductRelated = st.number_input("ProductRelated", 0, 1000, 0)
ProductRelated_Duration = st.number_input("ProductRelated_Duration", 0.0, 2000.0, 0.0)
BounceRates = st.number_input("BounceRates", 0.0, 1.0, 0.0)
ExitRates = st.number_input("ExitRates", 0.0, 1.0, 0.0)
PageValues = st.number_input("PageValues", 0.0, 100.0, 0.0)
SpecialDay = st.number_input("SpecialDay", 0.0, 1.0, 0.0)
Month = st.selectbox("Month", ["Jan","Feb","Mar","Apr","May","June","Jul","Aug","Sep","Oct","Nov","Dec"])
OperatingSystems = st.selectbox("OperatingSystems", [1,2,3,4,5])
Browser = st.selectbox("Browser", [1,2,3,4,5,6,7,8,9])
Region = st.selectbox("Region", [1,2,3,4,5,6,7,8,9])
TrafficType = st.selectbox("TrafficType", [1,2,3,4,5,6,7,8,9])
VisitorType = st.selectbox("VisitorType", ["Returning_Visitor","New_Visitor","Other"])
Weekend = st.selectbox("Weekend", ["TRUE","FALSE"])

# Convert Weekend to boolean
Weekend_bool = True if Weekend == "TRUE" else False

# -------------------------
# Helper functions
# -------------------------
def get_redis_key(payload: dict):
    """Create a unique key for Redis based on input data."""
    payload_str = json.dumps(payload, sort_keys=True)
    return "prediction:" + hashlib.md5(payload_str.encode()).hexdigest()

def save_prediction_to_redis(key: str, prediction_dict: dict, ttl=3600):
    """Save prediction to Redis with optional expiration (seconds)."""
    r.set(key, pickle.dumps(prediction_dict), ex=ttl)

def get_prediction_from_redis(key: str):
    """Retrieve prediction from Redis if available."""
    data = r.get(key)
    if data:
        return pickle.loads(data)
    return None

# -------------------------
# Prediction button
# -------------------------
if st.button("Predict Revenue"):
    payload = {
        "Administrative": Administrative,
        "Administrative_Duration": Administrative_Duration,
        "Informational": Informational,
        "Informational_Duration": Informational_Duration,
        "ProductRelated": ProductRelated,
        "ProductRelated_Duration": ProductRelated_Duration,
        "BounceRates": BounceRates,
        "ExitRates": ExitRates,
        "PageValues": PageValues,
        "SpecialDay": SpecialDay,
        "Month": Month,
        "OperatingSystems": OperatingSystems,
        "Browser": Browser,
        "Region": Region,
        "TrafficType": TrafficType,
        "VisitorType": VisitorType,
        "Weekend": Weekend_bool
    }

    redis_key = get_redis_key(payload)
    cached_result = get_prediction_from_redis(redis_key)

    if cached_result:
        st.info("⚡ Using cached prediction from Redis")
        result = cached_result
    else:
        # Call FastAPI
        try:
            response = requests.post("http://fastapi:8000/predict", json=payload)
            response.raise_for_status()
            result = response.json()
            save_prediction_to_redis(redis_key, result, ttl=3600)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            result = None

    if result:
        st.success(f"Predicted Revenue: {result['prediction']} ✅")
        st.info(f"Probability of Revenue=True: {result['probability']:.2f}")

        # Save input + prediction to MySQL DB
        try:
            db_password = os.getenv("MYSQL_PASSWORD", "Yunachan10")
            db_url = f"mysql+pymysql://sonu:{db_password}@mariadb-columnstore:3306/shoppers_db"
            engine = create_engine(db_url)

            df_to_save = pd.DataFrame([payload])
            df_to_save["Revenue"] = "TRUE" if result['prediction'] == 1 else "FALSE"
            df_to_save.to_sql("shoppers_predictions", con=engine, if_exists='append', index=False)
            st.success("Prediction saved to database ✅")
        except Exception as e:
            st.error(f"Failed to save prediction to DB: {e}")
