from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import redis
from catboost import CatBoostClassifier, Pool
import logging
import os

# --------------------------
# Logging
# --------------------------
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --------------------------
# FastAPI app
# --------------------------
app = FastAPI(title="Shoppers Prediction API")

# --------------------------
# Redis Config
# --------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
MODEL_KEY = "pipeline:trained_model"

r = None
catboost_model: CatBoostClassifier = None
feature_order = []

# --------------------------
# Connect Redis & Load Model
# --------------------------
@app.on_event("startup")
def load_model_from_redis():
    global r, catboost_model, feature_order

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    r.ping()
    log.info("✅ Connected to Redis")

    model_bytes = r.get(MODEL_KEY)
    if model_bytes is None:
        raise FileNotFoundError(f"Model not available in Redis with key {MODEL_KEY}")

    train_dict = pickle.loads(model_bytes)
    log.info(f"Keys in train_dict: {list(train_dict.keys())}")

    # Try to detect model key automatically
    if "model" in train_dict:
        catboost_model = train_dict["model"]
    elif "catboost_model" in train_dict:
        catboost_model = train_dict["catboost_model"]
    else:
        raise KeyError(
            f"Could not find model in Redis object. Keys available: {list(train_dict.keys())}"
        )

    feature_order = catboost_model.feature_names_
    log.info("✅ CatBoost model loaded successfully from Redis")
    log.info(f"Model expects {len(feature_order)} features.")

# --------------------------
# Input schema
# --------------------------
class ShopperData(BaseModel):
    Administrative: float
    Administrative_Duration: float
    Informational: float
    Informational_Duration: float
    ProductRelated: float
    ProductRelated_Duration: float
    BounceRates: float
    ExitRates: float
    PageValues: float
    SpecialDay: float
    Month: str
    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int
    VisitorType: str
    Weekend: bool

# --------------------------
# Preprocess input
# --------------------------
def preprocess_input(df: pd.DataFrame, feature_order: list) -> pd.DataFrame:
    df['total_duration'] = df['Administrative_Duration'] + df['Informational_Duration'] + df['ProductRelated_Duration']
    df['total_pages_visited'] = df['Administrative'] + df['Informational'] + df['ProductRelated']

    # Convert categorical
    df['Month'] = df['Month'].astype(str)
    df['VisitorType'] = df['VisitorType'].astype(str)
    df['Weekend'] = df['Weekend'].astype(int)

    # One-hot encode
    df = pd.get_dummies(df, columns=['Month', 'VisitorType'], drop_first=False)

    # Add missing columns
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns
    df = df[feature_order]
    return df

# --------------------------
# Prediction endpoint
# --------------------------
@app.post("/predict")
def predict_revenue(data: ShopperData):
    global catboost_model, feature_order
    if catboost_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        df = pd.DataFrame([data.dict()])
        df_preprocessed = preprocess_input(df, feature_order)

        pool = Pool(df_preprocessed)
        prediction = catboost_model.predict(pool)
        prediction_proba = catboost_model.predict_proba(pool)[:, 1]

        return {
            "prediction": int(prediction[0]),
            "probability": float(prediction_proba[0])
        }

    except Exception as e:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
