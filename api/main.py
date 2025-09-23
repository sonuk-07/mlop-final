from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
from catboost import CatBoostClassifier, Pool
import logging

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
# Model path & global variables
# --------------------------
MODEL_PATH = "/opt/artifacts/catboost_model.pkl"
catboost_model: CatBoostClassifier = None
feature_order = []  # Will be loaded from model

# --------------------------
# Load CatBoost model at startup
# --------------------------
import pickle

@app.on_event("startup")
def load_model():
    global catboost_model, feature_order
    if not os.path.exists(MODEL_PATH):
        log.error(f"Model file not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as f:
        catboost_model = pickle.load(f)

    feature_order = catboost_model.feature_names_

    log.info("CatBoost model loaded successfully.")
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
# Helper: Preprocessing / Feature Engineering
# --------------------------
def preprocess_input(df: pd.DataFrame, feature_order: list) -> pd.DataFrame:
    """
    Apply feature engineering to raw input, one-hot encode, and align with model feature order.
    """
    # Create engineered features
    df['total_duration'] = df['Administrative_Duration'] + df['Informational_Duration'] + df['ProductRelated_Duration']
    df['total_pages_visited'] = df['Administrative'] + df['Informational'] + df['ProductRelated']

    # Ensure categorical columns are string
    df['Month'] = df['Month'].astype(str)
    df['VisitorType'] = df['VisitorType'].astype(str)
    df['Weekend'] = df['Weekend'].astype(int)



    # One-hot encode Month and VisitorType
    df = pd.get_dummies(df, columns=['Month', 'VisitorType'], drop_first=False)

    # Fill missing columns expected by model
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match model
    df = df[feature_order]
    return df

# --------------------------
# Predict endpoint
# --------------------------
@app.post("/predict")
def predict_revenue(data: ShopperData):
    global catboost_model, feature_order
    if catboost_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data.dict()])

        # Preprocess input
        df_preprocessed = preprocess_input(df, feature_order)

        # Predict
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