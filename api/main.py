from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import redis
import os

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
MODEL_KEY = "pipeline:trained_model"

app = FastAPI()

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

# ---------------------------
# Input schema
# ---------------------------
class UserInput(BaseModel):
    Administrative: int
    Administrative_Duration: float
    Informational: int
    Informational_Duration: float
    ProductRelated: int
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

# ---------------------------
# Helper functions
# ---------------------------
def preprocess_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # Boolean to int
    df['Weekend'] = df['Weekend'].astype(int)

    # One-hot encode Month
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df['Month'] = pd.Categorical(df['Month'], categories=month_order, ordered=True)
    df = pd.get_dummies(df, columns=['Month'], prefix='Month', drop_first=True)

    # One-hot encode VisitorType
    df = pd.get_dummies(df, columns=['VisitorType'], prefix='VisitorType', drop_first=True)

    # Feature engineering (same as training)
    df['total_duration'] = df[['Administrative_Duration','Informational_Duration','ProductRelated_Duration']].sum(axis=1)
    df['total_pages_visited'] = df[['Administrative','Informational','ProductRelated']].sum(axis=1)
    df['ratio_product_duration'] = df['ProductRelated_Duration'] / (df['total_duration'] + 1e-5)
    df['bounce_per_page'] = df['BounceRates'] / (df['total_pages_visited'] + 1e-5)

    return df

def load_model():
    model_bytes = r.get(MODEL_KEY)
    if model_bytes is None:
        raise ValueError("Trained model not found in Redis.")
    model_dict = pickle.loads(model_bytes)
    with open(model_dict['model_path'], "rb") as f:
        model = pickle.load(f)
    return model

# ---------------------------
# API endpoints
# ---------------------------
@app.post("/predict")
def predict(input_data: UserInput):
    df = preprocess_input(input_data.dict())

    model = load_model()
    proba = model.predict_proba(df)[:,1][0]  # probability for Revenue=1

    # Optionally adjust threshold for better Revenue=1 recall
    threshold = 0.3
    prediction = int(proba >= threshold)

    return {
        "prediction": prediction,
        "probability": float(proba)
    }
