
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="XGBoost MLOps API for detecting fraudulent credit card transactions",
    version="1.0.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "xgboost_fraud_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

FEATURES = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0)


@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API is running",
        "status": "healthy"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict")
def predict(transaction: Transaction):
    try:
        data = pd.DataFrame(
            [[getattr(transaction, feature) for feature in FEATURES]],
            columns=FEATURES
        )

        # Apply the same preprocessing used during training
        data[["Time", "Amount"]] = scaler.transform(
            data[["Time", "Amount"]]
        )

        prediction = int(model.predict(data)[0])
        probability = float(model.predict_proba(data)[0][1])

        return {
            "prediction": prediction,
            "fraud_prediction": bool(prediction),
            "fraud_probability": round(probability, 6),
            "classification": "Fraud" if prediction == 1 else "Legitimate"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
