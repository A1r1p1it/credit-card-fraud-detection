from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Fraud Detection API")

model = joblib.load("Best_model.pkl")
scaler = joblib.load("Scaler.pkl")

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
    Amount: float

@app.get("/")
def home():
    return {"message": "API is running"}
 
@app.get("/health")
def health(): 
    return {"status": "ok"}

@app.post("/predict")
def predict(txn: Transaction):
    features = np.array([[
        txn.Time, txn.V1, txn.V2, txn.V3, txn.V4, txn.V5,
        txn.V6, txn.V7, txn.V8, txn.V9, txn.V10, txn.V11,
        txn.V12, txn.V13, txn.V14, txn.V15, txn.V16, txn.V17,
        txn.V18, txn.V19, txn.V20, txn.V21, txn.V22, txn.V23,
        txn.V24, txn.V25, txn.V26, txn.V27, txn.V28, txn.Amount
    ]])

    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    return {
        "is_fraud": bool(prediction),
        "fraud_probability": round(float(probability), 4),
        "risk_level": "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW"
    }
