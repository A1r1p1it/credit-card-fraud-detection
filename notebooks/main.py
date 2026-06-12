from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import sqlite3
from dotenv import load_dotenv
from src.explainer import explain_fraud

load_dotenv()

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


def get_suggested_action(risk_level: str, is_fraud: bool) -> str:
    if is_fraud or risk_level == "HIGH":
        return "Flag for manual review"
    elif risk_level == "MEDIUM":
        return "Monitor transaction closely"
    else:
        return "Approve transaction"


def get_similar_cases(prob: float, n: int = 3) -> list:
    try:
        conn = sqlite3.connect("data/fraud.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT amount, fraud_probability, risk_level, timestamp
            FROM predictions
            WHERE is_fraud = 1
            ORDER BY ABS(fraud_probability - ?) ASC
            LIMIT ?
        """, (prob, n))
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "amount": r[0],
                "fraud_probability": r[1],
                "risk_level": r[2],
                "timestamp": r[3]
            }
            for r in rows
        ]
    except Exception:
        return []


@app.get("/")
def home():
    return {"message": "API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(txn: Transaction):
    try:
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

        risk_level = "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW"
        similar_cases = get_similar_cases(float(probability))

        explanation = None
        if bool(prediction):
            try:
                explanation = explain_fraud(txn.dict(), float(probability))
            except Exception as e:
                err = str(e).lower()
                if "quota" in err or "rate limit" in err or "429" in err or "forbidden" in err:
                    explanation = "Explanation unavailable right now because the LLM quota was exceeded. Please try again in a moment."
                else:
                    explanation = "Explanation temporarily unavailable, but the fraud prediction completed successfully."

        suggested_action = get_suggested_action(risk_level, bool(prediction))

        return {
            "is_fraud": bool(prediction),
            "fraud_probability": round(float(probability), 4),
            "risk_level": risk_level,
            "explanation": explanation,
            "similar_cases": similar_cases,
            "suggested_action": suggested_action
        }

    except Exception as e:
        return {
            "is_fraud": False,
            "fraud_probability": 0.0,
            "risk_level": "ERROR",
            "explanation": f"Prediction failed: {str(e)}",
            "similar_cases": [],
            "suggested_action": "Unable to process transaction"
        }