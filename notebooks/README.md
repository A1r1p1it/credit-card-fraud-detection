# 💳 Credit Card Fraud Detection API

A machine learning-powered REST API that detects fraudulent credit card transactions in real-time.

🔴 **Live API**: https://arpitkr-fraud-detection-api.hf.space  
📖 **Swagger Docs**: https://arpitkr-fraud-detection-api.hf.space/docs  
📂 **GitHub**: https://github.com/A1r1p1it/credit-card-fraud-detection

---

## 🚀 Features

- Real-time fraud prediction via REST API
- Returns fraud probability score + risk level (LOW / MEDIUM / HIGH)
- Trained on 284,807 transactions (Kaggle Credit Card Fraud Dataset)
- Handles severe class imbalance (0.17% fraud rate)
- Dockerized and deployed on HuggingFace Spaces

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| PR-AUC | ~0.85+ |
| Precision | High |
| Recall | High |
| Best Model | XGBoost / Random Forest |

---

## 🛠️ Tech Stack

- **ML**: scikit-learn, XGBoost, pandas, numpy
- **API**: FastAPI, Pydantic, Uvicorn
- **Deployment**: Docker, HuggingFace Spaces
- **Tools**: Jupyter Notebook, joblib

---

## 📡 API Usage

### Health Check
```bash
GET https://arpitkr-fraud-detection-api.hf.space/health
```

### Predict
```bash
POST https://arpitkr-fraud-detection-api.hf.space/predict
Content-Type: application/json

{
  "Time": 406.0,
  "V1": -2.31, "V2": 1.95, "V3": -1.60,
  ...
  "Amount": 239.93
}
```

### Response
```json
{
  "is_fraud": true,
  "fraud_probability": 0.9983,
  "risk_level": "HIGH"
}
```

---

## 🏃 Run Locally

```bash
git clone https://github.com/A1r1p1it/credit-card-fraud-detection.git
cd credit-card-fraud-detection/notebooks
pip install -r requirements.txt
uvicorn main:app --reload
```

---

## 📁 Project Structure
