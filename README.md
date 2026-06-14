---
title: Fraud Detection UI
emoji: 🔍
colorFrom: red
colorTo: red
sdk: streamlit
app_file: app.py
pinned: false
---

# Credit Card Fraud Detection

**Live Demo**: [Fraud Detection UI](https://arpitkr-fraud-detection-ui.hf.space)  
**Live API**: [FastAPI Docs](https://fraud-api-kzt3.onrender.com/docs)

> If GitHub does not render the notebook properly, open the nbviewer link below for a clean static view.

>**Notebook View**: [fraud_clean.ipynb on nbviewer](https://nbviewer.org/github/A1r1p1it/credit-card-fraud-detection/blob/main/notebooks/fraud_clean.ipynb)

Binary classification system to detect fraudulent credit card transactions in a highly imbalanced dataset, with an interactive Streamlit UI, FastAPI backend, RAG-powered fraud explanation layer, and an **Agent Pipeline** with automated risk assessment and similar case retrieval.

## Problem Statement

Credit card fraud detection is a classic **imbalanced classification** problem. In this dataset, only 0.17% of transactions are fraudulent, which means a naive model that predicts every transaction as non-fraud can still achieve about 99.8% accuracy while catching no actual fraud.

Because of this, accuracy is not a meaningful metric here. The goal is to maximize fraud detection while minimizing false alarms that would hurt customer trust and operations.

## Dataset

- **Source**: Kaggle Credit Card Fraud Detection
- **Size**: 284,807 transactions
- **Features**: 30 total features
  - `V1`–`V28`: PCA-transformed anonymized features
  - `Time`
  - `Amount`
- **Target**: `Class`
  - `0` = Non-Fraud
  - `1` = Fraud
- **Fraud Cases**: 492
- **Fraud Rate**: 0.17%

## Approach

### 1. Data Preprocessing

- Stratified train-test split (80/20) to preserve class distribution
- StandardScaler for feature normalization
- Careful handling of class imbalance during training and evaluation

### 2. Models Compared

#### Logistic Regression
- Baseline model
- Used `class_weight='balanced'`
- High recall but poor precision
- Too many false positives for real-world usage

#### Random Forest
- 100 trees, `max_depth=5`
- Used `class_weight='balanced'`
- Better precision than Logistic Regression
- Still produced a high false alarm rate

#### XGBoost
- Best-performing model
- `n_estimators=200`
- `learning_rate=0.5`
- `scale_pos_weight=25`
- Strongest precision-recall balance for production-style fraud detection

### 3. Evaluation Strategy

Since the dataset is extremely imbalanced, the project uses:
- Precision
- Recall
- F1-score
- Precision-Recall AUC
- Confusion Matrix

Accuracy was intentionally not used as the primary metric because it is misleading in this setting.

## Results

| Model | Precision | Recall | F1-Score | PR-AUC | Trade-off |
|-------|-----------|--------|----------|--------|-----------|
| Logistic Regression | 0.06 | 0.92 | 0.11 | 0.764 | Catches most fraud but creates too many false alarms |
| Random Forest | 0.36 | 0.89 | 0.51 | 0.659 | Better balance, still too many false positives |
| XGBoost | 0.91 | 0.84 | 0.87 | 0.878 | Best production-style trade-off |

## Key Insight

XGBoost achieved the best balance between **precision** and **recall**, making it the most suitable model for deployment. It catches the majority of fraudulent transactions while keeping false positives low enough for a practical fraud detection workflow.

## Feature Importance

Top features identified by the tree-based models:

1. `V14`
2. `V10`
3. `V12`
4. `V17`
5. `V4`

These features are especially important in distinguishing fraud from normal transactions, even though the raw variables are anonymized through PCA.

## SQL Analysis

The dataset was also analyzed using SQLite for business and fraud-pattern insights:

- Overall fraud rate: **0.17%**
- High-value transactions (>$200) show roughly **2x higher fraud rate**
- Fraudulent transactions have a higher average amount than legitimate ones
- Peak fraud hours occur around **2AM–3AM**

## AI-Powered Features

### 1. LLM Fraud Explanation
When a transaction is predicted as fraud, the system generates a human-readable explanation using **LLaMA 3.1 8B via Groq**.

### 2. Context-Aware AI Chat

The application includes a conversational AI assistant that helps users understand fraud predictions, fraud patterns, and machine learning concepts.

Features include:

- Access to the latest prediction context (fraud probability, risk level, key feature values, and explanations)
- Retrieval-Augmented Generation (RAG) using the fraud knowledge base
- Multi-turn conversations with session-based chat history
- Grounded responses that combine model outputs with retrieved fraud knowledge

Example questions:

- "Why was this transaction flagged?"
- "What makes V14 suspicious?"
- "What fraud pattern does this transaction resemble?"
- "Why is PR-AUC more useful than accuracy for fraud detection?"

### 3. RAG-Based Fraud Knowledge Layer
The app includes a Retrieval-Augmented Generation pipeline so explanations are grounded in a curated fraud knowledge base instead of relying only on a direct LLM response.

### 4. Agent Pipeline
Every prediction now runs through a multi-step agent pipeline:

### 5. Natural Language Transaction Analysis

Users can describe a transaction in plain English instead of manually entering model features.

The system uses an LLM-powered feature extraction workflow to convert natural language descriptions into structured fraud signals before running the fraud detection pipeline.

Workflow:

Natural Language Description
→ LLM Feature Extraction
→ Structured Transaction Features
→ XGBoost Prediction
→ Risk Assessment
→ Similar Case Retrieval
→ RAG-Based Fraud Explanation

Example:

"Transaction of $3000 at 2AM from a new device in a foreign country"

The system automatically estimates relevant fraud indicators, generates structured model inputs, evaluates fraud risk, and produces a grounded explanation of the prediction.

| Step | Description |
|------|-------------|
| 1️ Predict | XGBoost model predicts fraud probability |
| 2️ Risk Level | Classified as HIGH / MEDIUM / LOW based on probability thresholds |
| 3️ Suggested Action | Automated recommendation (e.g. "Flag for manual review", "Block transaction") |
| 4️ Similar Cases | Retrieves past fraud cases from SQLite DB with matching risk profiles |

## RAG Upgrade

### Phase 1 — Build the Knowledge Base

Created `src/knowledge_base.py` with curated fraud-detection documents across multiple categories:

- **Feature-level rules**  
  Example: strong negative `V14`, `V10`, and `V12` values are closely associated with fraud risk patterns.
  
- **Fraud pattern descriptions**  
  Including:
  - Card-not-present fraud
  - POS skimming
  - Account takeover
  - Card testing
  - Friendly fraud / chargeback abuse
  - Synthetic identity fraud

- **Dataset-specific statistical context**  
  Such as fraud rate, peak fraud hours, amount-based risk patterns

- **General domain knowledge**  
  Fraud detection trade-offs, model interpretation context, and operational risk signals

### Phase 2 — Build the RAG Engine

Created `src/rag_engine.py` to retrieve the most relevant knowledge chunks before generating explanations.

Pipeline:
- Embed knowledge base documents using `sentence-transformers/all-MiniLM-L6-v2`
- Build vector representations for semantic search
- Retrieve the **top 3 most relevant chunks** for a fraud-related query
- Inject retrieved context into the Groq prompt before explanation generation

### Phase 3 — Update `app.py`

Upgraded from direct LLM explanation to a RAG-enhanced + Agent pipeline:

New UI features:
- 4-metric display: Fraud | Probability | Risk Level | Suggested Action
- Agent Pipeline table showing all decision steps
- Retrieved knowledge chunks shown in the interface
- Grounded fraud explanation generated from retrieved context
- Similar past fraud cases from persistent SQLite database
- **Knowledge Base** tab to browse all available fraud documents
- RAG-enhanced AI chat for fraud-related questions

## Project Structure

```bash
credit-card-fraud-detection/
│
├── data/
│   ├── creditcard.csv
│   └── fraud.db
│
├── notebooks/
│   ├── fraud.ipynb
│   └── main_experimental.py
│
├── src/
│   ├── __init__.py
│   ├── explainer.py
│   ├── knowledge_base.py
│   └── rag_engine.py
│
├── app.py
├── main.py
├── Best_model.pkl
├── Scaler.pkl
├── Dockerfile
├── Dockerfile.streamlit
├── requirements.txt
├── requirements_ui.txt
├── .env
│
└── README.md
```

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- SQLite
- FastAPI
- Pydantic
- Uvicorn
- Streamlit
- Docker
- Hugging Face Spaces
- Render
- Sentence Transformers
- Semantic Vector Retrieval
- Groq API
- LLaMA 3.1 8B

## Deployment

- **Frontend**: Streamlit app deployed on Hugging Face Spaces (`fraud-detection-ui`)
- **Backend**: FastAPI REST API deployed on Render (`fraud-detection-api`)
- **UI deployment**: Streamlit SDK with `app.py` as the entry point
- **API deployment**: Docker-based FastAPI container
- API returns:
  - `is_fraud` — boolean
  - `fraud_probability` — float
  - `risk_level` — HIGH / MEDIUM / LOW
  - `suggested_action` — automated recommendation string
  - `similar_cases` — list of past fraud cases from SQLite
  - `explanation` — RAG-grounded LLM explanation

## Key Learnings

- Accuracy is misleading for extreme class imbalance
- Precision-recall trade-offs matter more than raw accuracy
- XGBoost outperformed simpler baselines after imbalance-aware tuning
- High precision is critical because false positives damage user trust
- RAG improves explanation quality by grounding responses in curated fraud knowledge
- Agent pipelines add interpretability and automated decision support on top of ML predictions
- Separate Docker deployments per space prevent CMD conflicts in shared repos

## Future Improvements

- Replace in-memory semantic retrieval with a persistent vector index (e.g., FAISS)
- Add SHAP-based local explanations beside RAG explanations
- Add transaction history context for sequence-aware fraud analysis
- Introduce analyst feedback loops for continuous knowledge base refinement
- Add latency optimization and caching for faster inference

## Demo Questions

Try asking the app:
- "Why is V14 such a strong fraud signal?"
- "What does this transaction pattern suggest?"
- "What is account takeover fraud?"
- "Why is PR-AUC better than accuracy here?"
- "What makes high-value transactions riskier?"

## Author

**Arpit Kumar**  
AI/ML Engineer | Data Science | Fraud Detection | Applied ML Systems
