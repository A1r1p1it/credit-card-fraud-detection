# Credit Card Fraud Detection

Binary classification system to detect fraudulent credit card transactions in highly imbalanced dataset.

## Problem Statement
Detect credit card fraud with extreme class imbalance (0.17% fraud rate). Accuracy is misleading - a model predicting "no fraud" for everything achieves 99.8% accuracy but catches 0% fraud.

## Dataset
- **Source**: Kaggle Credit Card Fraud Detection
- **Size**: 284,807 transactions
- **Features**: 30 (V1-V28 PCA-transformed + Time + Amount)
- **Target**: Class (0 = Non-Fraud, 1 = Fraud)
- **Imbalance**: 492 frauds (0.17%)

## Approach

### 1. Data Preprocessing
- Stratified train-test split (80/20) to preserve class distribution
- StandardScaler for feature normalization

### 2. Models Compared

**Logistic Regression** (Baseline)
- Used `class_weight='balanced'` to handle imbalance
- High recall (92%) but very low precision (6%)
- 94% false alarm rate - too many false positives

**Random Forest**
- 100 trees, max_depth=5, `class_weight='balanced'`
- Improved precision to 36% while maintaining 89% recall
- Still 64% false alarm rate

**XGBoost** (Best Model)
- 200 estimators, learning_rate=0.5, `scale_pos_weight=25`
- Best performance: 91% precision, 84% recall
- Only 9% false alarm rate - optimal for production

### 3. Evaluation Strategy
- Evaluated Precision-Recall Curve and computed PR-AUC to assess performance under extreme class imbalance
- Used precision, recall, F1-score (not accuracy)
- Confusion matrix to analyze false positives vs false negatives
- Compared against baseline (model predicting all non-fraud gets 99.8% accuracy)

## Results

| Model | Precision | Recall | F1-Score | PR-AUC | Trade-off |
|-------|-----------|--------|----------|--------|-----------|
| Logistic Regression | 0.06 | 0.92 | 0.11 | 0.764 | Catches 92% fraud but 94% false alarms |
| Random Forest | 0.36 | 0.89 | 0.51 | 0.659 | Better balance, still 64% false alarms |
| XGBoost | 0.91 | 0.84 | 0.87 | 0.878 | Only 9% false alarms, catches 84% fraud |

**Key Insight**: XGBoost achieved best precision-recall balance, making it production-ready with minimal false positives while still detecting majority of fraud. XGBoost also achieved the highest PR-AUC (0.878), confirming superior performance across probability thresholds in highly imbalanced settings.

## Feature Importance
Top 5 features identified by Random Forest and XGBoost:
1. V14 (~20%)
2. V10 (~12%)
3. V12 (~10%)
4. V17 (~10%)
5. V4 (~9%)

## Technologies Used
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Jupyter Notebook

## Key Learnings
- Accuracy is misleading for imbalanced datasets
- Baseline model (predict all non-fraud) achieves 99.8% accuracy but is useless
- Precision-recall trade-off is critical in fraud detection
- XGBoost with `scale_pos_weight` tuning outperforms simpler models
- False positives damage customer trust, so high precision matters
