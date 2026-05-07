FRAUD_KNOWLEDGE_BASE = [


    {
        "id": "v14_rule",
        "category": "Feature Rule",
        "title": "V14 — Strongest Single Fraud Indicator",
        "content": (
            "V14 is the most powerful fraud predictor in this dataset, accounting for ~20% of feature importance. "
            "V14 values below -3.0 are highly suspicious. Values below -5.0 are almost exclusively associated with fraud. "
            "V14 captures transaction authentication patterns. Strongly negative V14 suggests the transaction "
            "bypassed normal card authentication, consistent with skimmed card data or card-not-present fraud."
        )
    },
    {
        "id": "v10_rule",
        "category": "Feature Rule",
        "title": "V10 — Second Strongest Fraud Signal",
        "content": (
            "V10 is the second most important fraud feature (~12% importance). "
            "Values below -3.0 are strongly associated with fraud. "
            "V10 correlates with merchant category and transaction routing anomalies. "
            "Fraudsters often use compromised cards at specific merchant types (electronics, gift cards) "
            "that show up as extreme V10 deviations."
        )
    },
    {
        "id": "v12_rule",
        "category": "Feature Rule",
        "title": "V12 — Third Key Fraud Feature",
        "content": (
            "V12 contributes ~10% to fraud detection importance. "
            "Strongly negative V12 (below -3.0) is a major red flag. "
            "V12 captures spending behaviour deviations relative to the cardholder's historical profile. "
            "A V12 anomaly combined with V14 or V10 anomaly multiplies fraud likelihood significantly."
        )
    },
    {
        "id": "v17_rule",
        "category": "Feature Rule",
        "title": "V17 — Transaction Velocity Indicator",
        "content": (
            "V17 (~10% importance) reflects transaction velocity and timing patterns. "
            "Strongly negative V17 suggests unusually rapid sequential transactions, "
            "a common pattern in card-present skimming fraud where stolen cards are used immediately "
            "before the victim notices. V17 < -3.0 combined with high amount is a critical alert."
        )
    },
    {
        "id": "v4_rule",
        "category": "Feature Rule",
        "title": "V4 — High Positive Value as Fraud Signal",
        "content": (
            "Unlike most fraud features that show negative extremes, V4 shows HIGH POSITIVE values in fraud. "
            "V4 > 2.0 combined with negative V14 is a strong fraud pattern. "
            "V4 (~9% importance) captures geographic or terminal anomalies. "
            "A cardholder transacting far from their usual location or on an unfamiliar terminal type "
            "produces elevated V4 scores."
        )
    },
    {
        "id": "v11_rule",
        "category": "Feature Rule",
        "title": "V11 — Account Age and History Signal",
        "content": (
            "V11 with strongly positive values (>2.0) in combination with negative V14 indicates "
            "account takeover fraud — where a legitimate long-standing account is suddenly used fraudulently. "
            "This pattern suggests the fraudster has stolen credentials of an established account."
        )
    },
    {
        "id": "v3_rule",
        "category": "Feature Rule",
        "title": "V3 — Transaction Frequency Anomaly",
        "content": (
            "Strongly negative V3 (below -2.0) suggests an unusual spike in transaction frequency. "
            "Fraudsters often make multiple rapid transactions to test a stolen card (small amounts first) "
            "then escalate to larger purchases. V3 captures this rapid-fire pattern."
        )
    },


    {
        "id": "card_skimming",
        "category": "Fraud Pattern",
        "title": "Card Skimming / POS Terminal Fraud",
        "content": (
            "Card skimming involves physically copying card data from legitimate POS terminals or ATMs. "
            "Skimmed cards are used immediately (within hours) at different merchant locations. "
            "Signature in data: strongly negative V14 (auth failure pattern), negative V10, "
            "transaction time between 11PM-4AM, amounts between $100-$500. "
            "This is the most common fraud type in this dataset."
        )
    },
    {
        "id": "card_not_present",
        "category": "Fraud Pattern",
        "title": "Card-Not-Present (CNP) / Online Fraud",
        "content": (
            "CNP fraud occurs when stolen card details are used for online purchases without the physical card. "
            "Fraudsters buy digital goods (gift cards, electronics, subscriptions) that are easily resalable. "
            "Data signature: moderate V14 anomaly, high V4 (geographic mismatch), "
            "amounts typically $50-$300, multiple transactions in short window. "
            "CNP fraud increased dramatically with e-commerce growth."
        )
    },
    {
        "id": "account_takeover",
        "category": "Fraud Pattern",
        "title": "Account Takeover Fraud",
        "content": (
            "Account takeover (ATO) occurs when a fraudster steals login credentials and takes control "
            "of a legitimate bank account. The victim's established account history paradoxically "
            "helps the fraud pass initial filters. "
            "Signature: positive V11 (old account), sudden V14 anomaly, large amounts, "
            "new device or location (V4 spike). ATO is hardest to detect because the account history is genuine."
        )
    },
    {
        "id": "card_testing",
        "category": "Fraud Pattern",
        "title": "Card Testing / Carding Attack",
        "content": (
            "Fraudsters test stolen card numbers with small transactions (under $10) before making large purchases. "
            "This creates a distinctive pattern: multiple tiny transactions followed by a large one. "
            "Data signature: very small Amount (<$5) followed by large Amount (>$200) within minutes, "
            "negative V3 (rapid velocity), consistent merchant category codes. "
            "Card testing is an early warning signal — the real fraud comes after the test succeeds."
        )
    },
    {
        "id": "friendly_fraud",
        "category": "Fraud Pattern",
        "title": "Friendly Fraud / Chargeback Fraud",
        "content": (
            "Friendly fraud is when a legitimate cardholder makes a purchase and then falsely claims "
            "it was unauthorized to get a chargeback refund. "
            "This is difficult to detect with ML because the transaction itself looks legitimate. "
            "Usually involves high-value items, online purchases, and repeat chargeback history. "
            "V features tend to be near-normal in friendly fraud cases."
        )
    },
    {
        "id": "synthetic_identity",
        "category": "Fraud Pattern",
        "title": "Synthetic Identity Fraud",
        "content": (
            "Synthetic identity fraud combines real and fake information to create a new identity. "
            "The fraudster builds credit slowly over months, then 'busts out' — maxing all credit at once. "
            "Very hard to detect at transaction level. Signature: sudden large transactions after "
            "long period of normal activity, diverse merchant categories, amounts near credit limit."
        )
    },


    {
        "id": "dataset_fraud_rate",
        "category": "Dataset Statistics",
        "title": "Overall Fraud Rate in Dataset",
        "content": (
            "This fraud detection system was trained on 284,807 real credit card transactions. "
            "Only 492 transactions (0.17%) are fraudulent — extreme class imbalance. "
            "This means a naive model predicting 'no fraud' for everything would be 99.83% accurate "
            "but completely useless. The model uses XGBoost with scale_pos_weight=25 to handle this imbalance."
        )
    },
    {
        "id": "high_value_risk",
        "category": "Dataset Statistics",
        "title": "High-Value Transactions Have 2x Higher Fraud Rate",
        "content": (
            "SQL analysis of the dataset reveals that transactions above $200 have a fraud rate of 0.29%, "
            "exactly double the overall 0.17% fraud rate. "
            "Fraudulent transactions average $122 in amount versus $88 for legitimate transactions. "
            "This confirms that fraudsters target higher-value transactions. "
            "A transaction above $200 with V14 anomaly should be treated as very high risk."
        )
    },
    {
        "id": "time_patterns",
        "category": "Dataset Statistics",
        "title": "Peak Fraud Hours: 2AM-3AM",
        "content": (
            "Time-based SQL analysis reveals fraud peaks dramatically between 2AM and 3AM, "
            "with a fraud rate of 1.33% during this window — nearly 8x the baseline rate. "
            "Fraudsters prefer late night hours because: victims are asleep and won't notice alerts, "
            "bank fraud teams are understaffed, and stolen card data is often used immediately after theft. "
            "A transaction with V14 anomaly occurring in the 2AM-3AM window (Time ~7200-10800 seconds) "
            "should be treated as extremely high risk."
        )
    },
    {
        "id": "amount_distribution",
        "category": "Dataset Statistics",
        "title": "Fraud Amount Distribution",
        "content": (
            "Legitimate transactions average $88.35 in amount. "
            "Fraudulent transactions average $122.21 — about 38% higher. "
            "However, the distribution is wide: some fraud occurs at very low amounts (card testing under $5) "
            "and some at very high amounts (bust-out fraud over $1000). "
            "Amount alone is a weak predictor — it must be combined with V-feature anomalies."
        )
    },


    {
        "id": "xgboost_model",
        "category": "Model Knowledge",
        "title": "XGBoost Model Configuration and Performance",
        "content": (
            "The fraud detection model is XGBoost with 200 estimators, learning_rate=0.5, scale_pos_weight=25. "
            "scale_pos_weight=25 means the model treats each fraud case as 25x more important than legitimate, "
            "compensating for the 0.17% fraud rate. "
            "Performance: Precision=91%, Recall=84%, F1=0.87, PR-AUC=0.878. "
            "This means 91% of flagged transactions are actually fraud (low false alarm rate) "
            "and 84% of real fraud is caught."
        )
    },
    {
        "id": "why_not_accuracy",
        "category": "Model Knowledge",
        "title": "Why Accuracy is Misleading for Fraud Detection",
        "content": (
            "Accuracy is the wrong metric for fraud detection. "
            "A model predicting 'no fraud' for every transaction achieves 99.83% accuracy but catches 0% of fraud. "
            "The correct metrics are: Precision (of all flagged transactions, what % are actually fraud), "
            "Recall (of all real fraud, what % did we catch), and PR-AUC (area under precision-recall curve). "
            "Our XGBoost achieves PR-AUC=0.878 vs Logistic Regression at 0.764."
        )
    },
    {
        "id": "false_positives",
        "category": "Model Knowledge",
        "title": "Cost of False Positives vs False Negatives",
        "content": (
            "False Positive (legitimate transaction flagged as fraud): frustrates the customer, "
            "damages trust, may cause card decline at point of sale — costs $10-50 in customer service. "
            "False Negative (fraud missed): bank absorbs the loss — costs $50-5000 per case. "
            "Our model prioritizes precision (91%) to minimize false positives that damage customer experience, "
            "while maintaining 84% recall to catch most fraud."
        )
    },
    {
        "id": "shap_explainability",
        "category": "Model Knowledge",
        "title": "SHAP Values for Model Explainability",
        "content": (
            "SHAP (SHapley Additive exPlanations) values show how each feature contributed to a specific prediction. "
            "A positive SHAP value means the feature pushed the prediction toward fraud. "
            "A negative SHAP value means the feature pushed toward legitimate. "
            "For fraud cases, V14 typically has the largest absolute SHAP value. "
            "SHAP values make the XGBoost black-box model interpretable for analysts and regulators."
        )
    },


    {
        "id": "high_risk_combo",
        "category": "Risk Rules",
        "title": "High Risk Combination: V14 + V10 + V12 Anomalies",
        "content": (
            "When V14 < -3.0 AND V10 < -2.0 AND V12 < -2.0 simultaneously, "
            "the fraud probability is extremely high (>90%). "
            "This triple-feature anomaly pattern is present in the majority of confirmed fraud cases. "
            "Each feature alone might be explainable, but their simultaneous extreme deviation "
            "is almost always indicative of card skimming or stolen card use."
        )
    },
    {
        "id": "medium_risk_signals",
        "category": "Risk Rules",
        "title": "Medium Risk Signals Worth Monitoring",
        "content": (
            "Medium risk indicators (fraud probability 30-70%): "
            "Single V14 anomaly below -2.0 with normal other features, "
            "High amount (>$300) with slightly elevated V4, "
            "V17 < -2.0 suggesting rapid sequential transactions, "
            "Transaction between 1AM-4AM with any single V-feature anomaly."
        )
    },
    {
        "id": "low_risk_context",
        "category": "Risk Rules",
        "title": "Low Risk Context — When Anomalies Are Expected",
        "content": (
            "Some transactions appear anomalous but are legitimate: "
            "International travel (V4 spike is expected), "
            "Large one-time purchases like electronics or jewelry (high amount normal), "
            "New merchant category the cardholder hasn't used before. "
            "Context matters — isolated V-feature deviations with normal Amount and Time "
            "have lower fraud probability than multi-feature anomalies."
        )
    },


    {
        "id": "pci_dss",
        "category": "Industry Context",
        "title": "PCI DSS and Data Privacy — Why Features Are Anonymized",
        "content": (
            "The V1-V28 features in this dataset are PCA-transformed for privacy compliance under PCI DSS "
            "(Payment Card Industry Data Security Standard). "
            "PCI DSS prohibits sharing raw cardholder data. PCA transformation preserves mathematical "
            "relationships needed for ML while making the data uninterpretable to humans. "
            "This is why we cannot say 'V14 = merchant ID' — the mapping is intentionally obscured."
        )
    },
    {
        "id": "industry_fraud_stats",
        "category": "Industry Context",
        "title": "Global Credit Card Fraud Statistics",
        "content": (
            "Credit card fraud costs the global economy over $32 billion annually. "
            "The United States accounts for ~35% of global card fraud despite having only 22% of card volume. "
            "Card-not-present fraud has grown 140% over 5 years with e-commerce expansion. "
            "Real-time fraud detection systems must decide in under 100ms whether to approve a transaction. "
            "Machine learning has reduced fraud losses by 40-60% compared to rule-based systems alone."
        )
    },
]