from dotenv import load_dotenv
load_dotenv()  # Must be called BEFORE os.getenv()

import streamlit as st
import requests
from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="Fraud Detection System", layout="centered")
st.title("Credit Card Fraud Detection")
st.divider()

sample_fraud = {
    "Time": 406.0, "V1": -2.3122, "V2": 1.9520, "V3": -1.6099,
    "V4": 3.9979, "V5": -0.5222, "V6": -1.4265, "V7": -2.5374,
    "V8": 1.3917, "V9": -2.7701, "V10": -2.7723, "V11": 3.2020,
    "V12": -2.8999, "V13": -0.5952, "V14": -4.2893, "V15": 0.3897,
    "V16": -1.1407, "V17": -2.8301, "V18": -0.0168, "V19": 0.4170,
    "V20": 0.1269, "V21": 0.5172, "V22": -0.0350, "V23": -0.4652,
    "V24": 0.3202, "V25": 0.0445, "V26": 0.1778, "V27": 0.2611,
    "V28": -0.1433, "Amount": 149.62
}

tab1, tab2 = st.tabs(["Fraud Detector", "AI Chat"])

with tab1:
    st.markdown("Enter transaction details below to check if it's fraudulent.")

    if st.button("Load Sample Fraud Transaction", use_container_width=True):
        for key, val in sample_fraud.items():
            st.session_state[key] = val

    st.subheader("Transaction Details")
    col1, col2 = st.columns(2)
    with col1:
        Time = st.number_input("Time", value=st.session_state.get("Time", 0.0))
        Amount = st.number_input("Amount ($)", value=st.session_state.get("Amount", 0.0))
    with col2:
        st.markdown("**V1 — V28** (PCA features)")

    st.markdown("**PCA Features (V1 - V28)**")
    cols = st.columns(4)
    v_values = {}
    for i in range(1, 29):
        col_idx = (i - 1) % 4
        with cols[col_idx]:
            v_values[f"V{i}"] = st.number_input(
                f"V{i}",
                value=st.session_state.get(f"V{i}", 0.0),
                format="%.4f"
            )

    st.divider()

    if st.button("Predict", use_container_width=True):
        payload = {"Time": Time, "Amount": Amount}
        payload.update(v_values)

        try:
            response = requests.post("http://127.0.0.1:8000/predict", json=payload)
            result = response.json()

            st.subheader("Prediction Result")

            if result["is_fraud"]:
                st.error("FRAUDULENT TRANSACTION DETECTED!")
            else:
                st.success("Legitimate Transaction")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Fraud", "YES" if result["is_fraud"] else "NO")
            with col2:
                st.metric("Probability", f"{result['fraud_probability']*100:.1f}%")
            with col3:
                st.metric("Risk Level", result["risk_level"])

            if result.get("explanation"):
                st.divider()
                st.markdown("### AI Fraud Analysis")
                st.info(result["explanation"])
                st.caption("Powered by LLaMA 3.3 via Groq")

            st.session_state["last_result"] = result
            st.session_state["last_payload"] = payload

        except Exception as e:
            st.error(f"API Error: {e}. Make sure your FastAPI server is running!")

with tab2:
    st.markdown("### 💬 Ask the AI about fraud detection")
    st.caption("Ask anything about the prediction, features, or fraud detection in general.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    system_prompt = """You are an expert AI assistant specializing in credit card fraud detection.
You help users understand fraud predictions made by an XGBoost machine learning model.
The dataset uses PCA-transformed features V1-V28 (anonymized for privacy), plus Time and Amount.
Key fraud indicators: V14, V10, V12, V17 (strongly negative = high fraud risk), V4 (high positive = suspicious).
Be concise, clear, and helpful. Explain technical concepts in simple terms."""

    if "last_result" in st.session_state:
        r = st.session_state["last_result"]
        p = st.session_state["last_payload"]
        system_prompt += f"""

Latest prediction context:
- Fraud detected: {r['is_fraud']}
- Probability: {r['fraud_probability']*100:.1f}%
- Risk level: {r['risk_level']}
- Amount: ${p['Amount']}
- Key features: V14={p['V14']:.4f}, V10={p['V10']:.4f}, V4={p['V4']:.4f}, V12={p['V12']:.4f}
- AI explanation: {r.get('explanation', 'N/A')}"""

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask me anything about fraud detection...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                messages = [{"role": "system", "content": system_prompt}]
                messages += st.session_state.chat_history

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    max_tokens=500
                )
                reply = response.choices[0].message.content
                st.write(reply)

        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()