import streamlit as st
import requests

st.set_page_config(page_title="Fraud Detection System", page_icon="🔍", layout="centered")

st.title("🔍 Credit Card Fraud Detection")
st.markdown("Enter transaction details below to check if it's fraudulent.")
st.divider()

st.subheader("Transaction Details")

col1, col2 = st.columns(2)
with col1:
    Time = st.number_input("Time", value=406.0)
    Amount = st.number_input("Amount ($)", value=149.62)

with col2:
    st.markdown("**V1 — V28** (PCA features)")

st.markdown("**PCA Features (V1 - V28)**")
cols = st.columns(4)
v_values = {}
for i in range(1, 29):
    col_idx = (i - 1) % 4
    with cols[col_idx]:
        v_values[f"V{i}"] = st.number_input(f"V{i}", value=0.0, format="%.4f")

st.divider()

if st.button("🔍 Predict", use_container_width=True):
    payload = {"Time": Time, "Amount": Amount}
    payload.update(v_values)

    try:
        response = requests.post("https://arpitkr-fraud-detection-api.hf.space/predict", json=payload)
        result = response.json()

        st.subheader("Prediction Result")

        if result["is_fraud"]:
            st.error(f"⚠️ FRAUDULENT TRANSACTION DETECTED!")
        else:
            st.success(f"✅ Legitimate Transaction")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fraud", "YES 🚨" if result["is_fraud"] else "NO ✅")
        with col2:
            st.metric("Probability", f"{result['fraud_probability']*100:.1f}%")
        with col3:
            st.metric("Risk Level", result["risk_level"])

    except Exception as e:
        st.error(f"API Error: {e}. Make sure your FastAPI server is running!")