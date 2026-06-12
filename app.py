from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import requests
import json
from groq import Groq
from src.rag_engine import rag_engine
import os
from src.knowledge_base import FRAUD_KNOWLEDGE_BASE
from collections import defaultdict

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

V_DEFAULTS = {f"V{i}": 0.0 for i in range(1, 29)}


def safe_groq_completion(messages, max_tokens=200, temperature=0.3):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip(), None
    except Exception as e:
        err = str(e).lower()
        if "quota" in err or "rate limit" in err or "429" in err or "forbidden" in err:
            return None, "Groq quota exceeded. Please try again in a moment."
        return None, f"LLM error: {str(e)}"


def run_prediction(payload):
    try:
        response = requests.post(
            "https://fraud-api-kzt3.onrender.com/predict",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()

        if "is_fraud" not in result:
            st.error(f"API Error: {result}")
            return

        st.subheader("Prediction Result")

        if result["is_fraud"]:
            st.error("FRAUDULENT TRANSACTION DETECTED!")
        else:
            st.success("Legitimate Transaction")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Fraud", "YES" if result["is_fraud"] else "NO")
        with col2:
            st.metric("Probability", f"{result['fraud_probability']*100:.1f}%")
        with col3:
            st.metric("Risk Level", result["risk_level"])
        with col4:
            st.metric("Action", result.get("suggested_action", "N/A"))

        st.divider()
        st.markdown("### 🤖 Agent Pipeline")
        risk = result["risk_level"]
        action = result.get("suggested_action", "N/A")
        action_color = "🔴" if risk == "HIGH" else "🟡" if risk == "MEDIUM" else "🟢"
        st.markdown(f"""
| Step | Output |
|------|--------|
| 1️ Predict | Fraud: **{'YES' if result['is_fraud'] else 'NO'}** — {result['fraud_probability']*100:.1f}% probability |
| 2️ Risk Level | {action_color} **{risk}** |
| 3️ Suggested Action | **{action}** |
""")

        similar = result.get("similar_cases", [])
        if similar:
            st.markdown(f"### Similar Past Fraud Cases ({len(similar)} found)")
            for i, c in enumerate(similar, 1):
                st.markdown(
                    f"**Case {i}** — Amount: `${c['amount']}` | "
                    f"Probability: `{c['fraud_probability']*100:.1f}%` | "
                    f"Risk: `{c['risk_level']}` | "
                    f"Time: `{c['timestamp']}`"
                )
        else:
            st.markdown("### Similar Past Fraud Cases")
            st.caption("No similar past fraud cases found yet. Run more fraud predictions to build history.")

        if result["is_fraud"]:
            st.divider()
            st.markdown("### RAG Fraud Analysis")

            with st.spinner("Retrieving relevant fraud knowledge..."):
                query = (
                    f"V14={payload['V14']:.2f} V10={payload['V10']:.2f} "
                    f"V12={payload['V12']:.2f} amount={payload['Amount']} fraud detected"
                )
                retrieved_docs = rag_engine.retrieve(query, top_k=3)

            st.markdown("**Retrieved Knowledge Chunks**")
            for i, doc in enumerate(retrieved_docs, 1):
                with st.expander(f"{i}. [{doc['category']}] {doc['title']} — relevance: {doc['score']:.3f}"):
                    st.write(doc["content"])

            with st.spinner("Generating grounded explanation..."):
                rag_prompt = rag_engine.build_rag_prompt(payload, result["fraud_probability"], retrieved_docs)
                rag_explanation, rag_error = safe_groq_completion(
                    messages=[{"role": "user", "content": rag_prompt}],
                    max_tokens=200
                )

            if rag_explanation:
                st.info(rag_explanation)
            else:
                st.warning(rag_error or "Explanation unavailable right now — try again in a moment.")

            st.caption("Powered by RAG (sentence-transformers + FAISS cosine similarity) + LLaMA 3.1 via Groq")

        elif result.get("explanation"):
            st.divider()
            st.markdown("### AI Fraud Analysis")
            st.info(result["explanation"])
            st.caption("Powered by LLaMA 3.1 via Groq")

        st.session_state["last_result"] = result
        st.session_state["last_payload"] = payload

    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
    except Exception as e:
        st.error(f"Something went wrong: {e}")


tab1, tab2, tab3, tab4 = st.tabs(["Fraud Detector", "Natural Language", "AI Chat", "Knowledge Base"])


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
        run_prediction(payload)


with tab2:
    st.markdown("### 🗣️ Describe a Transaction in Plain English")
    st.caption("Describe the transaction naturally — the AI will extract features and run fraud detection.")

    st.markdown("""
**Try these examples:**
- `Transaction of $500 at 2AM from an unknown location`
- `Small $12 purchase at a local coffee shop during business hours`
- `$3000 international wire transfer at midnight on a new device`
- `Contactless $80 tap payment at grocery store at 6PM`
""")

    nl_input = st.text_area(
        "Describe the transaction",
        placeholder="e.g. Transaction of $500 at 2AM from an unknown location using a new device...",
        height=100
    )

    if st.button("Analyze Transaction", use_container_width=True):
        if not nl_input.strip():
            st.warning("Please describe a transaction first.")
        else:
            with st.spinner("Parsing transaction with AI..."):
                parse_prompt = f"""You are a credit card fraud detection feature extractor.

A user described a transaction in plain English. Extract structured features for a fraud detection model.

The model uses these key PCA features. Estimate values based on the transaction description:
- V14: Most important fraud signal. Normal: near 0. Suspicious: -2 to -5
- V10: Second most important. Suspicious: -2 to -4
- V12: Third most important. Suspicious: -2 to -4
- V17: Fourth. Suspicious: -2 to -3
- V4: High positive (2 to 4) = suspicious geographic anomaly
- All other V features: set to 0.0

Risk factors that push V14/V10/V12/V17 negative and V4 positive:
- Late night (12AM-4AM): high risk
- Unknown/foreign location: high risk
- Large amount (>$500): moderate risk
- New device or card: high risk
- International transaction: high risk
- Online/card-not-present: moderate risk
- Normal business hours + known location: low risk
- Small everyday purchase: low risk

Transaction: "{nl_input}"

Respond ONLY with valid JSON, no explanation:
{{
  "Amount": <number>,
  "Time": <seconds from midnight, e.g. 2AM = 7200>,
  "V4": <number>,
  "V10": <number>,
  "V12": <number>,
  "V14": <number>,
  "V17": <number>,
  "reasoning": "<one sentence explaining your risk assessment>"
}}"""

                raw, parse_error = safe_groq_completion(
                    messages=[{"role": "user", "content": parse_prompt}],
                    max_tokens=300
                )

            if not raw:
                st.error(parse_error or "Could not parse transaction right now.")
            else:
                try:
                    start = raw.find("{")
                    end = raw.rfind("}") + 1
                    parsed = json.loads(raw[start:end])
                    reasoning = parsed.pop("reasoning", "")

                    payload = {**V_DEFAULTS}
                    payload["Time"] = float(parsed.get("Time", 0))
                    payload["Amount"] = float(parsed.get("Amount", 0))
                    for key in ["V4", "V10", "V12", "V14", "V17"]:
                        if key in parsed:
                            payload[key] = float(parsed[key])

                    st.markdown("### Extracted Features")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Amount", f"${payload['Amount']:.2f}")
                    with c2:
                        h = int(payload['Time'] // 3600)
                        st.metric("Time", f"{h:02d}:00")
                    with c3:
                        st.metric("V14 (key signal)", f"{payload['V14']:.2f}")

                    c4, c5, c6 = st.columns(3)
                    with c4:
                        st.metric("V10", f"{payload['V10']:.2f}")
                    with c5:
                        st.metric("V12", f"{payload['V12']:.2f}")
                    with c6:
                        st.metric("V4", f"{payload['V4']:.2f}")

                    if reasoning:
                        st.info(f"**AI Reasoning:** {reasoning}")

                    st.divider()
                    run_prediction(payload)

                except Exception as e:
                    st.error(f"Failed to parse AI response: {e}")
                    st.code(raw)


with tab3:
    st.markdown("### Ask the AI about fraud detection")
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
- Suggested action: {r.get('suggested_action', 'N/A')}
- Amount: ${p['Amount']}
- Key features: V14={p['V14']:.4f}, V10={p['V10']:.4f}, V4={p['V4']:.4f}, V12={p['V12']:.4f}
- AI explanation: {r.get('explanation', 'N/A')}"""

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask me anything about fraud detection...")

    if user_input:
        chat_docs = rag_engine.retrieve(user_input, top_k=2)
        rag_context = "\n\n".join([f"[{d['category']}] {d['title']}: {d['content']}" for d in chat_docs])
        enhanced_system = system_prompt + f"\n\nRelevant knowledge:\n{rag_context}"

        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                messages = [{"role": "system", "content": enhanced_system}]
                messages += st.session_state.chat_history
                reply, chat_error = safe_groq_completion(
                    messages=messages,
                    max_tokens=300
                )

                if reply:
                    st.write(reply)
                else:
                    reply = "AI chat is temporarily unavailable because the model quota was exceeded. Please try again shortly."
                    st.warning(reply)

        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


with tab4:
    st.markdown("### Fraud Knowledge Base")
    st.caption(f"{len(rag_engine.documents)} documents across 6 categories")

    by_category = defaultdict(list)
    for doc in FRAUD_KNOWLEDGE_BASE:
        by_category[doc["category"]].append(doc)

    for category, docs in by_category.items():
        st.markdown(f"#### {category} ({len(docs)} docs)")
        for doc in docs:
            with st.expander(doc["title"]):
                st.write(doc["content"])
        st.divider()