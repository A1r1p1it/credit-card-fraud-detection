import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def explain_fraud(features: dict, probability: float) -> str:
    prompt = f"""You are a fraud detection expert. A credit card transaction was flagged as FRAUDULENT with {probability:.1%} confidence.

Transaction details:
- Amount: ${features.get('Amount', 0):.2f}
- V14: {features.get('V14', 0):.3f} (key fraud indicator)
- V10: {features.get('V10', 0):.3f}
- V12: {features.get('V12', 0):.3f}
- V17: {features.get('V17', 0):.3f}
- V4:  {features.get('V4', 0):.3f}

In 2-3 sentences, explain why this transaction is suspicious. Be specific about which features are abnormal. Keep it clear enough for a bank analyst."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return response.choices[0].message.content