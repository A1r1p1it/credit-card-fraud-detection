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

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        err = str(e).lower()

        if "quota" in err or "rate limit" in err or "429" in err or "forbidden" in err:
            return "Explanation unavailable right now because the LLM quota was exceeded. Please try again in a moment."

        return "Explanation temporarily unavailable, but the fraud prediction completed successfully."