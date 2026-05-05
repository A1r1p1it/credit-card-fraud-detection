# src/rag_engine.py

import numpy as np
from src.knowledge_base import FRAUD_KNOWLEDGE_BASE

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

class FraudRAGEngine:
    def __init__(self):
        self.model = None
        self.embeddings = None
        self.documents = FRAUD_KNOWLEDGE_BASE

    def load(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [f"{doc['title']}. {doc['content']}" for doc in self.documents]
        self.embeddings = self.model.encode(texts, convert_to_numpy=True)

    def retrieve(self, query: str, top_k: int = 3):
        if self.model is None:
            self.load()
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        scores = [cosine_similarity(query_embedding, emb) for emb in self.embeddings]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "title": self.documents[i]["title"],
                "category": self.documents[i]["category"],
                "content": self.documents[i]["content"],
                "score": float(scores[i])
            }
            for i in top_indices
        ]

    def build_rag_prompt(self, features: dict, probability: float, retrieved_docs: list) -> str:
        context = "\n\n".join([
            f"[{doc['category']}] {doc['title']}:\n{doc['content']}"
            for doc in retrieved_docs
        ])

        return f"""You are a fraud detection expert. Use the retrieved knowledge below to explain this transaction.

RETRIEVED KNOWLEDGE:
{context}

TRANSACTION:
- Amount: ${features.get('Amount', 0):.2f}
- Fraud Probability: {probability:.1%}
- V14: {features.get('V14', 0):.3f} (key fraud indicator)
- V10: {features.get('V10', 0):.3f}
- V12: {features.get('V12', 0):.3f}
- V17: {features.get('V17', 0):.3f}
- V4:  {features.get('V4', 0):.3f}

Using ONLY the retrieved knowledge above, explain in 2-3 sentences why this transaction is suspicious. Be specific about which features are abnormal and what fraud pattern they suggest."""

# Singleton instance
rag_engine = FraudRAGEngine()