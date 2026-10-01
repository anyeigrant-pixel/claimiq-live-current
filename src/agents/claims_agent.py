from __future__ import annotations
from src.models.inference import ClaimModels
from src.rag.retriever import retrieve

class ClaimsAgent:
    def __init__(self):
        self.models = ClaimModels()

    def review(self, record: dict) -> dict:
        prediction = self.models.predict(record)
        passages = retrieve(f"{record['policy_type']} {record['claim_type']} coverage claim reporting")
        risk = prediction["investigation_probability"]
        if risk >= .70:
            action = "Route for enhanced human review."
        elif risk >= .40:
            action = "Route for standard adjuster review with additional verification."
        else:
            action = "Proceed with standard adjuster review."
        return {
            **prediction,
            "recommended_action": action,
            "policy_evidence": [{"source": p.source, "text": p.text, "score": round(p.score, 3), "citation": p.citation} for p in passages],
            "explanations": self.models.explain(record),
            "disclaimer": "Decision support only. Final claim decisions require qualified human review."
        }
