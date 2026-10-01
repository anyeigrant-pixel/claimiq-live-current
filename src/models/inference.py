import joblib
import pandas as pd
from typing import Any

from src.config import ARTIFACTS_DIR, get_logger
from src.models.explainability import local_explanation
from src.models.train_models import main as train_models

ART = ARTIFACTS_DIR
LOGGER = get_logger(__name__)

class ClaimModels:
    def __init__(self):
        self.severity, self.risk = self._load_artifacts()

    @staticmethod
    def _load_artifacts() -> tuple[Any, Any]:
        """Load compatible model artifacts, rebuilding only after a known pickle mismatch."""
        try:
            return (
                joblib.load(ART / "severity_model.joblib"),
                joblib.load(ART / "risk_model.joblib"),
            )
        except ModuleNotFoundError as error:
            LOGGER.warning(
                "Model artifacts are incompatible with the current Python environment (%s); retraining locally.",
                error,
            )
            train_models()
            return (
                joblib.load(ART / "severity_model.joblib"),
                joblib.load(ART / "risk_model.joblib"),
            )

    def predict(self, record: dict) -> dict:
        X = pd.DataFrame([record])
        loss = float(self.severity.predict(X)[0])
        risk = float(self.risk.predict_proba(X)[0, 1])
        return {"predicted_loss": round(loss, 2), "investigation_probability": round(risk, 4)}

    def predict_batch(self, records: pd.DataFrame) -> pd.DataFrame:
        """Score a DataFrame while preserving its index."""
        return pd.DataFrame(
            {
                "predicted_loss": self.severity.predict(records),
                "investigation_probability": self.risk.predict_proba(records)[:, 1],
            },
            index=records.index,
        )

    def explain(self, record: dict[str, Any]) -> dict[str, Any]:
        """Return local SHAP explanations for both models."""
        return {
            "severity": local_explanation(self.severity, record, classification=False),
            "risk": local_explanation(self.risk, record, classification=True),
        }
