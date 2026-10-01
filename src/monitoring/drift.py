"""Synthetic incoming batch monitoring using population stability index."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, get_logger
from src.data.generate_data import generate
from src.models.inference import ClaimModels
from src.models.train_models import FEATURES, NUM

LOGGER = get_logger(__name__)
REPORT_PATH = ARTIFACTS_DIR / "monitoring_report.json"


def population_stability_index(baseline: pd.Series, incoming: pd.Series, bins: int = 10) -> float:
    """Calculate PSI for numerical distributions; >0.25 conventionally signals material drift."""
    edges = np.unique(np.quantile(baseline.dropna(), np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    expected, _ = np.histogram(baseline, bins=edges)
    actual, _ = np.histogram(incoming, bins=edges)
    expected_pct = np.clip(expected / max(expected.sum(), 1), 1e-6, None)
    actual_pct = np.clip(actual / max(actual.sum(), 1), 1e-6, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def categorical_psi(baseline: pd.Series, incoming: pd.Series) -> float:
    categories = baseline.astype(str).unique()
    expected = baseline.astype(str).value_counts(normalize=True).reindex(categories, fill_value=0)
    actual = incoming.astype(str).value_counts(normalize=True).reindex(categories, fill_value=0)
    expected_pct = np.clip(expected.to_numpy(), 1e-6, None)
    actual_pct = np.clip(actual.to_numpy(), 1e-6, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def generate_incoming_claims(n: int = 1_500, seed: int = 2026) -> pd.DataFrame:
    """Create an intentionally shifted, synthetic operational batch."""
    incoming = generate(n=n, seed=seed)
    incoming["reported_amount"] *= 1.18
    incoming["days_to_report"] = np.minimum(incoming["days_to_report"] + 4, 60)
    incoming["prior_claims"] += 1
    return incoming


def run_monitoring(baseline: pd.DataFrame, incoming: pd.DataFrame | None = None) -> dict[str, Any]:
    """Score an incoming batch and persist data/prediction drift metrics."""
    incoming = incoming.copy() if incoming is not None else generate_incoming_claims()
    models = ClaimModels()
    incoming_predictions = models.predict_batch(incoming[FEATURES])
    baseline_predictions = models.predict_batch(baseline[FEATURES])
    features = {
        feature: round(
            population_stability_index(baseline[feature], incoming[feature]) if feature in NUM
            else categorical_psi(baseline[feature], incoming[feature]),
            4,
        )
        for feature in FEATURES
    }
    prediction_psi = {
        "severity_prediction": round(population_stability_index(baseline_predictions["predicted_loss"], incoming_predictions["predicted_loss"]), 4),
        "investigation_probability": round(population_stability_index(baseline_predictions["investigation_probability"], incoming_predictions["investigation_probability"]), 4),
    }
    report: dict[str, Any] = {
        "incoming_claims": int(len(incoming)),
        "feature_psi": features,
        "prediction_psi": prediction_psi,
        "alert_features": [name for name, value in features.items() if value >= 0.25],
        "alert_predictions": [name for name, value in prediction_psi.items() if value >= 0.25],
        "thresholds": {"watch": 0.1, "material_drift": 0.25},
    }
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Wrote drift report to %s", REPORT_PATH)
    return report


if __name__ == "__main__":
    data = pd.read_csv(ARTIFACTS_DIR.parent / "data" / "raw" / "synthetic_claims.csv")
    print(json.dumps(run_monitoring(data), indent=2))
