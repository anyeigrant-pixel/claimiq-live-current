"""SHAP explanations for the trained ClaimIQ model pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

from src.config import get_logger
from src.models.train_models import FEATURES

LOGGER = get_logger(__name__)


def _transformed_feature_names(pipeline: Any) -> list[str]:
    return [str(name) for name in pipeline.named_steps["prep"].get_feature_names_out()]


def _explain(pipeline: Any, records: pd.DataFrame, classification: bool) -> tuple[np.ndarray, float, list[str]]:
    """Explain transformed records with SHAP's tree explainer."""
    transformed = pipeline.named_steps["prep"].transform(records[FEATURES])
    model = pipeline.named_steps["model"]
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(transformed)
    except Exception as error:
        LOGGER.exception("Unable to create SHAP explanation")
        raise RuntimeError("SHAP explanation generation failed.") from error

    if classification and isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if classification and values.ndim == 3:
        values = values[:, :, 1]
    expected = explainer.expected_value
    if classification and isinstance(expected, (list, np.ndarray)):
        expected = expected[1]
    return values, float(np.asarray(expected).reshape(-1)[-1 if classification else 0]), _transformed_feature_names(pipeline)


def local_explanation(pipeline: Any, record: dict[str, Any], classification: bool) -> dict[str, Any]:
    """Return ranked SHAP feature contributions for one claim."""
    values, base_value, names = _explain(pipeline, pd.DataFrame([record]), classification)
    contributions = [
        {"feature": name.replace("num__", "").replace("cat__", ""), "shap_value": round(float(value), 4)}
        for name, value in zip(names, values[0], strict=True)
    ]
    contributions.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
    return {"base_value": round(base_value, 4), "contributions": contributions}


def global_importance(pipeline: Any, records: pd.DataFrame, classification: bool, max_rows: int = 500) -> list[dict[str, float | str]]:
    """Return mean absolute SHAP values, aggregated by transformed feature."""
    sample = records[FEATURES].head(max_rows)
    values, _, names = _explain(pipeline, sample, classification)
    importance = np.abs(values).mean(axis=0)
    results = [
        {"feature": name.replace("num__", "").replace("cat__", ""), "mean_abs_shap": round(float(value), 5)}
        for name, value in zip(names, importance, strict=True)
    ]
    return sorted(results, key=lambda item: float(item["mean_abs_shap"]), reverse=True)
