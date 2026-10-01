"""Text, embedding, topic, similarity, and fused-model benchmarks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ARTIFACTS_DIR, get_logger
from src.models.train_models import CAT, NUM

LOGGER = get_logger(__name__)
REPORT_PATH = ARTIFACTS_DIR / "nlp_metrics.json"
TEXT_MODEL_PATH = ARTIFACTS_DIR / "narrative_classifier.joblib"


def _severity_label(series: pd.Series) -> pd.Series:
    return series.map({"Minor": "low", "Moderate": "medium", "Major": "high"})


def _text_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)),
        ("classifier", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)),
    ])


def _dense_embedding_pipeline() -> Pipeline:
    """Dense local baseline; represents a deployable fallback if no transformer is cached."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)),
        ("svd", TruncatedSVD(n_components=48, random_state=42)),
        ("classifier", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)),
    ])


def transformer_embeddings(texts: list[str]) -> tuple[np.ndarray, str]:
    """Use an explicitly configured local transformer, with an offline dense fallback."""
    model_path = os.getenv("CLAIMIQ_SENTENCE_TRANSFORMER_PATH")
    if model_path and Path(model_path).is_dir():
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(model_path, local_files_only=True)
        return np.asarray(encoder.encode(texts, normalize_embeddings=True)), f"local SentenceTransformer: {Path(model_path).name}"
    vectors = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2).fit_transform(texts)
    return TruncatedSVD(n_components=min(48, vectors.shape[1] - 1), random_state=42).fit_transform(vectors), "offline dense TF-IDF/SVD fallback"


def classify_narrative(model: Pipeline, text: str) -> dict[str, Any]:
    probabilities = model.predict_proba([text])[0]
    classes = model.named_steps["classifier"].classes_
    return {"category": str(classes[probabilities.argmax()]), "confidence": round(float(probabilities.max()), 3)}


def extract_topics(texts: pd.Series, topic_count: int = 6) -> list[dict[str, Any]]:
    vectorizer = TfidfVectorizer(stop_words="english", min_df=3)
    matrix = vectorizer.fit_transform(texts)
    nmf = NMF(n_components=topic_count, random_state=42, init="nndsvda", max_iter=300).fit(matrix)
    words = np.array(vectorizer.get_feature_names_out())
    return [{"topic": index + 1, "terms": words[component.argsort()[-6:][::-1]].tolist()}
            for index, component in enumerate(nmf.components_)]


def similar_claims(texts: pd.Series, query_index: int, top_k: int = 5) -> list[int]:
    matrix = TfidfVectorizer(stop_words="english").fit_transform(texts)
    scores = cosine_similarity(matrix[query_index], matrix).ravel()
    candidates = np.argsort(scores)[::-1]
    return [int(index) for index in candidates if index != query_index][:top_k]


def train_nlp_models(df: pd.DataFrame) -> dict[str, Any]:
    """Benchmark narrative classification and a structured-plus-text loss model."""
    if "claim_narrative" not in df:
        raise ValueError("claim_narrative is required; regenerate synthetic data first.")
    train_index, test_index = train_test_split(df.index, test_size=0.2, stratify=df["claim_type"], random_state=42)
    train, test = df.loc[train_index], df.loc[test_index]

    category_model = _text_pipeline().fit(train["claim_narrative"], train["claim_type"])
    category_pred = category_model.predict(test["claim_narrative"])
    severity_model = _text_pipeline().fit(train["claim_narrative"], _severity_label(train["incident_severity"]))
    severity_pred = severity_model.predict(test["claim_narrative"])
    embedding_matrix, embedding_backend = transformer_embeddings(train["claim_narrative"].tolist() + test["claim_narrative"].tolist())
    train_embeddings, test_embeddings = embedding_matrix[:len(train)], embedding_matrix[len(train):]
    embedding_model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42).fit(train_embeddings, train["claim_type"])
    dense_pred = embedding_model.predict(test_embeddings)

    structured = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("text", TfidfVectorizer(stop_words="english", min_df=2, max_features=200), "claim_narrative"),
    ])
    combined_model = Pipeline([
        ("features", structured),
        ("model", __import__("sklearn.ensemble", fromlist=["RandomForestRegressor"]).RandomForestRegressor(
            n_estimators=120, min_samples_leaf=3, random_state=42, n_jobs=-1)),
    ]).fit(train[NUM + CAT + ["claim_narrative"]], train["actual_loss"])
    combined_pred = combined_model.predict(test[NUM + CAT + ["claim_narrative"]])

    metrics = {
        "narrative_category": {"accuracy": round(float(accuracy_score(test["claim_type"], category_pred)), 3),
                               "macro_f1": round(float(f1_score(test["claim_type"], category_pred, average="macro")), 3)},
        "narrative_severity": {"accuracy": round(float(accuracy_score(_severity_label(test["incident_severity"]), severity_pred)), 3),
                               "macro_f1": round(float(f1_score(_severity_label(test["incident_severity"]), severity_pred, average="macro")), 3)},
        "embedding_comparison": {"backend": embedding_backend,
                                 "accuracy": round(float(accuracy_score(test["claim_type"], dense_pred)), 3)},
        "combined_structured_text": {"mae": round(float(mean_absolute_error(test["actual_loss"], combined_pred)), 2)},
        "topics": extract_topics(df["claim_narrative"]),
    }
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(category_model, TEXT_MODEL_PATH)
    joblib.dump(severity_model, ARTIFACTS_DIR / "narrative_severity_classifier.joblib")
    joblib.dump(combined_model, ARTIFACTS_DIR / "combined_structured_text_model.joblib")
    REPORT_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Trained ClaimIQ NLP baseline models")
    return metrics


if __name__ == "__main__":
    from src.config import ROOT
    print(json.dumps(train_nlp_models(pd.read_csv(ROOT / "data" / "raw" / "synthetic_claims.csv")), indent=2))
