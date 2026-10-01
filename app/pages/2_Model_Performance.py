from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from common import load_claims, load_models, load_report
from src.models.explainability import global_importance
from src.models.train_models import main as train_models

st.set_page_config(page_title="ClaimIQ | Model Performance", layout="wide")
st.title("Model Performance & Explainability")
metrics = load_report("metrics.json")
if not metrics:
    with st.spinner("Preparing model artifacts and performance metrics..."):
        train_models()
    metrics = load_report("metrics.json")
if not metrics:
    st.error("Model metrics could not be generated.")
    st.stop()

severity, risk = st.columns(2)
with severity:
    st.subheader("Severity model")
    for name, value in metrics["severity"].items():
        st.metric(name.upper(), f"{value:,.3f}")
with risk:
    st.subheader("Investigation-risk model")
    for name, value in metrics["risk"].items():
        st.metric(name.replace("_", " ").upper(), f"{value:.3f}")

if st.button("Calculate global SHAP importance"):
    models, data = load_models(), load_claims()
    for label, pipeline, classification in [("Severity", models.severity, False), ("Investigation risk", models.risk, True)]:
        importance = pd.DataFrame(global_importance(pipeline, data, classification))
        st.plotly_chart(px.bar(importance.head(15).sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature",
                               orientation="h", color="mean_abs_shap",
                               color_continuous_scale=["#fce7f3", "#ec4899", "#9d174d"],
                               title=f"{label}: mean absolute SHAP importance"),
                        use_container_width=True)
st.caption("SHAP values describe model behavior.")
