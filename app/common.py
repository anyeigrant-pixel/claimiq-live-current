"""Shared cached resources and visual helpers for Streamlit pages."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.claims_agent import ClaimsAgent
from src.models.inference import ClaimModels

DATA = ROOT / "data" / "raw" / "synthetic_claims.csv"
ARTIFACTS = ROOT / "artifacts"
PINK_PALETTE = ["#9d174d", "#be185d", "#db2777", "#ec4899", "#f472b6", "#f9a8d4"]


@st.cache_data
def load_claims() -> pd.DataFrame:
    return pd.read_csv(DATA)


@st.cache_resource
def load_agent() -> ClaimsAgent:
    return ClaimsAgent()


@st.cache_resource
def load_models() -> ClaimModels:
    return ClaimModels()


def load_report(name: str) -> dict:
    path = ARTIFACTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def shap_chart(contributions: list[dict], title: str) -> None:
    frame = pd.DataFrame(contributions[:10]).sort_values("shap_value")
    chart = px.bar(frame, x="shap_value", y="feature", orientation="h", color="shap_value",
                   color_continuous_scale=["#fce7f3", "#ec4899", "#9d174d"], title=title)
    st.plotly_chart(chart, use_container_width=True)
