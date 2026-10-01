from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from common import load_claims, load_report
from src.monitoring.drift import run_monitoring

st.set_page_config(page_title="ClaimIQ | Monitoring", layout="wide")
st.title("Incoming Data & Prediction Monitoring")
st.caption("PSI: <0.10 stable, 0.10–0.25 watch, >=0.25 material drift.")
if st.button("Generate incoming batch and run monitoring", type="primary"):
    with st.spinner("Scoring incoming claims and calculating PSI..."):
        st.session_state["monitoring_report"] = run_monitoring(load_claims())
report = st.session_state.get("monitoring_report", load_report("monitoring_report.json"))
if report:
    c1, c2, c3 = st.columns(3)
    c1.metric("Incoming claims", f"{report['incoming_claims']:,}")
    c2.metric("Feature alerts", len(report["alert_features"]))
    c3.metric("Prediction alerts", len(report["alert_predictions"]))
    psi = pd.DataFrame({"feature": list(report["feature_psi"]), "psi": list(report["feature_psi"].values())}).sort_values("psi")
    st.plotly_chart(px.bar(psi, x="psi", y="feature", orientation="h", color="psi",
                           color_continuous_scale=["#fce7f3", "#ec4899", "#9d174d"],
                           title="Feature PSI"), use_container_width=True)
    st.dataframe(pd.DataFrame([report["prediction_psi"]]), use_container_width=True)
else:
    st.info("Run monitoring to create an incoming-data report.")
