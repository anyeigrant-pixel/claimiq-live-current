from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from common import load_claims, load_report
from src.nlp.modeling import similar_claims, train_nlp_models
from src.nlp.narrative import analyze_narrative

st.set_page_config(page_title="ClaimIQ | Narrative Intelligence", layout="wide")
st.title("Claim Narrative Intelligence")
st.caption("Narrative analysis for adjuster support.")
df = load_claims()
claim_id = st.selectbox("Analyze claim", df.claim_id.tolist(), key="nlp_claim")
position = int(df.index[df.claim_id == claim_id][0])
row = df.iloc[position]

left, right = st.columns(2)
with left:
    st.subheader("Narrative")
    st.write(row.claim_narrative)
with right:
    signals = analyze_narrative(row.claim_narrative)
    st.metric("Urgency score", f"{signals['urgency_score']:.0%}", signals["sentiment"])
    st.write("High-risk terms:", ", ".join(signals["risk_terms"]) or "None")

st.subheader("Similar claims")
similar = similar_claims(df.claim_narrative, position)
st.dataframe(df.iloc[similar][["claim_id", "claim_type", "incident_severity", "actual_loss", "claim_narrative"]],
             hide_index=True, use_container_width=True)

st.subheader("Text-model benchmarks")
if st.button("Train narrative, embedding, and combined models", type="primary"):
    with st.spinner("Training local NLP benchmarks..."):
        st.session_state["nlp_metrics"] = train_nlp_models(df)
report = st.session_state.get("nlp_metrics", load_report("nlp_metrics.json"))
if report:
    cols = st.columns(4)
    cols[0].metric("Narrative category accuracy", f"{report['narrative_category']['accuracy']:.1%}")
    cols[1].metric("Text severity accuracy", f"{report['narrative_severity']['accuracy']:.1%}")
    cols[2].metric("Embedding accuracy", f"{report['embedding_comparison']['accuracy']:.1%}")
    cols[3].metric("Structured + text MAE", f"${report['combined_structured_text']['mae']:,.0f}")
    st.caption(f"Embedding backend: {report['embedding_comparison']['backend']}")
    st.subheader("Discovered narrative topics")
    st.dataframe(pd.DataFrame(report["topics"]), hide_index=True, use_container_width=True)
