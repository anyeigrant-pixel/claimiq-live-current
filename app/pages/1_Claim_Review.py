from __future__ import annotations

import streamlit as st

from common import load_agent, load_claims, shap_chart
from src.nlp.narrative import analyze_narrative

st.set_page_config(page_title="ClaimIQ | Claim Review", layout="wide")
st.title("Individual Claim Review")
df = load_claims()
claim_id = st.selectbox("Claim ID", df["claim_id"].tolist())
row = df.loc[df.claim_id == claim_id].iloc[0]
record = row.drop(labels=["claim_id", "customer_id", "policy_id", "actual_loss", "investigation_flag"]).to_dict()

left, right = st.columns([1, 1])
with left:
    st.subheader("Claim profile")
    st.dataframe(row.to_frame("value"), use_container_width=True)
    st.subheader("Narrative signals")
    st.write(row["claim_narrative"])
    analysis = analyze_narrative(row["claim_narrative"])
    st.caption(f"Sentiment: {analysis['sentiment']} • Urgency: {analysis['urgency_score']:.0%}")
    st.write("Risk phrases:", ", ".join(analysis["risk_terms"]) or "None")
with right:
    if st.button("Run AI claim review", type="primary"):
        with st.spinner("Scoring claim and retrieving policy evidence..."):
            result = load_agent().review(record)
        a, b = st.columns(2)
        a.metric("Predicted loss", f"${result['predicted_loss']:,.0f}")
        b.metric("Investigation probability", f"{result['investigation_probability']:.1%}")
        st.subheader("Recommended action")
        st.write(result["recommended_action"])
        st.subheader("Policy evidence")
        for passage in result["policy_evidence"]:
            st.markdown(f"**{passage['citation']}**  \n{passage['text']}")
        st.subheader("Local SHAP explanations")
        exp1, exp2 = st.columns(2)
        with exp1:
            shap_chart(result["explanations"]["severity"]["contributions"], "Severity prediction drivers")
        with exp2:
            shap_chart(result["explanations"]["risk"]["contributions"], "Investigation-risk drivers")
