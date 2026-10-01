"""ClaimIQ executive dashboard landing page."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from common import PINK_PALETTE, load_claims

st.set_page_config(page_title="ClaimIQ | Executive Overview", page_icon="📊", layout="wide")
st.title("ClaimIQ")
st.caption("AI-powered insurance claims intelligence")

df = load_claims()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total claims", f"{len(df):,}")
c2.metric("Reported loss exposure", f"${df.reported_amount.sum() / 1_000_000:.1f}M")
c3.metric("Investigation queue", f"{df.investigation_flag.sum():,}", f"{df.investigation_flag.mean():.1%} of claims")
c4.metric("Average loss", f"${df.actual_loss.mean():,.0f}")

left, right = st.columns(2)
with left:
    st.plotly_chart(px.histogram(df, x="actual_loss", color="incident_severity", nbins=50,
                                 color_discrete_sequence=PINK_PALETTE,
                                 title="Loss severity distribution"), use_container_width=True)
with right:
    by_type = df.groupby("claim_type", as_index=False)["actual_loss"].sum().sort_values("actual_loss")
    st.plotly_chart(px.bar(by_type, x="actual_loss", y="claim_type", orientation="h",
                           color="actual_loss", color_continuous_scale=["#fce7f3", "#ec4899", "#9d174d"],
                           title="Total actual loss by claim type"), use_container_width=True)

st.subheader("Portfolio operating view")
summary = df.groupby(["policy_type", "claim_type"], as_index=False).agg(
    claims=("claim_id", "count"), average_loss=("actual_loss", "mean"),
    investigation_rate=("investigation_flag", "mean"),
)
st.dataframe(summary, hide_index=True, use_container_width=True,
             column_config={"average_loss": st.column_config.NumberColumn(format="$%.0f"),
                            "investigation_rate": st.column_config.NumberColumn(format="%.1f%%")})
st.info("Use the pages in the sidebar to review individual claims, model performance, policy evidence, and monitoring alerts.")
