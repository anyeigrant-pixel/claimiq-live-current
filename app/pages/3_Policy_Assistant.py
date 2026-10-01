from __future__ import annotations

import streamlit as st

from common import load_report
from src.rag.evaluate import evaluate
from src.rag.retriever import retrieve

st.set_page_config(page_title="ClaimIQ | Policy Assistant", layout="wide")
st.title("Policy & RAG Assistant")
st.caption("Answers are grounded in the local policy corpus. Every response includes citations.")
question = st.text_input("Ask a policy question", "Does comprehensive coverage include theft?")
if st.button("Retrieve policy answer", type="primary"):
    passages = retrieve(question)
    st.markdown(f"**Answer:** {passages[0].text}")
    st.markdown(f"**Citation:** {passages[0].citation}")
    with st.expander("Retrieved evidence"):
        for passage in passages:
            st.markdown(f"**{passage.citation}** (similarity {passage.score:.2f})  \n{passage.text}")

st.subheader("Evaluation report")
if st.button("Run 25-question RAG evaluation"):
    with st.spinner("Evaluating retrieval and grounded answers..."):
        st.session_state["rag_report"] = evaluate()
report = st.session_state.get("rag_report", load_report("rag_evaluation.json"))
if report:
    cols = st.columns(5)
    for column, (name, value) in zip(cols, [
        ("Questions", report["question_count"]), ("Retrieval hit rate", report["retrieval_hit_rate"]),
        ("Answer correctness", report["answer_correctness"]), ("Citation accuracy", report["citation_accuracy"]),
        ("Groundedness", report["groundedness"]),
    ], strict=True):
        column.metric(name, f"{value:.1%}" if isinstance(value, float) else value)
    st.caption(f"Average retrieval latency: {report['average_latency_ms']:.2f} ms")
