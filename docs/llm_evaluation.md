# Policy RAG evaluation

The local policy assistant retrieves from a persisted FAISS index and returns the highest-ranked synthetic policy passage as a deterministic, citation-bearing answer. This design makes evaluation repeatable without an external LLM.

## Benchmark

`src/rag/policy_qa.json` contains **25** synthetic insurance-policy questions. Each record specifies a question, expected source file, and expected grounded phrase. Run:

```bash
python -m src.rag.evaluate
```

The report is written to `artifacts/rag_evaluation.json` and rendered in the Policy Assistant page.

## Metrics

| Metric | Definition |
|---|---|
| Retrieval hit rate | Expected source appears in top-k retrieved documents |
| Answer correctness | Returned answer contains the benchmark's expected policy phrase |
| Citation accuracy | Returned citation names the expected source |
| Groundedness | Returned answer is exactly a retrieved passage |
| Latency | Local retrieval wall-clock time per benchmark question |

For a production LLM, add expert review, citation entailment checks, adversarial questions, abstention scoring, PII checks, and versioned benchmark governance. The current results measure deterministic retrieval/answer behavior only, not general language-model quality.
