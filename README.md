# ClaimIQ — Explainable Insurance Claims Intelligence

ClaimIQ is a portfolio-ready, end-to-end insurance analytics platform. It turns synthetic claims into explainable severity and investigation-review signals, retrieves cited policy evidence, measures RAG quality, and watches incoming data for drift. It is intentionally **decision support**, not automated claim adjudication.

## Business value

| Audience | Outcome |
|---|---|
| Claims leaders | Portfolio exposure, queue volume, and loss-concentration KPIs |
| Adjusters | A single claim view with reserve signal, review recommendation, drivers, and policy evidence |
| Model risk teams | Reproducible training metrics, SHAP explanations, RAG evaluation, and PSI monitoring |
| Platform teams | Local-first artifacts with a documented SageMaker migration path |

## Capabilities

- Synthetic claims generation with normalized SQLite persistence.
- Scikit-learn severity regression and investigation-risk classification pipelines.
- Local and global SHAP explanations for both models.
- Five-page Streamlit experience: Executive Overview, Claim Review, Model Performance, Policy Assistant, and Monitoring.
- Local, persisted FAISS vector retrieval over synthetic policy documents with citations on every assistant response.
- A deterministic, 25-question RAG evaluation covering retrieval hit rate, answer correctness, citation accuracy, groundedness, and latency.
- Synthetic incoming-batch monitoring with feature and prediction PSI alerts.
- Synthetic claim-narrative classification, severity-from-text, named-entity and high-risk phrase extraction, urgency scoring, topic discovery, and similar-claim retrieval.
- Baseline TF-IDF/logistic-regression text classifiers, SentenceTransformer-ready embedding comparison with a local dense fallback, and a combined structured-plus-text severity benchmark.
- Credential-free AWS SageMaker upload, registry, and endpoint scaffolding.

## Architecture

```text
Synthetic claims -> sklearn preprocessing -> severity / investigation models
       |                         |                    |
       v                         v                    v
 SQLite + CSV              SHAP explanations     prediction monitoring (PSI)
                                                          |
Synthetic policy Markdown -> TF-IDF embeddings -> FAISS -> cited policy assistant
                                                          |
                                             Streamlit executive dashboard
```

Detailed design: [architecture](docs/architecture.md), [model card](docs/model_card.md), [RAG evaluation](docs/llm_evaluation.md), and [SageMaker scaffold](docs/sagemaker.md).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.data.generate_data
python -m src.models.train_models
python -m src.rag.evaluate
python -m src.monitoring.drift
streamlit run app/streamlit_app.py
```

Run the test suite:

```bash
pytest -q
```

## Repository map

```text
app/                    Streamlit landing page and dedicated business pages
src/models/             Training, inference, and SHAP explanation service
src/rag/                FAISS retrieval, corpus evaluation, QA benchmark
src/monitoring/         PSI-based synthetic incoming data and prediction monitoring
scripts/                Optional SageMaker integration entry points
docs/                   Architecture, governance, evaluation, and cloud guidance
tests/                  Data, inference, retrieval, evaluation, and monitoring tests
```

## Responsible-use guardrails

All data, policies, labels, and metrics are synthetic. The investigation-risk output identifies claims that may benefit from human verification; it is not a fraud determination. Explanations describe model behavior, are not causal claims, and should never replace qualified human judgment, governance review, or regulatory controls.

## AWS SageMaker scaffold

Copy `.env.example` to `.env`, set values through a secure environment or secret manager, and review [docs/sagemaker.md](docs/sagemaker.md). No credentials are stored or required for local use.

For an optional local transformer comparison, set `CLAIMIQ_SENTENCE_TRANSFORMER_PATH` to an already-downloaded SentenceTransformer directory. ClaimIQ otherwise remains offline and uses dense TF-IDF/SVD embeddings.

## Interview walkthrough

Use [DEMO.md](DEMO.md) for a concise 3–5 minute product and technical walkthrough.
