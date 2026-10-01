# ClaimIQ architecture

ClaimIQ uses a local-first architecture that keeps the original training and inference path intact while adding explainability, retrieval quality controls, and monitoring.

```text
Synthetic CSV / SQLite
  -> sklearn ColumnTransformer + models -> joblib artifacts + metrics
  -> SHAP service ----------------------> local/global explanation views
  -> incoming synthetic batch -> inference -> feature/prediction PSI report

Policy Markdown -> chunking -> TF-IDF dense embeddings -> local FAISS index
  -> cited evidence -> claim-review agent / policy assistant
  -> 25-question benchmark -> persisted RAG evaluation report
```

## Design decisions

- **Reproducible local path:** `src/models/train_models.py` remains the canonical trainer and produces stable Joblib artifacts.
- **Explainability:** SHAP runs against the transformed model feature space, including one-hot encoded fields. UI language explicitly limits interpretations to model behavior.
- **Retrieval:** a persisted FAISS index stores local dense TF-IDF vectors. This avoids credentials and paid APIs while retaining a swappable vector-store boundary for a sentence-transformer or managed index.
- **Narrative embeddings:** NLP benchmarks use an explicitly configured, pre-downloaded SentenceTransformer when `CLAIMIQ_SENTENCE_TRANSFORMER_PATH` is set; otherwise an offline dense TF-IDF/SVD embedding is used. This prevents hidden network access during local demonstrations.
- **Observability:** monitoring calculates PSI for numerical and categorical model features plus both prediction distributions. A PSI of 0.25 or greater is surfaced as a material-drift alert.
- **Safety:** the review agent recommends routing only; it does not approve, deny, or label claims as fraud.

## Cloud evolution

Use encrypted S3 for curated inputs and artifacts, SageMaker Processing for transformations, Training for model fitting, Model Registry for approved packages, and real-time/batch endpoints for inference. Map `monitoring_report.json` to Model Monitor and CloudWatch alarms. Policy vectors can move to an approved managed vector store, retaining source metadata and citations.
