# ClaimIQ model card

## Intended use

ClaimIQ is a synthetic-data portfolio demonstration for claim triage, reserve planning, explainability, and evidence retrieval. It is not approved for production adjudication, underwriting, fraud determination, or any automated adverse action.

## Models

| Model | Target | Primary use | Reported metrics |
|---|---|---|---|
| Severity | Synthetic `actual_loss` | Reserve/prioritization signal | MAE, RMSE, R² |
| Investigation risk | Synthetic `investigation_flag` | Human-review prioritization | ROC-AUC, precision, recall, F1 |

Both models use a common `ColumnTransformer` for imputation, scaling, and categorical encoding. Metrics are generated on a deterministic holdout split and saved to `artifacts/metrics.json`.

## Explainability and monitoring

Local and global SHAP values communicate model feature influence in transformed feature space. They are non-causal and should be checked for stability and fairness before use. PSI monitoring compares incoming data and prediction distributions with training-reference data; 0.10 is a watch threshold and 0.25 is material drift.

## Limitations and safeguards

- All data and policy language are synthetic.
- A synthetic target cannot validate real-world predictive validity or fairness.
- Feature effects may reflect simulated data-generation choices.
- Human review remains mandatory for consequential decisions.
- Production use would require legal/compliance review, calibration, fairness testing, privacy controls, secure audit logging, and ongoing performance monitoring.
