# ClaimIQ interview demo (3–5 minutes)

1. **0:00–0:40 — Business framing.** Open Executive Overview. Explain that ClaimIQ prioritizes claims using a severity estimate and an investigation-support score, not automated adjudication. Point out exposure, queue volume, and loss concentration.
2. **0:40–1:50 — Individual claim.** Select a claim in Claim Review and run the analysis. Show the predicted reserve signal, review recommendation, and cited policy evidence. Emphasize the final decision stays with a qualified human.
3. **1:50–2:45 — Explainability.** Show the two local SHAP charts. Explain that the bars show model drivers for this record, not causality; global importance is available in Model Performance.
4. **2:45–3:35 — RAG quality.** Ask the Policy Assistant about collision or water coverage. Show citation-bearing answers and the 25-question offline evaluation.
5. **3:35–4:30 — Operational readiness.** Run Monitoring to show PSI alerts across shifted incoming data and explain local FAISS persistence plus optional SageMaker handoff.

**Business talking points:** faster triage, consistent evidence access, portfolio visibility, human oversight, and monitoring before operational decisions.

**Technical talking points:** scikit-learn pipelines prevent preprocessing drift; SHAP explains tree models; FAISS performs local vector retrieval; deterministic evaluation measures retrieval, answer, citation, and groundedness; PSI flags distribution shift; cloud scripts deliberately externalize credentials.
