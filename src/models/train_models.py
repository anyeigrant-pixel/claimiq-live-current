from __future__ import annotations
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "synthetic_claims.csv"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

FEATURES = [
    "age","customer_tenure_years","prior_claims","policy_type","annual_premium","deductible",
    "policy_tenure_months","claim_type","incident_severity","vehicle_age","reported_amount",
    "days_to_report","police_report","witness_count"
]
CAT = ["policy_type","claim_type","incident_severity"]
NUM = [c for c in FEATURES if c not in CAT]


def preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUM),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
    ])


def main():
    df = pd.read_csv(DATA)
    X = df[FEATURES]
    X_train, X_test, y_train, y_test = train_test_split(X, df["actual_loss"], test_size=.2, random_state=42)
    sev = Pipeline([("prep", preprocessor()), ("model", HistGradientBoostingRegressor(max_iter=260, learning_rate=.06, max_leaf_nodes=31, random_state=42))])
    sev.fit(X_train, y_train)
    pred = sev.predict(X_test)
    severity_metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(mean_squared_error(y_test, pred) ** .5),
        "r2": float(r2_score(y_test, pred)),
    }

    X_train, X_test, y_train, y_test = train_test_split(X, df["investigation_flag"], test_size=.2, stratify=df["investigation_flag"], random_state=42)
    risk = Pipeline([("prep", preprocessor()), ("model", RandomForestClassifier(n_estimators=280, max_depth=10, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1))])
    risk.fit(X_train, y_train)
    proba = risk.predict_proba(X_test)[:,1]
    cls = (proba >= .5).astype(int)
    risk_metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "precision": float(precision_score(y_test, cls, zero_division=0)),
        "recall": float(recall_score(y_test, cls, zero_division=0)),
        "f1": float(f1_score(y_test, cls, zero_division=0)),
    }

    joblib.dump(sev, ART / "severity_model.joblib")
    joblib.dump(risk, ART / "risk_model.joblib")
    metrics = {"severity": severity_metrics, "risk": risk_metrics}
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
