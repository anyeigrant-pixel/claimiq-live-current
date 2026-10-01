from __future__ import annotations
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from src.nlp.narrative import build_narratives

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
DB = ROOT / "claimiq.db"
SCHEMA = ROOT / "sql" / "schema.sql"


def generate(n: int = 12000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    claim_id = np.arange(1, n + 1)
    customer_id = rng.integers(1, n // 2 + 1, size=n)
    policy_id = claim_id
    age = rng.integers(18, 81, size=n)
    customer_tenure = np.round(rng.gamma(2.2, 2.0, size=n), 1)
    prior_claims = rng.poisson(0.8, size=n)
    policy_type = rng.choice(["Auto", "Home"], size=n, p=[0.72, 0.28])
    annual_premium = np.round(rng.normal(1550, 430, size=n).clip(450, 4200), 2)
    deductible = rng.choice([250, 500, 1000, 1500, 2500], size=n, p=[.08,.35,.36,.14,.07])
    policy_tenure_months = rng.integers(1, 121, size=n)
    claim_type = np.where(
        policy_type == "Auto",
        rng.choice(["Collision", "Comprehensive", "Liability", "Glass"], size=n, p=[.43,.24,.25,.08]),
        rng.choice(["Water", "Wind", "Fire", "Theft"], size=n, p=[.36,.29,.12,.23])
    )
    severity = rng.choice(["Minor", "Moderate", "Major"], size=n, p=[.48,.37,.15])
    vehicle_age = np.where(policy_type == "Auto", rng.integers(0, 21, size=n), 0)
    days_to_report = rng.integers(0, 31, size=n)
    police_report = rng.binomial(1, np.where(policy_type == "Auto", .64, .28), size=n)
    witness_count = rng.poisson(0.7, size=n).clip(0, 5)

    sev_mult = pd.Series(severity).map({"Minor": 1.0, "Moderate": 2.4, "Major": 6.0}).to_numpy()
    type_mult = pd.Series(claim_type).map({
        "Collision":1.5,"Comprehensive":1.1,"Liability":1.8,"Glass":0.35,
        "Water":1.4,"Wind":1.25,"Fire":4.5,"Theft":1.9
    }).to_numpy()
    base = 1100 * sev_mult * type_mult
    tenure_discount = np.clip(1 - 0.01 * customer_tenure, 0.82, 1.0)
    prior_factor = 1 + 0.10 * prior_claims
    noise = rng.lognormal(mean=0, sigma=0.32, size=n)
    actual_loss = np.round((base + annual_premium * .55 + vehicle_age * 70) * tenure_discount * prior_factor * noise, 2)
    reported_amount = np.round(actual_loss * rng.normal(1.08, .18, size=n).clip(.65, 1.75), 2)

    logit = (
        -4.10
        + 1.25 * (prior_claims >= 2)
        + 1.75 * (policy_tenure_months <= 3)
        + 0.085 * days_to_report
        + 1.45 * (reported_amount > np.quantile(reported_amount, .88))
        + 0.75 * (reported_amount > actual_loss * 1.28)
        - 0.80 * police_report
        - 0.35 * np.minimum(witness_count, 2)
    )
    fraud_prob = 1 / (1 + np.exp(-logit))
    investigation_flag = rng.binomial(1, fraud_prob)

    data = pd.DataFrame({
        "claim_id":claim_id,"customer_id":customer_id,"policy_id":policy_id,
        "age":age,"customer_tenure_years":customer_tenure,"prior_claims":prior_claims,
        "policy_type":policy_type,"annual_premium":annual_premium,"deductible":deductible,
        "policy_tenure_months":policy_tenure_months,"claim_type":claim_type,
        "incident_severity":severity,"vehicle_age":vehicle_age,"reported_amount":reported_amount,
        "days_to_report":days_to_report,"police_report":police_report,"witness_count":witness_count,
        "actual_loss":actual_loss,"investigation_flag":investigation_flag
    })
    data["claim_narrative"] = build_narratives(data, seed=seed)
    return data


def persist(df: pd.DataFrame) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW / "synthetic_claims.csv", index=False)
    if DB.exists(): DB.unlink()
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA.read_text())
    customers = df[["customer_id","age","customer_tenure_years","prior_claims"]].drop_duplicates("customer_id")
    policies = df[["policy_id","customer_id","policy_type","annual_premium","deductible","policy_tenure_months"]]
    claims = df[["claim_id","policy_id","claim_type","incident_severity","vehicle_age","reported_amount","days_to_report","police_report","witness_count","actual_loss","investigation_flag"]]
    customers.to_sql("customers", con, if_exists="append", index=False)
    policies.to_sql("policies", con, if_exists="append", index=False)
    claims.to_sql("claims", con, if_exists="append", index=False)
    con.close()


if __name__ == "__main__":
    data = generate()
    persist(data)
    print(f"Generated {len(data):,} synthetic claims at {RAW / 'synthetic_claims.csv'}")
    print(f"SQLite database: {DB}")
