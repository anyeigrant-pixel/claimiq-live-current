CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    age INTEGER NOT NULL,
    customer_tenure_years REAL NOT NULL,
    prior_claims INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS policies (
    policy_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    policy_type TEXT NOT NULL,
    annual_premium REAL NOT NULL,
    deductible REAL NOT NULL,
    policy_tenure_months INTEGER NOT NULL,
    FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id INTEGER PRIMARY KEY,
    policy_id INTEGER NOT NULL,
    claim_type TEXT NOT NULL,
    incident_severity TEXT NOT NULL,
    vehicle_age INTEGER NOT NULL,
    reported_amount REAL NOT NULL,
    days_to_report INTEGER NOT NULL,
    police_report INTEGER NOT NULL,
    witness_count INTEGER NOT NULL,
    actual_loss REAL NOT NULL,
    investigation_flag INTEGER NOT NULL,
    FOREIGN KEY(policy_id) REFERENCES policies(policy_id)
);

CREATE TABLE IF NOT EXISTS model_predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id INTEGER NOT NULL,
    predicted_loss REAL,
    investigation_probability REAL,
    model_version TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(claim_id) REFERENCES claims(claim_id)
);
