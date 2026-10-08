import pandas as pd
from starlette.testclient import TestClient
import pytest
from datetime import datetime, timedelta
from api import app
from src.risk_engine import Decision, UPIRiskEngine
from src.preprocessing import convert_into_datetime_obj, drop_columns

client = TestClient(app)

# Testing the api.py
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert 'status' in response.json()
    assert response.json()['status'] == 'active'

def test_ingest_transaction_valid_payload():
    payload = {
        "transaction_id": "TXN00100015",
        "user_id": "U123",
        "merchant_id": "M456",
        "merchant_category": "Retail",
        "timestamp": datetime.now().isoformat(),
        "amount": 500.0,
        "transaction_type": "P2M",
        "upi_app": "GPay",
        "device_id": "DEV-999",
        "device_is_new_for_user": 0,
        "device_distinct_users_seen_so_far": 1,
        "location": "Mumbai",
        "location_mismatch": 0,
        "account_created_date": (datetime.now() - timedelta(days=30)).isoformat()
        }

    response = client.post('/api/transactions', json= payload)
    assert response.status_code == 200
    assert response.json()['status'] == 'success'
    assert response.json()['transaction_id'] == 'TXN00100015'


def test_ingest_transaction_missing_fields():
    payload = {
        "transaction_id": "TXN00100015",
        "user_id": "U123",
        "location_mismatch": 0}

    response = client.post('/api/transactions', json= payload)
    assert response.status_code == 422 # fastapi standard validation error


# testing risk_engine.py
@pytest.fixture
def risk_engine():
    return UPIRiskEngine(block_threshold=0.75, review_threshold=0.35, anomaly_threshold=0.65)

def test_risk_engine_allow(risk_engine):
    eval_result = risk_engine.evaluate(transaction_id="TXN-ALLOW",
                                       amount=500.0, fraud_prob=0.10, # Below review threshold
                                       raw_anomaly_score=0.20, # Normal behavior 
                                       predicted_scenario="Clean")
    assert eval_result.decision == Decision.ALLOW

def test_risk_engine_review(risk_engine):
    eval_result = risk_engine.evaluate(transaction_id="TXN-REVIEW",
                                       amount=1000.0, fraud_prob=0.45, # between 0.35 to 0.75
                                       raw_anomaly_score=0.10, 
                                       predicted_scenario="Suspicious Velocity")
    assert eval_result.decision == Decision.REVIEW

def test_risk_engine_block(risk_engine):
    eval_result = risk_engine.evaluate(transaction_id="TXN-ALLOW",
                                       amount=50000.0, fraud_prob=0.85, # exceeds 0.35 block threshold
                                       raw_anomaly_score=0.40, 
                                       predicted_scenario="Account Takeover")
    assert eval_result.decision == Decision.BLOCK

# testing preprocessing.py
def test_convert_into_datetime_obj():
    df = pd.DataFrame({"timestamp": ["2026-09-29T14:00:00", "2026-09-29T15:00:00"]})
    df['timestamp'] = convert_into_datetime_obj(df, 'timestamp')
    assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])

def test_drop_columns():
    df = pd.DataFrame({
        "transaction_id": ["T1"],
        "user_id": ["U1"],
        "timestamp": ["2026-09-29"],
        "account_created_date": ["2026-01-01"],
        "amount": [500]
    })

    df_clean = drop_columns(df.copy())

    assert "transaction_id" not in df_clean.columns
    assert "user_id" not in df_clean.columns
    assert "amount" in df_clean.columns