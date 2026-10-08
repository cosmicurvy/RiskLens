import sqlite3
import os
import json
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from src.artifacts import load_encoders, load_models
from src.risk_engine import UPIRiskEngine, Decision
from src.preprocessing import (convert_into_datetime_obj, extract_features, drop_columns)

TOPIC = "transactions"
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

ENCODER_PATH = "models/ordinal_encoder.joblib"
TARGET_ENCODER_PATH = "models/target_encoder.joblib"

FRAUD_MODEL_PATH = "models/xgb_fraud_model.joblib"
SCENARIO_MODEL_PATH = "models/xgb_scenario_model.joblib" 
ISO_FOREST_PATH = "models/iso_forest.joblib"

DATABASE_PATH = "fraud_data.db"

APPROVED_TOPIC = "tx-approved"
REVIEW_TOPIC = "tx-review"
BLOCKED_TOPIC = "tx-blocked"

# Load encoders and models
encoder, target_encoder = load_encoders(ENCODER_PATH, TARGET_ENCODER_PATH)
fraud_model, scenario_model, iso_forest = load_models(FRAUD_MODEL_PATH, SCENARIO_MODEL_PATH, ISO_FOREST_PATH)
print("Encoders and Models loaded successfully.")

# Database Setup
try:
    db_conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    cursor = db_conn.cursor()
    
    # Final Risk evaluation log 
    cursor.execute("""CREATE TABLE IF NOT EXISTS fraud_log (
    transaction_id TEXT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    amount REAL,
    fraud_probability REAL,
    anomaly_score REAL,
    risk_score REAL,
    scenario TEXT,
    decision TEXT,
    triggers TEXT,
    action_notes TEXT,
    device_is_new INTEGER,
    location_mismatch INTEGER)"""
                   )

    
    # Raw transactions used for historical context
    cursor.execute("""CREATE TABLE IF NOT EXISTS raw_transactions (
        transaction_id TEXT PRIMARY KEY,
        user_id TEXT,
        merchant_id TEXT,
        merchant_category TEXT,
        timestamp DATETIME,
        amount REAL,
        transaction_type TEXT,
        upi_app TEXT,
        device_id TEXT,
        device_is_new_for_user INTEGER,
        device_distinct_users_seen_so_far INTEGER,
        location TEXT,
        location_mismatch INTEGER,
        account_created_date DATETIME)""")
    
    # # Helpful for user/time-window lookups
    cursor.execute("""
                   CREATE INDEX IF NOT EXISTS idx_raw_user_timestamp
                   ON raw_transactions(user_id, timestamp)""")
    
    db_conn.commit()
    print("SQLite database created and connected successfully.")

except Exception as e:
    print(f"Database setup failed: {e}")
    raise SystemExit(1)

# Initialize Risk engine
risk_engine = UPIRiskEngine(block_threshold=0.75, review_threshold=0.35, anomaly_threshold=0.65,
                            severe_anomaly_threshold=0.80, high_amount_threshold=30000)
print("Risk engine initialized.")

# Kafka Consumer
consumer = KafkaConsumer(TOPIC,
                         bootstrap_servers= KAFKA_BOOTSTRAP_SERVERS,
                         auto_offset_reset="latest",
                         enable_auto_commit = True,
                         value_deserializer=lambda value: json.loads(value.decode("utf-8")))

# Kafka Producer
producer = KafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                         value_serializer=lambda v: json.dumps(v).encode("utf-8"))

print("Risk engine is active.")

# Helper Functions
def get_historical_transactions(user_id: str, connection: sqlite3.Connection) -> pd.DataFrame:
    """
    Retrieve the user's transactions from the previous hour.
    These transactions are used to create contextual/time-based
    features for the current transaction.
    """

    query = """SELECT * FROM raw_transactions
    WHERE user_id = ?
    AND timestamp >= datetime('now', '-1 hour')
    ORDER BY timestamp ASC"""
    return pd.read_sql(query, connection,params=(user_id,))

def store_raw_transaction(event: dict, connection: sqlite3.Connection):
    """Store the current transaction so that future transactions
    can use it as historical context."""

    query = """INSERT OR REPLACE INTO raw_transactions (transaction_id, user_id, merchant_id, 
    merchant_category, timestamp, amount, transaction_type, upi_app, device_id, device_is_new_for_user,
    device_distinct_users_seen_so_far, location, location_mismatch, account_created_date)
    
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

    values = (event.get("transaction_id"),
              event.get("user_id"),
              event.get("merchant_id"),
              event.get("merchant_category"),
              event.get("timestamp"),
              event.get("amount"),
              event.get("transaction_type"),
              event.get("upi_app"),
              event.get("device_id"),
              int(event.get("device_is_new_for_user", 0)),
              int(event.get("device_distinct_users_seen_so_far", 0)),
              event.get("location"),
              int(event.get("location_mismatch", 0)),
              event.get("account_created_date"))

    connection.execute(query, values)
    connection.commit()


def log_evaluation(evaluation, amount: float, connection: sqlite3.Connection, event: dict):
    """Store the final risk evaluation in fraud_log."""

    query = """
        INSERT OR REPLACE INTO fraud_log (
        transaction_id,
            amount,
            fraud_probability,
            anomaly_score,
            risk_score,
            scenario,
            decision,
            triggers,
            action_notes,
            device_is_new,
            location_mismatch
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    values = (evaluation.transaction_id, amount, float(evaluation.fraud_probability),
              float(evaluation.anomaly_score), float(evaluation.risk_score), evaluation.predicted_scenario, 
              evaluation.decision.value, json.dumps(evaluation.triggers), evaluation.action_notes,
              int(bool(event.get("device_is_new_for_user", 0))), int(bool(event.get("location_mismatch", 0)))
              )

    connection.execute(query, values)
    connection.commit()

def publish_decision(evaluation, producer: KafkaProducer):
    """
    Publish the final decision to the appropriate Kafka topic.
    """

    output_payload = {"transaction_id": evaluation.transaction_id,
                      "decision": evaluation.decision.value,
                      "risk_score": round(evaluation.risk_score, 2),
                      "fraud_probability": round(evaluation.fraud_probability, 4),
                      "anomaly_score": round(evaluation.anomaly_score, 4),
                      "scenario": evaluation.predicted_scenario,
                      "triggers": evaluation.triggers,
                      "action_notes": evaluation.action_notes}

    if evaluation.decision == Decision.BLOCK:
        topic = BLOCKED_TOPIC
        print(f"[BLOCK] {evaluation.transaction_id} | "
              f"Score: {evaluation.risk_score:.1f} | "
              f"Triggers: {evaluation.triggers}")

    elif evaluation.decision == Decision.REVIEW:
        topic = REVIEW_TOPIC
        print(f"[REVIEW] {evaluation.transaction_id} | "
              f"Score: {evaluation.risk_score:.1f} | "
              f"Triggers: {evaluation.triggers}")

    else:
        topic = APPROVED_TOPIC
        print(f"[ALLOW] {evaluation.transaction_id} | "
              f"Score: {evaluation.risk_score:.1f} | "
              f"Fast-track approved")

    producer.send(topic, value=output_payload)
    return output_payload



# Streaming Loop
for message in consumer:
    try:
        # Read Kafka event
        event = message.value

        txn_id = event.get("transaction_id", "UNKNOWN")
        user_id = event.get("user_id")
        amount = float(event.get('amount', 0.0))

        print(f"\nProcessing Transaction: {txn_id}")

        # Retrieve historical transaction
        historical_data = get_historical_transactions(user_id=user_id, connection=db_conn)
        
        # Create live dataframe
        df_live = pd.DataFrame([event])

        # combine historical + current transaction
        if not historical_data.empty:
            df_context = pd.concat([historical_data, df_live], ignore_index=True)
        else:
            df_context = df_live.copy()

        # Convert datetime features
        df_context['timestamp'] = convert_into_datetime_obj(df_context, feature='timestamp')
        df_context['account_created_date'] = convert_into_datetime_obj(df_context, feature='account_created_date')

        # Ensures contextual flags are integers
        df_context['device_is_new_for_user'] = df_context['device_is_new_for_user'].astype(int)
        df_context['location_mismatch'] = df_context['location_mismatch'].astype(int)
        
        # feature engineering
        df_context = extract_features(df_context)
        
        # select only current transaction
        df_live_features = df_context.iloc[[-1]].copy()

        # remove columns that are not used by models
        df_live_features = drop_columns(df_live_features)
 
        # encode categorical features
        cols_to_encode= ['merchant_category', 'transaction_type', 'upi_app', 'location']
        df_live_features[cols_to_encode] = encoder.transform(df_live_features[cols_to_encode])
        
        # target encoding on merchant_id and device_id
        df_live_features[['merchant_id','device_id']] = target_encoder.transform(df_live_features[['merchant_id','device_id']])
       
        # fraud model prediction
        fraud_prob= float(fraud_model.predict_proba(df_live_features)[0][1])
        # anomaly model prediction
        raw_anomaly = float(iso_forest.decision_function(df_live_features)[0])
        # scenario prediction
        if fraud_prob > 0.20 or raw_anomaly < -0.10:
            scenario_prediction = scenario_model.predict(df_live_features)[0]
            
            mapped_fs = {0: "velocity_abuse", 1: "suspicious_device", 2: "account_takeover", 3: "unusual_amount"}

            final_scenario = mapped_fs.get(int(scenario_prediction), "Unknown Scenario")
        else:
            final_scenario = "Clean"

        # Risk evaluation
        evaluation = risk_engine.evaluate(transaction_id=txn_id, amount=amount, fraud_prob=fraud_prob,
                                          raw_anomaly_score=raw_anomaly, predicted_scenario=final_scenario,
                                          device_is_new=bool(event.get("device_is_new_for_user", 0)),
                                          location_mismatch=bool(event.get("location_mismatch", 0)))

        # Publish ALLOW/REVIEW/BLOCK
        publish_decision(evaluation=evaluation, producer=producer)

        # Log final evaluation
        log_evaluation(evaluation=evaluation, amount=amount, connection=db_conn, event = event)

        # Store current transaction for future context
        store_raw_transaction(event=event, connection=db_conn)
        print(f"Transaction {txn_id} processed successfully.")

    except Exception as e:
        print(f"Error processing transaction: {e}")
        continue # continue processing the next kafka event

# Cleanup
producer.flush()
consumer.close()
db_conn.close()
