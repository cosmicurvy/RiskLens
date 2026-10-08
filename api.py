from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
from kafka import KafkaProducer
import json
import os
from datetime import datetime

app = FastAPI(title="Real-Time UPI Fraud Detection API",
              description="Real-time transaction ingestion gateway.",
              version="1.0.0")

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

try:
    producer = KafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))
except Exception as e:
    print(f"Warning: Could not connect to Kafka broker. Error: {e}")
    producer = None

class TransactionEvent(BaseModel):
    transaction_id: str = Field(..., description="Unique ID for the transaction")
    user_id: str = Field(..., description="Unique ID for the user")
    merchant_id: str = Field(..., description="Unique ID for the merchant")
    merchant_category: str = Field(..., description="Category of the merchant")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="ISO 8601 timestamp")
    amount: float = Field(..., description="Transaction amount in INR")
    transaction_type: str = Field(..., description="e.g., P2P, P2M")
    upi_app: str = Field(..., description="e.g., GPay, PhonePe, Paytm")
    device_id: str = Field(..., description="Unique hardware ID of the user's phone")
    device_is_new_for_user: int = Field(..., description="1 if new device, 0 if known")
    device_distinct_users_seen_so_far: int = Field(..., description="Count of distinct users on this device")
    location: str = Field(..., description="City or Region")
    location_mismatch: int = Field(..., description="1 if location differs from usual, 0 otherwise")
    account_created_date: str = Field(..., description="ISO 8601 date string of account creation")

def push_to_kafka(topic: str, data: dict):
    """Helper function to send validated data to Kafka asynchronously."""
    if producer:
        producer.send(topic, value=data)
        producer.flush()
    else:
        print("Kafka producer not initialized.")

@app.post("/api/transactions")
async def ingest_transaction(event: TransactionEvent, background_tasks: BackgroundTasks):
    """Recieves a live transaction, validates its data types and queues it in Kafka."""
    event_dict = event.model_dump()
    # push to Kafka asynchronously
    background_tasks.add_task(push_to_kafka, 'transactions', event_dict)
    
    return {"status": "success",
            "message": "Transaction queued for fraud analysis.",
            "transaction_id": event_dict["transaction_id"]}  

    
@app.get("/health")
async def health_check():
    return {"status": "active", 
            "kafka_connected": producer is not None}


