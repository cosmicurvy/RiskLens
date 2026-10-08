import pandas as pd
import time
import json
import os
from kafka import KafkaProducer

DATA_PATH = "data/streaming_test.csv"
TOPIC = "transactions"
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

producer = KafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

df = pd.read_csv("data/test_upi_transactions.csv")
df.drop(columns=['is_fraud', 'fraud_scenario'], inplace=True)

for idx, row in df.iterrows():
    event = row.to_dict()
    producer.send(TOPIC, value=event)

    time.sleep(0.1)

producer.flush()
