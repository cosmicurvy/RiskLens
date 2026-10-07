import pandas as pd
import time
import json
from kafka import KafkaProducer

DATA_PATH = "data/streaming_test.csv"
TOPIC = "transactions"
KAFKA_SERVER = "localhost:9092"

producer = KafkaProducer(bootstrap_servers=KAFKA_SERVER,
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

df = pd.read_csv("data/test_upi_transactions.csv")
df.drop(columns=['is_fraud', 'fraud_scenario'], inplace=True)

for idx, row in df.iterrows():
    event = row.to_dict()
    producer.send(TOPIC, value=event)

    time.sleep(0.1)

producer.flush()
