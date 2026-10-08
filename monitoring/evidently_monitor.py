import os
import pandas as pd
import sqlite3
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset
from evidently.future.datasets import Dataset

# configuration
DATABASE_PATH = "fraud_data.db"
TRAINING_DATA_PATH = "data/upi_transactions.csv"
TEST_DATA_PATH = "data/test_upi_transactions.csv"
REPORT_DIRECTORY = "monitoring/reports"
PRODUCTION_HOURS = 12000

# loading the data
train_df = pd.read_csv(TRAINING_DATA_PATH)
reference_df = train_df.sample(n=min(10000, len(train_df)), random_state=42)
current_df = pd.read_csv(TEST_DATA_PATH)

# features to monitor
monitoring_columns = ["amount", "merchant_category", "transaction_type", "upi_app", "location",
                      "device_distinct_users_seen_so_far", "device_is_new_for_user", "location_mismatch"]
prediction_columns = ["amount", "fraud_probability", "anomaly_score", "scenario", "decision"]

# create directories
os.makedirs(REPORT_DIRECTORY, exist_ok=True)

missing_columns = [col for col in monitoring_columns if col not in reference_df.columns]
if missing_columns:
    raise ValueError("Reference dataset is missing " 
                     f"columns: {missing_columns}")

reference_df = reference_df[monitoring_columns].copy()

# function to load production data
def load_production_data(database_path):
    connection = sqlite3.connect(database_path,)

    query = """SELECT amount, merchant_category,transaction_type, upi_app, location, device_distinct_users_seen_so_far,
            device_is_new_for_user, location_mismatch
            FROM raw_transactions
            WHERE timestamp >= datetime('now', ?)
            ORDER BY timestamp ASC"""

    df = pd.read_sql(query, connection, params=(f"-{PRODUCTION_HOURS} hours",))
    connection.close()
    return df

# function to load Prediction Data
def load_prediction_data():
    connection = sqlite3.connect(DATABASE_PATH)

    query = """SELECT
            timestamp, transaction_id, amount, fraud_probability, anomaly_score, scenario, decision
            FROM fraud_log
            WHERE timestamp >= datetime(
            'now',
            ?)
            ORDER BY timestamp ASC"""

    df = pd.read_sql_query(query, connection, params=(f"-{PRODUCTION_HOURS} hours",))
    connection.close()
    return df

# Validate Production Data
def validate_production_data(production_df):
    if production_df.empty:
        print("No production transactions available for monitoring.")
        return False

    missing_columns = [col for col in monitoring_columns if col not in production_df.columns]
    if missing_columns:
        raise ValueError("Production data is missing "
                         f"columns: {missing_columns}")
    return True

# Prepare Data Types
def prepare_raw_data(df):
    df = df.copy()
    numeric_columns = ["amount", "device_distinct_users_seen_so_far", "device_is_new_for_user", "location_mismatch"]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column],errors="coerce")

    categorical_columns = ["merchant_category", "transaction_type", "upi_app", "location"]

    for column in categorical_columns:
        if column in df.columns:
            df[column] = df[column].astype(str)
    return df

# Data Drift Report
def generate_data_drift_report(reference_df,production_df):
    print("Generating data drift report...")

    reference_df = prepare_raw_data(reference_df)
    production_df = prepare_raw_data(production_df)

    reference_dataset = Dataset.from_pandas(reference_df, data_definition=DataDefinition())
    production_dataset = Dataset.from_pandas(production_df, data_definition=DataDefinition())

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(production_dataset, reference_dataset)

    output_path = os.path.join(REPORT_DIRECTORY,"data_drift.html")
    snapshot.save_html(output_path)

    print(f"Data drift report saved to: "
          f"{output_path}")

# Prediction Distribution Monitoring
def create_prediction_distribution_report(production_df):
    if production_df.empty:
        print("No prediction data available.")
        return 
    
    print("Creating prediction monitoring dataset...")
    prediction_columns = ["amount", "fraud_probability", "anomaly_score", "scenario", "decision"]

    available_columns = [col for col in prediction_columns if col in production_df.columns]
    prediction_df = production_df[available_columns].copy()

    # save prediction window
    output_csv = os.path.join(REPORT_DIRECTORY,"latest_predictions.csv")
    prediction_df.to_csv(output_csv, index=False)

    print(f"Prediction data saved to: "
          f"{output_csv}")

    # Basic monitoring statistics
    print("\nPrediction Monitoring")
    print("-" * 50)
    if "fraud_probability" in prediction_df:
        print("Average fraud probability:", round(prediction_df["fraud_probability"].mean(), 4))
        print("Maximum fraud probability:", round(prediction_df["fraud_probability"].max(), 4))
    
    if "anomaly_score" in prediction_df:
        print("Average anomaly risk:", round(prediction_df["anomaly_score"].mean(), 4))
    
    if "decision" in prediction_df:
        print("\nDecision distribution:")
        print(prediction_df["decision"].value_counts(normalize=True))
    
    if "scenario" in prediction_df:
        print("\nScenario distribution:")
        print(prediction_df["scenario"].value_counts())

# Data Quality Checks
def run_basic_data_quality_checks(production_df):
    print("\nData Quality Checks")
    print("-" * 50)
    print(f"Rows: {len(production_df)}")

    # missing values
    missing_values = production_df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if missing_values.empty:
        print("Missing values: None")
    else:
        print("Missing values:")
        print(missing_values)

    # duplicate transactions
    if "transaction_id" in production_df:
        duplicate_count = production_df["transaction_id"].duplicated().sum()
        print("Duplicate transaction IDs:", duplicate_count)

    # negative amounts
    if "amount" in production_df:
        negative_amounts = (production_df["amount"] < 0).sum()
        print("Negative amounts:",negative_amounts)

# Main
def main():
    print("-" * 70)
    print("EVIDENTLY FRAUD DETECTION MONITORING")
    print("-" * 70)
    print(f"Monitoring window: last {PRODUCTION_HOURS} hour(s)")

    print(f"\nReference rows: {len(reference_df)}")

    # Production raw data
    production_df = load_production_data(DATABASE_PATH)

    print(f"Production rows: {len(production_df)}")

    if not validate_production_data(production_df):
        return 

    # Data quality
    run_basic_data_quality_checks(production_df)
    # Data drift
    generate_data_drift_report(reference_df=reference_df, production_df=production_df)
    # Prediction monitoring
    prediction_df = load_prediction_data()

    create_prediction_distribution_report(prediction_df)

    # Finished
    print("\nMonitoring completed successfully.")
    print(f"Reports available in: "
          f"{REPORT_DIRECTORY}")

if __name__ == "__main__":
    main()