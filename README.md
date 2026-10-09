# RiskLens: Real-Time UPI Fraud Detection & Risk Analytics Platform 

A real-time upi fraud detection system that combines machine learning, Kafka streaming, and risk-based decision-making to analyze financial transactions and flag potentially fraudulent activity.

## Overview

This project simulates a real-time transaction monitoring pipeline. Incoming transactions are ingested through FastAPI, streamed through Apache Kafka, and analyzed using three machine learning models. A risk engine combines model predictions and contextual signals to determine whether a transaction should be allowed, reviewed, or blocked.

The platform also includes a Streamlit dashboard for monitoring transaction activity and Evidently AI for detecting changes in production data.

## Key Features

* **Real-time transaction ingestion:** FastAPI validates incoming transactions and publishes them to Kafka.
* **Fraud detection:** An XGBoost classifier estimates the probability of fraud.
* **Fraud scenario classification:** A second XGBoost model identifies potential fraud scenarios.
* **Anomaly detection:** Isolation Forest identifies unusual transaction patterns.
* **Risk-based decisions:** A custom risk engine combines model outputs and contextual signals to assign risk scores and recommend actions.
* **Transaction logging:** SQLite stores transaction records, predictions, risk scores, and decisions.
* **Interactive dashboard:** Streamlit displays transaction activity, risk assessments, and fraud-related insights.
* **Data drift monitoring:** Evidently AI helps identify changes in production data distributions.
* **Containerization and CI/CD:** Docker and GitHub Actions support reproducability and automated checks.

## How It Works? 

1. A transaction is submitted to the FastAPI endpoint.
2. FastAPI validates the input and publishes the event to Kafka.
3. A Kafka consumer processes the transaction and prepares the required model features.
4. The fraud classifier, scenario classifier, and anomaly detector generate their respective outputs.
5. The risk engine evaluates the results and produces a final decision: `ALLOW`, `REVIEW`, or `BLOCK`.
6. Transaction details and risk assessments are stored in SQLite and displayed on the Streamlit dashboard.
7. Evidently AI monitors production data for potential distribution shifts.

## Technology Stack

**Languages:** Python, SQL

**Machine Learning:** XGBoost, Isolation Forest, scikit-learn, pandas, NumPy, Seaborn, Matplotlib

**Streaming and APIs:** Apache Kafka, FastAPI

**Storage and Dashboard:** SQLite, Streamlit

**Monitoring:** Evidently AI

**Deployment and Automation:** Docker, Docker Compose, GitHub Actions

## Getting Started

### Prerequisites

* Python 3.11 or a compatible version for your dependencies
* Apache Kafka
* Docker and Docker Compose (for containerized execution)
* Git

### Installation

Clone the repository:

```bash
git clone https://github.com/cosmicurvy/RiskLens.git
cd https://github.com/cosmicurvy/RiskLens.git
```

Create and activate a virtual environment:

```bash
python -m venv fraud
```

On macOS or Linux:

```bash
source fraud/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Configure the Kafka bootstrap server and any required environment variables for your local setup.

### Running the Application

Start Kafka and ensure the `transactions` topic is available.

Start the FastAPI application:

```bash
uvicorn api:app --reload
```

Start the Kafka consumer in a separate terminal using your project's consumer module.

Launch the Streamlit dashboard:

```bash
streamlit run dashboard/streamlit_app.py
```

The FastAPI interactive documentation is available at:

`http://127.0.0.1:8000/docs`

The Streamlit dashboard is typically available at:

`http://localhost:8501`

**Note:** Run these commands from the appropriate project directory. If using Docker Compose, use the service configuration and Kafka address appropriate to the Docker network.

## Limitations

* The transaction data is synthetic and does not establish real-world upi fraud detection data.
* The risk engine uses configurable thresholds that require calibration and validation before real-world deployment.
* Production use would require additional security, access control, reliability, and operational safeguards.