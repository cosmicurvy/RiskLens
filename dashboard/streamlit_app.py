import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import sqlite3

# Page configuration
st.set_page_config(page_title="Real-Time Fraud Detection & Risk Analytics Platform",
                   layout='wide',
                   initial_sidebar_state='expanded')

DATABASE_PATH = "fraud_data.db"

# Customer CSS
st.markdown(
    """
    <style>

    .metric-card {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.2);
        text-align: center;
    }

    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
    )

# Database functions
@st.cache_data(ttl=5) # refreshes every five seconds to pull new kafka events
def load_live_data(hours=24, limit = 5000):
    """Load processed transactions from fraud_log and data is refreshed every 5 seconds"""
    try:
        connection = sqlite3.connect(DATABASE_PATH)
        query = """
            SELECT *
            FROM fraud_log
            WHERE timestamp >= datetime('now', ?)
            ORDER BY timestamp DESC
            LIMIT ?
        """

        df = pd.read_sql(query, connection, params=(f"-{hours} hours", limit))

        connection.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        return df
    
    except Exception as e:
        st.error(f"Database Connection Error {e}")
        return pd.DataFrame()

@st.cache_data(ttl=5) 
def load_transaction_context(transaction_id):
    """Retreive raw transaction/context information."""
    try:
        connection = sqlite3.connect(DATABASE_PATH)
        query = """
            SELECT *
            FROM raw_transactions
            WHERE transaction_id = ?
        """

        df = pd.read_sql(query, connection, params=(transaction_id,))
        connection.close()

        return df
    
    except Exception as e:
        return pd.DataFrame()


# Helper Functions
def format_percentage(value):
    if pd.isna(value):
        return "N/A"
    return f"{value*100:.1f}%"

def decision_color(decision):
    if decision == "BLOCK":
        return "#ff4b4b"
    elif decision == "REVIEW":
        return "#ffa421"
    return "#00c04b"

def decision_icon(decision):
    if decision == "BLOCK":
        return "🚫"
    elif decision == "REVIEW":
        return "⚠️"
    return "✅"

def safe_json_load(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except Exception:
        return [str(value)]

def display_context_signal(value):
    if bool(value):
        return "YES"
    return "NO"


# HEader
st.title("Real-Time Fraud Detection & Risk Analytics Platform")
st.caption("Kafka → ML Models → Risk Engine → Decision Monitoring")

# Sidebar navigation & filters
st.sidebar.title("RiskLens")
st.sidebar.markdown("---")

view_mode = st.sidebar.radio("Navigation", ["Overview", "Transaction Inspector", "Risk Monitoring","Model Insights"])

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

hours_filter = st.sidebar.selectbox("Time Window",[1, 6, 12, 24],index=3)

min_amount = st.sidebar.number_input("Minimum Amount (₹)", min_value=0.0, value=0.0, step=500.0)

status_filter = st.sidebar.multiselect("Decision Status",["ALLOW","REVIEW","BLOCK"],
                                       default=["ALLOW","REVIEW","BLOCK"])

# loading the data accroding to selected time-window
df = load_live_data(hours=hours_filter, limit=6000)

# filter data
if not df.empty:
    filtered_df = df[(df["amount"] >= min_amount) & (df["decision"].isin(status_filter))].copy()
else:
    filtered_df = pd.DataFrame()

# Overview
if view_mode == "Overview":
    st.subheader("Real-Time Transaction Overview")
    
    if filtered_df.empty:
        st.info("No transactions available for the selected filters.")
    else:
        # kpis
        total_transactions = len(filtered_df)
        blocked_transactions = len(filtered_df[filtered_df["decision"]== "BLOCK"])
        review_transactions = len(filtered_df[filtered_df["decision"]== "REVIEW"])
        total_value = filtered_df["amount"].sum()
        avg_fraud_probability = (filtered_df["fraud_probability"].mean())

        #KPI row
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Transactions", f"{total_transactions:,}")
        col2.metric("Blocked", f"{blocked_transactions:,}")
        col3.metric("Under Review", f"{review_transactions:,}")
        col4.metric("Value Processed", f"₹{total_value:,.0f}")
        col5.metric("Avg Fraud Probability", f"{avg_fraud_probability:.1f}")
        
        st.markdown("---")

        # Decision Breakdown
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.subheader("Decision Breakdown")
            decision_counts = (filtered_df["decision"].value_counts().reset_index())
            decision_counts.columns = ["Decision","Count"]

            fig_pie = px.pie(decision_counts, names='Decision', values="Count", hole=0.45, color='Decision', color_discrete_map={"ALLOW": "#00c04b","REVIEW": "#ffa421", "BLOCK": "#ff4b4b"})

            fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_pie, use_container_width=True)

        # Scenarios
        with chart_col2:
            st.subheader("Detected Fraud Scenarios")
            scenario_df = filtered_df[~filtered_df["scenario"].isin(["Clean","None","No Fraud Detected"])]
            scenario_counts = scenario_df['scenario'].value_counts().reset_index()
            scenario_counts.columns = ['Scenario', 'Count']

            if scenario_counts.empty:
                st.info("No suspicious scenarios detected.")
            else:
                fig_bar = px.bar(scenario_counts, x='Count', y='Scenario', orientation='h')
                fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20), yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    
    # Live Transaction Feed
    st.subheader("Live Transaction Feed")

    display_columns = ['timestamp', 'transaction_id', 'amount', 'fraud_probability', "anomaly_score", 'scenario', 'decision']
    display_columns = [col for col in display_columns if col in filtered_df.columns]

    display_df = filtered_df[display_columns].head(20).copy()
    if "amount" in display_df:
        display_df['amount'] = display_df['amount'].apply(lambda x: f"₹{x:,.2f}")
    if "anomaly_score" in display_df:
        display_df["anomaly_score"] = (display_df["anomaly_score"].apply(lambda x:f"{x:.1%}"))
    if "fraud_probability" in display_df:
        display_df['fraud_probability'] = display_df['fraud_probability'].apply(lambda x: f"{x:.1%}")
    if "timestamp" in display_df:
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    st.dataframe(display_df,use_container_width=True,hide_index=True)

# Transaction Inspector View
elif view_mode == "Transaction Inspector":
    st.subheader("Transaction Inspector")
    st.caption("Investigate why the risk engine produced a specific decision.")

    search_id = st.text_input("Transaction ID", placeholder="TXN00123456")
    
    if search_id:
        if filtered_df.empty:
            st.error("No Transaction data available.")
        else:
            # Fetch the specific transaction
            txn_data = filtered_df[filtered_df["transaction_id"].astype(str) == search_id]

            if txn_data.empty:
                st.error("Transaction not found.")
            else:
                txn = txn_data.iloc[0]
                
                decision = txn["decision"]
                color = decision_color(decision)
                icon = decision_icon(decision)

                # Decision header
                st.markdown(f"""<h2 style='color: {color};'>{icon} {decision}</h2>""", unsafe_allow_html=True)

                # Risk metrics
                col1, col2, col3 = (st.columns(3))

                col1.metric("Fraud Probability", format_percentage(txn["fraud_probability"]))
                col2.metric("Anomaly Risk", format_percentage(txn["anomaly_score"]))
                if "risk_score" in txn.index and pd.notna(txn["risk_score"]):
                    risk_value = float(txn["risk_score"])
                    col3.metric("Composite Risk", f"{risk_value:.1f}/100")
                else:
                    # Fallback for current schema
                    calculated_risk = (txn["fraud_probability"] * 0.60 + txn["anomaly_score"] * 0.30) * 100
                    col3.metric("Composite Risk",f"{calculated_risk:.1f}/100")

                st.markdown("---")

                # Transaction details
                detail_col1, detail_col2 = st.columns(2)

                with detail_col1:
                    st.subheader("Transaction Details")
                    st.write(f"**Transaction ID:** " f"{txn['transaction_id']}")
                    st.write(f"**Timestamp:** " f"{txn['timestamp']}")
                    st.write(f"**Amount:** " f"₹{txn['amount']:,.2f}")
                    st.write(f"**Scenario:** " f"{txn['scenario']}")

                # Context signals
                with detail_col2:
                    st.subheader("Context Signals")
                    context_df = (load_transaction_context(search_id))
                    if not context_df.empty:
                        context = context_df.iloc[0]
                        if "device_is_new_for_user" in context:
                            st.write("**New Device:** " + display_context_signal(context["device_is_new_for_user"]))
                        if "location_mismatch" in context:
                            st.write("**Location Mismatch:** "+ display_context_signal(context["location_mismatch"]))
                        if "device_distinct_users_seen_so_far" in context:
                            st.write("**Device Users Seen:** "f"{context['device_distinct_users_seen_so_far']}")

                    else:

                        st.info( "Context information unavailable.")

                st.markdown("-----")
                
                # Risk Gauge
                st.subheader("Composite Risk Score")
                if "risk_score" in txn.index:
                    risk_value = float(txn["risk_score"])
                else:
                    risk_value = (txn["fraud_probability"] * 0.60 + txn["anomaly_score"] * 0.30) * 100

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_value,
                    number = {"suffix": "/100"},
                    title={'text': "Overall Risk"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 35], 'color': "lightgreen"},
                            {'range': [35, 75], 'color': "navajowhite"},
                            {'range': [75, 100], 'color': "lightcoral"}
                        ],
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Actual Risk engine triggers
                st.subheader("Risk Engine Triggers")
                if "triggers" in txn.index:
                    triggers = safe_json_load(txn["triggers"])
                    if triggers:
                        for trigger in triggers:
                            if decision == "BLOCK":
                                st.error(f"• {trigger}")
                            elif decision == "REVIEW":
                                st.warning(f"• {trigger}")
                            else:
                                st.success(f"• {trigger}")
                    else:
                        st.success("No risk triggers recorded.")

                else:
                    st.info("Actual triggers are not stored in "
                            "the current fraud_log schema.")
                
                # Action notes
                if "action_notes" in txn.index:
                    st.subheader("Recommended Action")
                    st.info(txn["action_notes"])

# Risk Monitoring
elif view_mode == "Risk Monitoring":
    st.subheader("Risk Monitoring")
    if filtered_df.empty:
        st.info("No transactions available.")
    else:
        # Risk over time
        st.subheader( "Fraud Probability Over Time")
        risk_time_df = (filtered_df.sort_values("timestamp"))
        fig = px.line(risk_time_df, x="timestamp", y="fraud_probability", color="decision",
                      color_discrete_map={"ALLOW": "#00c04b", "REVIEW": "#ffa421", "BLOCK": "#ff4b4b"})
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True,)

        # Anomaly over time
        st.subheader("Anomaly Risk Over Time")
        fig = px.scatter(risk_time_df, x="timestamp", y="anomaly_score", color="decision",
                         color_discrete_map={"ALLOW": "#00c04b", "REVIEW": "#ffa421", "BLOCK": "#ff4b4b"},
                         hover_data=["transaction_id","amount", "scenario"])
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig,use_container_width=True)

        # Amount vs Fraud Risk
        st.subheader("Transaction Amount vs Fraud Risk")
        fig = px.scatter(filtered_df,x="amount",y="fraud_probability", color="decision", size="amount",
                         hover_data=["transaction_id", "scenario"], 
                         color_discrete_map={"ALLOW": "#00c04b","REVIEW": "#ffa421","BLOCK": "#ff4b4b"})
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

# Model Insights
elif view_mode == "Model Insights":
    st.subheader("Model Insights")
    if filtered_df.empty:
        st.info("No model predictions available.")
    else:
        # Model metrics
        col1, col2, col3 = st.columns(3)

        col1.metric("Avg Fraud Probability", f"{filtered_df['fraud_probability'].mean():.1%}")
        col2.metric("Avg Anomaly Risk", f"{filtered_df['anomaly_score'].mean():.1%}")

        if "risk_score" in filtered_df.columns:
            col3.metric("Avg Composite Risk", f"{filtered_df['risk_score'].mean():.1f}/100")
        else:
            calculated_avg = ((filtered_df["fraud_probability"] * 0.60 + filtered_df["anomaly_score"] * 0.30) * 100).mean()
            col3.metric("Avg Composite Risk", f"{calculated_avg:.1f}/100")

        st.markdown("---")

        # Fraud probability distribution
        st.subheader("Fraud Probability Distribution")
        fig = px.histogram(filtered_df, x="fraud_probability", nbins=30)
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        # Anomaly distribution
        st.subheader("Anomaly Risk Distribution")
        fig = px.histogram(filtered_df,x="anomaly_score",nbins=30)
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        # Scenario table
        st.subheader("Scenario Model Output")

        scenario_summary = (filtered_df.groupby("scenario").agg(transactions=("transaction_id","count"),
                                                                avg_fraud_probability=("fraud_probability","mean"),
                                                                avg_anomaly_score=("anomaly_score","mean")).reset_index().sort_values("transactions",ascending=False))

        scenario_summary["avg_fraud_probability"] = (scenario_summary["avg_fraud_probability"].map(lambda x:f"{x:.1%}"))

        scenario_summary["avg_anomaly_score"] = (scenario_summary["avg_anomaly_score"].map(lambda x:f"{x:.1%}"))

        st.dataframe(scenario_summary, use_container_width=True, hide_index=True)

# FOOTER
st.sidebar.markdown("---")

st.sidebar.caption("Real-Time Fraud Detection & Risk Analytics Platform")

st.sidebar.caption("Kafka • XGBoost • Isolation Forest • Risk Engine • Streamlit")
         