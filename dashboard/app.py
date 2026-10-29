import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Suzuki Market Intelligence Dashboard", layout="wide")

st.title("🚗 Suzuki Alto Market Intelligence - Karachi Demand Predictor")
st.markdown("Interactive scenario simulation model driving local sales forecasts based on macroeconomic signals.")

# Load Model & Data
@st.cache_data
def load_data():
    if os.path.exists("master_market_intelligence_features.csv"):
        return pd.read_csv("master_market_intelligence_features.csv")
    return None

df = load_data()

st.sidebar.header("Scenario Simulation Controls")
petrol = st.sidebar.slider("Petrol Price (PKR/Liter)", 200.0, 350.0, 280.0)
kibor = st.sidebar.slider("6-Month KIBOR Rate (%)", 8.0, 25.0, 11.5)
duty = st.sidebar.slider("Total Effective Duty Rate (%)", 15.0, 30.0, 22.0)
sro = st.sidebar.selectbox("July 2026 SRO Active?", [1, 0], index=0)

auto_finance = kibor + 2.5

# Make Prediction
if os.path.exists("src/suzuki_alto_model.pkl"):
    model = joblib.load("src/suzuki_alto_model.pkl")
    input_data = pd.DataFrame({
        'petrol_pkr_per_liter': [petrol],
        'kibor_6m_pct': [kibor],
        'auto_finance_avg_rate_pct': [auto_finance],
        'total_duty_rate_pct': [duty],
        'sro_event_flag': [sro]
    })
    predicted_sales = model.predict(input_data)[0]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Monthly Sales (Karachi)", f"{int(predicted_sales):,} units")
    col2.metric("Effective Auto Financing Rate", f"{auto_finance:.2f}%")
    col3.metric("July 2026 Duty Regime", "Active" if sro == 1 else "Baseline")
else:
    st.warning("Model file 'src/suzuki_alto_model.pkl' not found. Please train the model first.")

if df is not None:
    st.subheader("Historical Market Trends")
    st.line_chart(df.set_index('date')[['suzuki_alto_sales_units', 'petrol_pkr_per_liter']])
