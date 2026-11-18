import pandas as pd
import streamlit as st

from components.styles import setup_page, stretch
from components.helpers import require_artifacts, require_data
from components.prediction_card import show_prediction
from components.alerts import fraud_alert
from utils.config import DEFAULT_THRESHOLD, FEATURES, TARGET
from utils.predict import predict_frame
from utils.preprocess import missing_columns

setup_page("Live Prediction", "💳")
st.title("💳 Live Fraud Prediction")
st.markdown("Score one transaction as **Genuine** or **Fraudulent** with the trained model.")

df, is_sample = require_data()
model, scaler = require_artifacts()

c1, c2 = st.columns([3, 2])
mode = c1.radio("Choose transaction",
                ["🟢 Random genuine", "🔴 Random fraud", "📁 Upload CSV"], horizontal=True)
threshold = c2.slider("Decision threshold", 0.05, 0.95, DEFAULT_THRESHOLD, 0.05,
                      help="Lower = catch more fraud but more false alarms.")

amount_is_scaled = False
if mode == "📁 Upload CSV":
    amount_is_scaled = st.checkbox("Amount column is already standardised", value=False)
    uploaded = st.file_uploader("Upload a transaction CSV", type="csv")
    if uploaded is None:
        st.info("Upload a CSV with columns Time, V1-V28, Amount.")
        st.stop()
    up = pd.read_csv(uploaded)
    missing = missing_columns(up)
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()
    row = 1
    if len(up) > 1:
        row = st.number_input("Row to score", 1, len(up), 1)
    sample = up.iloc[[int(row) - 1]]
else:
    label = 0 if "genuine" in mode else 1
    key = f"live_sample_{label}"  # kept in session so a click doesn't change the transaction
    if key not in st.session_state:
        st.session_state[key] = df[df[TARGET] == label].sample(1)
    if st.button("🎲 Draw another"):
        st.session_state[key] = df[df[TARGET] == label].sample(1)
    sample = st.session_state[key]

st.subheader("Transaction")
st.dataframe(sample[FEATURES], **stretch())

result = predict_frame(model, scaler, sample, threshold, amount_is_scaled).iloc[0]
col_a, col_b = st.columns(2)
with col_a:
    show_prediction(bool(result["Prediction"]), result["Fraud Probability"], threshold)
with col_b:
    fraud_alert(result["Risk"])
    if TARGET in sample.columns:
        st.caption(f"Actual label in the data: **{'Fraud' if int(sample[TARGET].iloc[0]) else 'Genuine'}**")
    if result["Prediction"]:
        st.error("Recommended action: block or manually review this transaction.")
    else:
        st.success("No action needed.")
