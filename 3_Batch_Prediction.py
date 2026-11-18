import pandas as pd
import streamlit as st

from components.styles import setup_page, stretch
from components.helpers import require_artifacts
from utils.config import DEFAULT_THRESHOLD, TARGET
from utils.predict import predict_frame
from utils.preprocess import missing_columns

setup_page("Batch Prediction", "📂")
st.title("📂 Batch Fraud Prediction")
st.markdown("Upload a CSV with many transactions (columns `Time`, `V1`-`V28`, `Amount`). "
            "An optional `Class` column is used to show how well the model did.")

model, scaler = require_artifacts()

c1, c2 = st.columns(2)
threshold = c1.slider("Decision threshold", 0.05, 0.95, DEFAULT_THRESHOLD, 0.05)
amount_is_scaled = c2.checkbox("Amount column is already standardised", value=False)

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is None:
    st.info("👆 Upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("📄 Uploaded data")
st.caption(f"{len(df):,} rows")
st.dataframe(df.head(), **stretch())

missing = missing_columns(df)
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

run_key = (uploaded.name, uploaded.size, threshold, amount_is_scaled)
if st.button("🚀 Predict all transactions"):
    with st.spinner("Scoring..."):
        scored = predict_frame(model, scaler, df, threshold, amount_is_scaled)
    st.session_state["batch"] = (run_key, pd.concat([df, scored], axis=1))

stored = st.session_state.get("batch")
if not stored or stored[0] != run_key:  # results survive the download click's rerun
    st.stop()
result = stored[1]

fraud = int(result["Prediction"].sum())
m1, m2, m3, m4 = st.columns(4)
m1.metric("Transactions", f"{len(result):,}")
m2.metric("Flagged as fraud", f"{fraud:,}")
m3.metric("Flagged as genuine", f"{len(result) - fraud:,}")
m4.metric("Flag rate", f"{fraud / len(result) * 100:.2f}%")

if TARGET in result.columns:
    actual = result[TARGET].astype(int)
    tp = int(((result["Prediction"] == 1) & (actual == 1)).sum())
    fn = int(((result["Prediction"] == 0) & (actual == 1)).sum())
    fp = int(((result["Prediction"] == 1) & (actual == 0)).sum())
    st.subheader("Check against the `Class` column")
    k1, k2, k3 = st.columns(3)
    k1.metric("Frauds caught", f"{tp} / {tp + fn}")
    k2.metric("Recall", f"{tp / (tp + fn):.1%}" if tp + fn else "n/a")
    k3.metric("Precision", f"{tp / (tp + fp):.1%}" if tp + fp else "n/a")

st.subheader("Most suspicious transactions")
st.dataframe(result.sort_values("Fraud Probability", ascending=False).head(100), **stretch())
st.download_button("⬇ Download full result", result.to_csv(index=False).encode(),
                   "Prediction_Result.csv", "text/csv")
