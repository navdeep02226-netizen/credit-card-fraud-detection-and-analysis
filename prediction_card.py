import streamlit as st


def show_prediction(is_fraud, probability, threshold):
    probability = float(probability)
    if is_fraud:
        st.error("## 🚨 Fraudulent transaction")
    else:
        st.success("## ✅ Genuine transaction")
    st.metric("Fraud probability", f"{probability * 100:.2f}%",
              help=f"Flagged as fraud when probability >= {threshold:.0%}")
    st.progress(min(max(probability, 0.0), 1.0))
