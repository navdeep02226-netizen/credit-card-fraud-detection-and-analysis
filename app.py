"""Home page.  Run from the project root:  streamlit run dashboard/app.py"""
import streamlit as st

from components.styles import setup_page
from components.footer import show_footer

setup_page("FraudShield AI", "💳")

st.title("💳 AI Credit Card Fraud Detection")
st.markdown(
    """
Welcome to **FraudShield AI**. Pick a page from the sidebar:

| Page | What it does |
|---|---|
| **Dashboard** | Key numbers and charts for the transaction data |
| **Live Prediction** | Score a single transaction |
| **Batch Prediction** | Upload a CSV and score many transactions |
| **Fraud Analytics** | Explore fraud patterns |
| **Model Performance** | Compare models, ROC / PR curves, confusion matrix |
| **Explainability** | SHAP plots and feature importances |
"""
)
show_footer()
