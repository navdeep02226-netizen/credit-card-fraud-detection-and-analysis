import streamlit as st


def fraud_alert(level):
    """`level` is one of HIGH / MEDIUM / LOW / SAFE (see utils.predict.risk_level)."""
    if level == "HIGH":
        st.error("🔴 HIGH RISK")
    elif level == "MEDIUM":
        st.warning("🟠 MEDIUM RISK")
    elif level == "LOW":
        st.info("🟡 LOW RISK")
    else:
        st.success("🟢 SAFE")
