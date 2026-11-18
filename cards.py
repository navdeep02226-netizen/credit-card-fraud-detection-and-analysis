import streamlit as st


def show_kpi_cards(kpi):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💳 Transactions", f"{kpi['total']:,}")
    c2.metric("🚨 Fraud", f"{kpi['fraud']:,}")
    c3.metric("✅ Genuine", f"{kpi['genuine']:,}")
    c4.metric("📈 Fraud rate", f"{kpi['rate']:.3f}%")
