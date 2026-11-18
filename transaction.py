import streamlit as st

from components.styles import stretch


def show_recent_transactions(df, rows=15):
    st.subheader("📋 Recent transactions")
    st.dataframe(df.tail(rows), height=420, **stretch())
