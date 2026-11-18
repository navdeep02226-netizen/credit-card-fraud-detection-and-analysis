import streamlit as st

from components.styles import setup_page, stretch
from components.helpers import require_data, sample_notice
from components.header import show_header
from components.footer import show_footer
from components.cards import show_kpi_cards
from components.transaction import show_recent_transactions
from utils.helper import calculate_kpis
from utils.visualization import (amount_distribution_chart, fraud_distribution_chart,
                                 fraud_trend_chart)

setup_page("Dashboard", "💳")
show_header()

df, is_sample = require_data()
sample_notice(is_sample)

st.subheader("📌 Key indicators")
show_kpi_cards(calculate_kpis(df))
st.divider()

st.subheader("📊 Fraud overview")
left, right = st.columns(2)
left.plotly_chart(fraud_distribution_chart(df), **stretch())
right.plotly_chart(amount_distribution_chart(df), **stretch())
st.plotly_chart(fraud_trend_chart(df), **stretch())
st.divider()

show_recent_transactions(df)
show_footer()
