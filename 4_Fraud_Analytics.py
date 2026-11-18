import streamlit as st

from components.styles import setup_page, stretch
from components.helpers import require_data, sample_notice
from components.cards import show_kpi_cards
from components.footer import show_footer
from utils.helper import calculate_kpis
from utils.visualization import (amount_box_chart, amount_distribution_chart,
                                 class_correlation_chart, fraud_distribution_chart,
                                 fraud_trend_chart)

setup_page("Fraud Analytics", "📊")
st.title("📊 Fraud Analytics")
st.markdown("Explore fraud distribution, amounts, timing and the features most tied to fraud.")

df, is_sample = require_data()
sample_notice(is_sample)

show_kpi_cards(calculate_kpis(df))
st.divider()

left, right = st.columns(2)
left.plotly_chart(fraud_distribution_chart(df), **stretch())
right.plotly_chart(amount_distribution_chart(df), **stretch())

st.plotly_chart(fraud_trend_chart(df), **stretch())

left, right = st.columns(2)
left.plotly_chart(amount_box_chart(df), **stretch())
right.plotly_chart(class_correlation_chart(df), **stretch())

st.subheader("📋 Dataset statistics")
st.dataframe(df.describe(), **stretch())

st.subheader("🚨 Most recent fraud transactions")
st.dataframe(df[df["Class"] == 1].tail(20), **stretch())
show_footer()
