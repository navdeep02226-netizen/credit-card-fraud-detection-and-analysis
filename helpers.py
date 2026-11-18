"""Cached loaders - the dataset and model are read once, not on every click."""
import streamlit as st

from components import ROOT  # noqa: F401
from utils.data import load_dataset
from utils.predict import load_artifacts


@st.cache_data(show_spinner="Loading transactions...")
def get_data():
    return load_dataset()  # (DataFrame, is_sample)


@st.cache_resource(show_spinner="Loading model...")
def get_artifacts():
    return load_artifacts()  # (model, scaler)


def require_data():
    """Load data or show a friendly error and stop the page."""
    try:
        return get_data()
    except FileNotFoundError as err:
        st.error(str(err))
        st.stop()


def require_artifacts():
    try:
        return get_artifacts()
    except (FileNotFoundError, ValueError) as err:
        st.error(str(err))
        st.stop()


def sample_notice(is_sample):
    if is_sample:
        st.info("Showing the bundled 20,000-row sample (all 492 frauds + random genuine "
                "rows), so totals and the fraud rate are not the full-dataset numbers. "
                "Place the Kaggle file at `data/raw/creditcard.csv` for the full data.")
