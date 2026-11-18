import streamlit as st

from components import ROOT  # noqa: F401
from utils.config import ASSETS_DIR, BEST_MODEL, SCALER
from utils.data import find_dataset


def show_sidebar():
    with st.sidebar:
        logo = ASSETS_DIR / "logo.png"
        if logo.exists():
            st.image(str(logo), width=90)
        st.title("FraudShield AI")
        st.caption("Use the page list above to navigate.")
        st.divider()

        (st.success if BEST_MODEL.exists() and SCALER.exists() else st.error)(
            "Model ready" if BEST_MODEL.exists() and SCALER.exists() else "Model files missing")
        try:
            _, is_sample = find_dataset()
            (st.warning if is_sample else st.success)(
                "Dataset: bundled sample" if is_sample else "Dataset: full data")
        except FileNotFoundError:
            st.error("Dataset missing")
