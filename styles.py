import streamlit as st

from components import ROOT  # noqa: F401  (bootstraps sys.path)
from utils.config import ASSETS_DIR


def stretch():
    """Full-width kwargs that work on old and new Streamlit versions."""
    try:
        from packaging.version import Version
        if Version(st.__version__) >= Version("1.50"):
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}


def configure_page(title="FraudShield AI", icon="💳"):
    st.set_page_config(page_title=title, page_icon=icon, layout="wide",
                       initial_sidebar_state="expanded")


def load_css():
    css = ASSETS_DIR / "style.css"
    if css.exists():
        st.markdown(f"<style>{css.read_text(encoding='utf-8')}</style>",
                    unsafe_allow_html=True)


def setup_page(title, icon="💳"):
    """Call first on every page: page config, CSS and sidebar."""
    from components.sidebar import show_sidebar
    configure_page(title, icon)
    load_css()
    show_sidebar()
