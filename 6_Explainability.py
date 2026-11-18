import streamlit as st

from components.styles import setup_page, stretch
from utils.config import BEST_MODEL_METRICS, IMAGES_DIR
from utils.helper import read_json
from utils.visualization import feature_importance_chart

setup_page("Explainability", "🤖")
st.title("🤖 Explainable AI")
st.markdown("Why does the model call a transaction fraud or genuine? "
            "**SHAP** shows how much each feature pushed the prediction.")

images = {
    "📊 SHAP summary": IMAGES_DIR / "shap_summary.png",
    "🌊 SHAP waterfall (one transaction)": IMAGES_DIR / "shap_waterfall.png",
    "⚡ SHAP force (one transaction)": IMAGES_DIR / "shap_force.png",
    "📈 SHAP scatter (V14)": IMAGES_DIR / "shap_scatter.png",
}
choice = st.selectbox("SHAP visualisation", list(images))
path = images[choice]
if path.exists():
    st.image(str(path), **stretch())
else:
    st.warning(f"Image not found: {path.name}. Re-run notebook 05 to create it.")

st.info("• Positive values push toward **Fraud**.  • Negative values push toward **Genuine**.  "
        "• Larger magnitude = bigger influence.")

rep = read_json(BEST_MODEL_METRICS)
if rep and rep.get("feature_importance"):
    st.divider()
    st.plotly_chart(feature_importance_chart(rep["feature_importance"]), **stretch())
