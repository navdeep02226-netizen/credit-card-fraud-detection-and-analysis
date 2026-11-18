import pandas as pd
import streamlit as st

from components.styles import setup_page, stretch
from utils.config import BEST_MODEL_METRICS, MODEL_COMPARISON
from utils.helper import read_csv_if_exists, read_json
from utils.visualization import (confusion_matrix_chart, curve_chart, metric_bar_chart)

setup_page("Model Performance", "📈")
st.title("📈 Model Performance")

results = read_csv_if_exists(MODEL_COMPARISON)
if results is None:
    st.error("`reports/model_comparison.csv` not found. Run `python train.py` first.")
    st.stop()

st.caption("All numbers are measured on the held-out test set (threshold 0.5). "
           "Fraud is only ~0.17% of transactions, so look at Recall, Precision and PR AUC - "
           "accuracy alone is misleading.")
st.dataframe(results.style.format(precision=4, na_rep="-"), **stretch())

metrics = [c for c in results.columns if c != "Model"]
metric = st.selectbox("Metric", metrics, index=metrics.index("F1 Score") if "F1 Score" in metrics else 0)
if results[metric].notna().any():
    st.plotly_chart(metric_bar_chart(results, metric), **stretch())
    best = results.sort_values(metric, ascending=False).iloc[0]
    st.success(f"🏆 Best on {metric}: {best['Model']} ({best[metric]:.4f})")
else:
    st.info(f"No values for {metric} yet - run `python train.py` to fill them in.")

rep = read_json(BEST_MODEL_METRICS)
if rep:
    st.divider()
    st.subheader("Deployed model (best_model.pkl)")
    a, b = st.columns(2)
    a.plotly_chart(curve_chart(rep["roc"]["fpr"], rep["roc"]["tpr"],
                               f"ROC curve (AUC {rep['roc_auc']:.3f})",
                               "False positive rate", "True positive rate", diagonal=True), **stretch())
    b.plotly_chart(curve_chart(rep["pr"]["recall"], rep["pr"]["precision"],
                               f"Precision-recall curve (AP {rep['pr_auc']:.3f})",
                               "Recall", "Precision"), **stretch())
    st.plotly_chart(confusion_matrix_chart(rep["confusion_matrix"]), **stretch())
