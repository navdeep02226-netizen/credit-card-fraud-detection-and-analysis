"""Plotly figure builders. Heavy work is aggregated first so big data stays fast."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

TEMPLATE = "plotly_dark"
COLORS = {"Genuine": "#16A34A", "Fraud": "#DC2626"}


def _label(df):
    return df["Class"].map({0: "Genuine", 1: "Fraud"})


def fraud_distribution_chart(df):
    counts = _label(df).value_counts().reindex(["Genuine", "Fraud"]).fillna(0).reset_index()
    counts.columns = ["Type", "Count"]
    fig = px.pie(counts, names="Type", values="Count", hole=0.7, color="Type",
                 color_discrete_map=COLORS, title="Fraud vs Genuine")
    fig.update_layout(template=TEMPLATE)
    return fig


def amount_distribution_chart(df):
    cap = float(df["Amount"].quantile(0.99))
    d = df.loc[df["Amount"] <= cap, ["Amount", "Class"]].copy()
    d["Type"] = _label(d)
    fig = px.histogram(d, x="Amount", color="Type", nbins=60, barmode="overlay",
                       opacity=0.75, log_y=True, color_discrete_map=COLORS,
                       title="Transaction amount (up to the 99th percentile, log scale)")
    fig.update_layout(template=TEMPLATE)
    return fig


def fraud_trend_chart(df):
    d = pd.DataFrame({"Hour": (df["Time"] // 3600).astype(int), "Class": df["Class"]})
    hourly = d.groupby("Hour")["Class"].agg(Fraud="sum", Total="count").reset_index()
    fig = px.bar(hourly, x="Hour", y="Fraud", hover_data=["Total"],
                 title="Fraud transactions per hour since first transaction")
    fig.update_traces(marker_color=COLORS["Fraud"])
    fig.update_layout(template=TEMPLATE)
    return fig


def amount_box_chart(df, max_rows=30000):
    d = df[["Amount", "Class"]]
    if len(d) > max_rows:  # keep every fraud row, thin out genuine rows
        fraud = d[d["Class"] == 1]
        genuine = d[d["Class"] == 0].sample(max_rows - len(fraud), random_state=42)
        d = pd.concat([fraud, genuine])
    d = d.assign(Type=_label(d))
    fig = px.box(d, x="Type", y="Amount", color="Type",
                 color_discrete_map=COLORS, title="Amount by transaction type")
    fig.update_layout(template=TEMPLATE, showlegend=False)
    return fig


def class_correlation_chart(df, top=12):
    corr = df.drop(columns=["Class"]).corrwith(df["Class"]).dropna()
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index[:top])[::-1]
    fig = px.bar(x=corr.values, y=corr.index, orientation="h",
                 labels={"x": "Correlation with fraud", "y": "Feature"},
                 title=f"Top {top} features correlated with fraud")
    fig.update_traces(marker_color=["#DC2626" if v > 0 else "#2563EB" for v in corr.values])
    fig.update_layout(template=TEMPLATE)
    return fig


def metric_bar_chart(results, metric):
    d = results.dropna(subset=[metric])
    fig = px.bar(d, x="Model", y=metric, color="Model", text=d[metric].round(4),
                 title=f"{metric} comparison")
    fig.update_layout(template=TEMPLATE, yaxis_range=[0, 1.05], showlegend=False)
    return fig


def curve_chart(x, y, title, xlabel, ylabel, diagonal=False):
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", line=dict(width=3)))
    if diagonal:
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line=dict(dash="dash", color="gray")))
    fig.update_layout(template=TEMPLATE, title=title, xaxis_title=xlabel,
                      yaxis_title=ylabel, showlegend=False)
    return fig


def confusion_matrix_chart(cm):
    z = np.array(cm)
    fig = px.imshow(z, text_auto=True, color_continuous_scale="Blues",
                    x=["Genuine", "Fraud"], y=["Genuine", "Fraud"],
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    title="Confusion matrix (test set)")
    fig.update_layout(template=TEMPLATE)
    return fig


def feature_importance_chart(importances, top=15):
    s = pd.Series(importances).sort_values(ascending=False).head(top)[::-1]
    fig = px.bar(x=s.values, y=s.index, orientation="h",
                 labels={"x": "Importance", "y": "Feature"},
                 title=f"Top {top} feature importances")
    fig.update_layout(template=TEMPLATE)
    return fig
