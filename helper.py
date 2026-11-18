"""Small pure helpers (no Streamlit)."""
import json
from pathlib import Path

import pandas as pd


def calculate_kpis(df, target="Class"):
    total = len(df)
    fraud = int(df[target].sum())
    return {
        "total": total,
        "fraud": fraud,
        "genuine": total - fraud,
        "rate": (fraud / total * 100) if total else 0.0,
    }


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def read_csv_if_exists(path):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else None
