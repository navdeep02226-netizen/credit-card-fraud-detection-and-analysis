"""Central configuration: every path and constant lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Data
DATA_RAW = ROOT / "data" / "raw" / "creditcard.csv"
DATA_SAMPLE = ROOT / "data" / "sample" / "creditcard_sample.csv"
DATA_PROCESSED = ROOT / "data" / "processed"

# Artifacts
MODELS_DIR = ROOT / "models"
BEST_MODEL = MODELS_DIR / "best_model.pkl"
SCALER = MODELS_DIR / "scaler.pkl"

# Reports
REPORTS_DIR = ROOT / "reports"
IMAGES_DIR = REPORTS_DIR / "images"
MODEL_COMPARISON = REPORTS_DIR / "model_comparison.csv"
BEST_MODEL_METRICS = REPORTS_DIR / "best_model_metrics.json"

# Dashboard
ASSETS_DIR = ROOT / "dashboard" / "assets"

# Schema (order matters: the model was trained with exactly this column order)
TARGET = "Class"
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Prediction
DEFAULT_THRESHOLD = 0.5
RISK_BANDS = [  # (minimum probability, label) - checked from the top
    (0.80, "HIGH"),
    (0.50, "MEDIUM"),
    (0.20, "LOW"),
]
