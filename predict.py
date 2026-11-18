"""Model loading and prediction."""
import joblib
import pandas as pd

from utils.config import BEST_MODEL, DEFAULT_THRESHOLD, FEATURES, RISK_BANDS, SCALER
from utils.preprocess import prepare_features


def load_artifacts(model_path=BEST_MODEL, scaler_path=SCALER):
    for p in (model_path, scaler_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}. Run `python train.py` to create it.")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    names = list(getattr(model, "feature_names_in_", FEATURES))
    if names != FEATURES:
        raise ValueError("Model feature names do not match the expected schema.")
    return model, scaler


def risk_level(probability):
    for minimum, label in RISK_BANDS:
        if probability >= minimum:
            return label
    return "SAFE"


def predict_frame(model, scaler, df, threshold=DEFAULT_THRESHOLD, amount_is_scaled=False):
    """Score every row. Returns a frame with the same index as `df`."""
    X = prepare_features(df, scaler, amount_is_scaled)
    proba = model.predict_proba(X)[:, 1]
    return pd.DataFrame(
        {
            "Fraud Probability": proba.round(4),
            "Prediction": (proba >= threshold).astype(int),
            "Risk": [risk_level(p) for p in proba],
        },
        index=df.index,
    )
