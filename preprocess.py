"""Turn raw transaction rows into exactly what the model expects."""
from utils.config import FEATURES


def missing_columns(df, required=FEATURES):
    return [c for c in required if c not in df.columns]


def prepare_features(df, scaler, amount_is_scaled=False):
    """Return a feature frame in training order.

    The model was trained on a *standardised* Amount (see models/scaler.pkl),
    while raw data holds the original currency amount. Feeding the raw value
    silently degrades predictions, so the scaler is applied here unless the
    caller says Amount is already scaled.
    """
    missing = missing_columns(df)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    X = df[FEATURES].astype("float64").copy()
    if not amount_is_scaled:
        X["Amount"] = scaler.transform(X[["Amount"]]).ravel()
    return X
