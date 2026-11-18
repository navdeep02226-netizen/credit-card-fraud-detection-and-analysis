import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import BEST_MODEL, DATA_SAMPLE, FEATURES, SCALER  # noqa: E402
from utils.helper import calculate_kpis  # noqa: E402
from utils.predict import load_artifacts, predict_frame, risk_level  # noqa: E402
from utils.preprocess import missing_columns, prepare_features  # noqa: E402

pytestmark = pytest.mark.skipif(not (BEST_MODEL.exists() and DATA_SAMPLE.exists()),
                                reason="model or sample data missing")


@pytest.fixture(scope="module")
def parts():
    model, scaler = load_artifacts()
    return model, scaler, pd.read_csv(DATA_SAMPLE)


def test_kpis_have_the_keys_the_dashboard_uses(parts):
    k = calculate_kpis(parts[2])
    assert set(k) == {"total", "fraud", "genuine", "rate"}
    assert k["fraud"] + k["genuine"] == k["total"]


def test_amount_is_scaled_unless_told_otherwise(parts):
    _, scaler, df = parts
    X = prepare_features(df.head(5), scaler)
    assert list(X.columns) == FEATURES
    expected = (df["Amount"].head(5) - scaler.mean_[0]) / scaler.scale_[0]
    assert np.allclose(X["Amount"], expected)
    raw = prepare_features(df.head(5), scaler, amount_is_scaled=True)
    assert np.allclose(raw["Amount"], df["Amount"].head(5))


def test_missing_columns_detected(parts):
    assert missing_columns(parts[2].drop(columns=["V3"])) == ["V3"]
    with pytest.raises(ValueError):
        prepare_features(parts[2].drop(columns=["V3"]), parts[1])


def test_model_separates_fraud_from_genuine(parts):
    model, scaler, df = parts
    fraud = predict_frame(model, scaler, df[df.Class == 1])
    genuine = predict_frame(model, scaler, df[df.Class == 0].head(2000))
    assert fraud["Prediction"].mean() > 0.7
    assert genuine["Prediction"].mean() < 0.01


def test_threshold_changes_flags(parts):
    model, scaler, df = parts
    rows = df[df.Class == 1].head(50)
    low = predict_frame(model, scaler, rows, threshold=0.05)["Prediction"].sum()
    high = predict_frame(model, scaler, rows, threshold=0.95)["Prediction"].sum()
    assert low >= high


def test_risk_levels():
    assert [risk_level(p) for p in (0.95, 0.6, 0.3, 0.01)] == ["HIGH", "MEDIUM", "LOW", "SAFE"]
