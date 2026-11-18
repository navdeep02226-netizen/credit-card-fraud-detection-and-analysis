"""Dataset loading with a safe fallback to the bundled sample."""
import pandas as pd

from utils.config import DATA_RAW, DATA_SAMPLE, FEATURES, TARGET


def find_dataset():
    """Return (path, is_sample). Prefer the full dataset, fall back to the sample."""
    if DATA_RAW.exists():
        return DATA_RAW, False
    if DATA_SAMPLE.exists():
        return DATA_SAMPLE, True
    raise FileNotFoundError(
        f"Dataset not found. Put the Kaggle file at '{DATA_RAW}' "
        f"(or restore the sample at '{DATA_SAMPLE}')."
    )


def load_dataset():
    """Load the transactions with compact dtypes. Returns (DataFrame, is_sample)."""
    path, is_sample = find_dataset()
    dtypes = {c: "float32" for c in FEATURES}
    dtypes[TARGET] = "int8"
    df = pd.read_csv(path, dtype=dtypes)
    return df, is_sample
