"""Reproducible training pipeline.

    python train.py                       # full run (recommended)
    python train.py --quick               # small/fast models for a smoke test
    python train.py --imbalance smote     # oversample the TRAINING split only
    python train.py --imbalance class_weight

What it fixes compared with the old notebooks
  * split first, then fit the scaler on the training rows only (no leakage)
  * resampling (SMOTE) is applied to the training split only and is actually used
  * ROC-AUC is computed from probabilities, not from hard 0/1 predictions
  * the best model is chosen by PR-AUC (better than accuracy for 0.17% fraud)
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from utils.config import DATA_RAW, DATA_SAMPLE, FEATURES, ROOT, TARGET

SEED = 42


def build_models(quick, imbalance):
    cw = "balanced" if imbalance == "class_weight" else None
    trees = 50 if quick else 200
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight=cw),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED, class_weight=cw),
        "Random Forest": RandomForestClassifier(
            n_estimators=trees, random_state=SEED, n_jobs=-1, class_weight=cw),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=50 if quick else 200, learning_rate=0.1, max_depth=6,
            random_state=SEED, eval_metric="logloss", n_jobs=-1)
    except ImportError:
        print("xgboost not installed - skipping XGBoost (pip install xgboost)")
    return models


def evaluate(model, X_test, y_test, threshold=0.5):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1 Score": f1_score(y_test, pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, proba),
        "PR AUC": average_precision_score(y_test, proba),
    }


def best_model_report(model, X_test, y_test, threshold=0.5):
    """Everything the dashboard's Model Performance page plots, as plain JSON."""
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    prec, rec, _ = precision_recall_curve(y_test, proba)

    def thin(*arrays, n=400):
        idx = np.unique(np.linspace(0, len(arrays[0]) - 1, n).astype(int))
        return [np.asarray(a)[idx].round(5).tolist() for a in arrays]

    fpr, tpr = thin(fpr, tpr)
    rec, prec = thin(rec, prec)
    cm = confusion_matrix(y_test, (proba >= threshold).astype(int)).tolist()
    importances = getattr(model, "feature_importances_", None)
    return {
        "threshold": threshold,
        "test_rows": int(len(y_test)),
        "test_fraud": int(np.sum(y_test)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "confusion_matrix": cm,
        "roc": {"fpr": fpr, "tpr": tpr},
        "pr": {"recall": rec, "precision": prec},
        "feature_importance": (dict(zip(FEATURES, np.round(importances, 6).tolist()))
                               if importances is not None else None),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, default=None, help="CSV with Time,V1..V28,Amount,Class")
    ap.add_argument("--imbalance", choices=["none", "smote", "class_weight"], default="none")
    ap.add_argument("--keep-duplicates", action="store_true", help="do not drop duplicate rows")
    ap.add_argument("--quick", action="store_true", help="smaller models, for a fast check")
    ap.add_argument("--out-root", type=Path, default=ROOT, help="where models/ and reports/ go")
    a = ap.parse_args()

    path = a.data or (DATA_RAW if DATA_RAW.exists() else DATA_SAMPLE)
    print(f"Loading {path}")
    df = pd.read_csv(path)
    if not a.keep_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        print(f"Dropped {before - len(df)} duplicate rows")

    X, y = df[FEATURES].copy(), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y)

    scaler = StandardScaler().fit(X_train[["Amount"]])           # train rows only
    for part in (X_train, X_test):
        part["Amount"] = scaler.transform(part[["Amount"]]).ravel()

    if a.imbalance == "smote":
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            sys.exit("SMOTE needs imbalanced-learn: pip install imbalanced-learn")
        X_train, y_train = SMOTE(random_state=SEED).fit_resample(X_train, y_train)
        print(f"After SMOTE: {int(y_train.sum())} fraud / {int((y_train == 0).sum())} genuine")

    models_dir, reports_dir = a.out_root / "models", a.out_root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows, trained = [], {}
    for name, model in build_models(a.quick, a.imbalance).items():
        print(f"Training {name} ...")
        model.fit(X_train, y_train)
        rows.append({"Model": name, **evaluate(model, X_test, y_test)})
        trained[name] = model
        joblib.dump(model, models_dir / (name.lower().replace(" ", "_") + ".pkl"))

    results = pd.DataFrame(rows).sort_values("PR AUC", ascending=False)
    best_name = results.iloc[0]["Model"]
    joblib.dump(trained[best_name], models_dir / "best_model.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")
    results.to_csv(reports_dir / "model_comparison.csv", index=False)
    (reports_dir / "best_model_metrics.json").write_text(
        json.dumps(best_model_report(trained[best_name], X_test, y_test)))

    print(results.round(4).to_string(index=False))
    print(f"\nBest model: {best_name}  (saved to {models_dir})")


if __name__ == "__main__":
    main()
