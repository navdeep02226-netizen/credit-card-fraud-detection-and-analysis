# 💳 FraudShield AI – Credit Card Fraud Detection

Machine-learning fraud detection on the Kaggle *Credit Card Fraud Detection* dataset
(284,807 transactions, 492 frauds) with a multi-page Streamlit dashboard.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard/app.py                        # run from the project root
```
It works immediately using the bundled 20,000-row sample (`data/sample/`).
For the full numbers, download `creditcard.csv` from Kaggle and place it at
`data/raw/creditcard.csv` – the dashboard picks it up automatically.

## Project layout
```
dashboard/   Streamlit app (app.py, pages/, components/, assets/)
utils/       config, data loading, preprocessing, prediction, charts (no Streamlit)
models/      trained models + scaler.pkl
reports/     model_comparison.csv, best_model_metrics.json, SHAP images
notebooks/   01 EDA → 02 preprocessing → 03 training → 04 evaluation → 05 SHAP
data/        raw/ (you add the CSV), processed/, sample/ (bundled)
tests/       pytest tests for the prediction pipeline
train.py     reproducible training script
```

## Retrain
```bash
python train.py                      # all models, picks best by PR-AUC
python train.py --imbalance smote    # SMOTE on the training split only
python train.py --quick              # fast smoke test
```
This rewrites `models/` and `reports/`, which the dashboard reads.

## Test
```bash
pytest -q
```

## Results (held-out test set, 56,962 transactions, 98 frauds, threshold 0.5)
| Model | Precision | Recall | F1 | ROC AUC | PR AUC |
|---|---|---|---|---|---|
| Random Forest (deployed) | 0.942 | 0.827 | 0.880 | 0.962 | 0.873 |
| XGBoost | 0.928 | 0.786 | 0.851 | – | – |
| Decision Tree | 0.753 | 0.745 | 0.749 | 0.872 | 0.561 |
| Logistic Regression | 0.817 | 0.684 | 0.744 | 0.947 | 0.739 |

XGBoost's AUC columns are blank because its pickle could not be re-scored when this
report was built; `python train.py` fills them in.

## Notes
* The model was trained on a **standardised `Amount`**; the app applies `scaler.pkl`
  to raw amounts automatically (tick the box if your CSV is already scaled).
* Fraud is ~0.17% of rows, so judge models by recall / precision / PR-AUC, not accuracy.
* Use the **decision threshold** slider to trade missed fraud against false alarms.
* Pickles are tied to the scikit-learn / xgboost versions that created them. If loading
  fails after an upgrade, run `python train.py`.

Developed by Navdeep Singh Thakur – B.Tech CSE (AI & Data Science).
