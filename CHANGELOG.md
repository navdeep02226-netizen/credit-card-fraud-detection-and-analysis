# What was fixed

## Crashes / wrong behaviour in the dashboard
- `calculate_kpis` returned `Total/Fraud/...` but the KPI cards read `total/fraud/...` → **KeyError** on the Dashboard page.
- Data path `data/creditcard.csv` did not exist (file is in `data/raw/`) and every path was relative to the launch folder → now resolved from the project root.
- `load_css` / assets used relative paths and silently failed outside `dashboard/`.
- Live Prediction re-drew a **new random transaction** when you clicked Predict, so the result did not belong to the row shown → sample is kept in session state.
- Uploaded CSVs in Live Prediction were sent to the model unchecked (extra `Class` column, wrong order) → columns are validated and ordered.
- Batch results disappeared after clicking Download → results kept in session state.
- `st.progress` / probability handling mixed 0-1 and 0-100 values → cleaned up.

## Model-input bug
- The model was trained on a **standardised Amount** but the dashboard fed it the raw amount → `scaler.pkl` is now applied (`utils/preprocess.py`).

## Fake / inconsistent numbers
- Model Performance page used hard-coded metrics (e.g. Random Forest recall 0.94, ROC AUC 0.99) that did not match the real results (recall 0.83, ROC AUC 0.96). It now reads `reports/`.
- Sidebar always claimed "Model Loaded / Dataset Connected" → now real checks.

## Training pipeline (notebooks)
- 02: scaler was fitted before the split (data leakage); SMOTE output was never saved – `X_train_smote.csv` actually held the un-resampled data → rewritten (split → scale on train → SMOTE on train → save).
- 03: ROC AUC was computed from 0/1 labels; best model was hard-coded → uses probabilities and picks the best by score.
- 01/02/03 used paths that wrote 150 MB files into `notebooks/` → everything under `data/`.
- Added `train.py` to do all of the above reproducibly.

## Performance
- Dataset/model loaded on every click → cached. Charts aggregate first (a 284k-row pie/box/histogram is no longer sent to the browser). Fraud trend is per hour instead of per second.

## Project hygiene
- `utils/*.py`, `requirements.txt`, `README.md`, `.gitignore` were empty → written.
- Removed: `__pycache__`, empty `bg.jpg` / `fraud.json`, stray `notebooks/hsddccb` (a copy of scaler.pkl), empty notebook, duplicate 150 MB CSVs, `best_random_forest.pkl` (unused).
- Added a bundled sample dataset, dark theme config, tests.
