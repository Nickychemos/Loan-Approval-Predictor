# Loan Approval Predictor

Asset-backed loan-approval model trained on **real**
US HMDA mortgage data — every row is a genuine application with the lender's actual
approve/deny decision, plus collateral value and loan-to-value (LTV).

See the design doc: `../Loan_Approval_Prediction_Project_Draft.docx`.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Data

Real HMDA data, pulled from the CFPB data-browser API (no auth):

```bash
# multi-state 2023, approved (1) + denied (3)
curl -L "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?states=MD,VA,CO,OR,TN,MO&years=2023&actions_taken=1,3" -o data/raw/hmda_multistate_2023.csv
```

## Pipeline

```bash
# 1. Clean raw HMDA -> tidy parquet (parses DTI bands, fixes income units,
#    winsorises outliers, drops label-leaking post-origination fields)
PYTHONPATH=src .venv/bin/python -m loan_predictor.clean --input data/raw/hmda_multistate_2023.csv

# 2. Train + compare 6 algorithms (LR, Decision Tree, Random Forest, KNN, SVM, XGBoost)
PYTHONPATH=src .venv/bin/python -m loan_predictor.train

# 3. Rigorous: 5-fold cross-validation + isotonic calibration of the winner
PYTHONPATH=src .venv/bin/python -m loan_predictor.evaluate

# 4. Hyperparameter-tune XGBoost (RandomizedSearchCV), refit on full data,
#    recalibrate, and save the best model
PYTHONPATH=src .venv/bin/python -m loan_predictor.tune
```

**Canonical model: `models/loan_approval_model_tuned.joblib`** (tuned + calibrated
XGBoost, test PR-AUC 0.9513). Best params in `reports/best_params.json`.

## v1 results (975,986 rows, 76% approved / 24% denied)

Ranked by PR-AUC (imbalance-robust). `rec(denied)` = recall on the denied class
(how many bad applicants we catch). KNN/SVM train on a subsample (they don't scale).

| Model | PR-AUC | ROC-AUC | rec(denied) | trained on |
|-------|--------|---------|-------------|------------|
| **XGBoost** (saved) | **0.953** | **0.885** | 0.553 | full |
| Random Forest | 0.952 | 0.885 | 0.557 | full |
| Decision Tree | 0.940 | 0.871 | 0.541 | full |
| SVM (RBF) | 0.933 | 0.846 | 0.397 | 20k subsample |
| KNN | 0.930 | 0.846 | 0.446 | 50k subsample |
| Logistic Regression | 0.920 | 0.807 | 0.311 | full |

Best model -> `models/loan_approval_model.joblib`; full metrics -> `reports/metrics.json`.
We report PR-AUC / per-class precision-recall and tune the decision threshold rather
than trusting accuracy, because the classes are imbalanced.

### Cross-validated & calibrated (`evaluate.py`)

5-fold stratified CV (20k common subsample) confirms the ranking is stable, not a
lucky split — XGBoost (0.9458 ± 0.0020) and Random Forest (0.9449 ± 0.0021) are tied
at the top. The lone Decision Tree drops to 0.873 once cross-validated (it was
overfitting on the single split).

The winner (XGBoost) is then **calibrated** on full data with isotonic regression:
Brier score improves **0.127 → 0.099**, so predicted probabilities are trustworthy
(a 0.80 score ≈ 80% chance) — important for the score-vs-threshold decision logic.

Artifacts: `models/loan_approval_model_calibrated.joblib`, `reports/cv_results.json`,
`reports/calibration.json` (includes the reliability curve).

## Serving (Django REST Framework + Postgres)

The model is served through a Django/DRF API backed by PostgreSQL (run via Docker).
`src/loan_predictor/predictor.py` holds the reusable inference; the DRF view scores
each application and persists the decision.

```bash
make setup     # install deps + start Postgres (Docker) + run migrations
make run       # start Postgres, migrate, and serve the API (http://127.0.0.1:8001)
make test      # run the API + model test suite
```

Key endpoints (JWT-authenticated; CORS enabled for a React SPA):

| Method & path | Purpose |
|---|---|
| `POST /api/register/` | Create a user account |
| `POST /api/token/` `POST /api/token/refresh/` | Obtain / refresh a JWT |
| `GET /api/me/` | Current user |
| `GET/POST /api/applications/` | Submit & list housing loan applications |
| `/admin/` | Django admin (users, applications, decisions, audit) |

Submitting an application returns a 0-100 score, an approve/deny decision at the
tuned threshold, the top SHAP reasons, and a plain-English `summary` + `explanation`
(adverse-action reasons for denials). The loan is housing-backed: `property_value` is
the mortgaged property's value (collateral) and `loan_to_value_ratio` = loan / property
value. Example body:

```json
{"income": 180000, "loan_amount": 200000, "property_value": 500000,
 "loan_to_value_ratio": 40, "dti": 18, "loan_term": 360}
```

## Next (per design doc)

Done: DRF API + JWT/CORS + adverse-action reasons + Postgres persistence + Docker
(Postgres). Remaining: Prefect orchestration → Evidently drift monitoring → Dockerize
the Django service → React SPA frontend. Richer applicant features (credit score, prior
defaults) are the main lever to lift denied-class recall.
