"""FastAPI service that serves the trained loan-approval model.

POST /predict with an application -> returns:
  - score          : 0-100 (calibrated probability of approval x100)
  - decision        : "approve" / "deny" at the tuned threshold
  - probability     : raw calibrated P(approve)
  - top_reasons     : the features pushing the decision (XGBoost SHAP contributions)
  - model_version   : which artifact produced this

Run the server:
    PYTHONPATH=src .venv/bin/uvicorn loan_predictor.serve:app --reload
Then open http://127.0.0.1:8000/docs for an interactive form.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Tuned threshold from the calibrated model (overridable via env).
THRESHOLD = float(os.getenv("APPROVAL_THRESHOLD", "0.465"))
MODEL_CANDIDATES = [
    "models/loan_approval_model_tuned.joblib",
    "models/loan_approval_model_calibrated.joblib",
    "models/loan_approval_model.joblib",
]
NUMERIC = ["loan_amount", "loan_to_value_ratio", "property_value", "income",
           "loan_term", "dti", "loan_to_income"]
CATEGORICAL = ["loan_type", "loan_purpose", "lien_status", "occupancy_type",
               "preapproval", "conforming_loan_limit", "total_units",
               "construction_method", "derived_loan_product_type"]


def _load_model():
    for path in MODEL_CANDIDATES:
        if Path(path).exists():
            return joblib.load(path), path
    raise RuntimeError("No model artifact found — train one first (loan_predictor.train).")


MODEL, MODEL_PATH = _load_model()


class Application(BaseModel):
    """Loan application. Missing fields are imputed by the model pipeline.

    The loan is housing-backed: property_value is the value of the mortgaged
    property (the collateral) and loan_to_value_ratio = loan_amount / property_value."""
    income: Optional[float] = Field(None, description="Annual income in $")
    loan_amount: Optional[float] = Field(None, description="Requested loan amount in $")
    property_value: Optional[float] = Field(None, description="Collateral/asset value in $")
    loan_to_value_ratio: Optional[float] = Field(None, description="loan/collateral as a %")
    dti: Optional[float] = Field(None, description="Debt-to-income ratio (%)")
    loan_term: Optional[float] = Field(None, description="Term in months")
    loan_to_income: Optional[float] = Field(None, description="auto-computed if omitted")
    loan_type: Optional[str] = Field("1", description="1=Conventional 2=FHA 3=VA")
    loan_purpose: Optional[str] = Field("1", description="1=Purchase 2=Improvement 31/32=Refi")
    lien_status: Optional[str] = Field("1", description="1=first lien 2=subordinate")
    occupancy_type: Optional[str] = Field("1", description="1=principal 2=second 3=investment")
    preapproval: Optional[str] = "2"
    conforming_loan_limit: Optional[str] = "C"
    total_units: Optional[str] = "1"
    construction_method: Optional[str] = "1"
    derived_loan_product_type: Optional[str] = None


def _to_frame(app: Application) -> pd.DataFrame:
    d = app.model_dump()
    if d.get("loan_to_income") is None and d.get("loan_amount") and d.get("income"):
        d["loan_to_income"] = d["loan_amount"] / d["income"] if d["income"] else None
    row = {c: d.get(c) for c in NUMERIC + CATEGORICAL}
    df = pd.DataFrame([row])
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CATEGORICAL:
        df[c] = df[c].astype("object")
    return df


def _top_reasons(df: pd.DataFrame, k: int = 4):
    """Best-effort per-prediction explanation via XGBoost SHAP contributions."""
    try:
        # unwrap: CalibratedClassifierCV -> base pipeline -> preprocessor + xgb
        cc = MODEL.calibrated_classifiers_[0]
        pipe = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
        Xt = pre.transform(df)
        names = list(pre.get_feature_names_out())
        booster = clf.get_booster()
        import xgboost as xgb
        contribs = booster.predict(xgb.DMatrix(Xt, feature_names=names), pred_contribs=True)[0]
        pairs = sorted(zip(names, contribs[:-1]), key=lambda p: abs(p[1]), reverse=True)
        return [{"feature": n.split("__")[-1], "impact": round(float(v), 3),
                 "direction": "towards approval" if v > 0 else "towards denial"}
                for n, v in pairs[:k]]
    except Exception as e:  # explanation is best-effort, never breaks a prediction
        return [{"note": f"reasons unavailable: {type(e).__name__}"}]


app = FastAPI(title="Loan Approval Predictor", version="1.0")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH, "threshold": THRESHOLD}


@app.post("/predict")
def predict(application: Application):
    df = _to_frame(application)
    proba = float(MODEL.predict_proba(df)[0, 1])
    decision = "approve" if proba >= THRESHOLD else "deny"
    return {
        "decision": decision,
        "score": round(proba * 100, 1),
        "probability": round(proba, 4),
        "threshold": THRESHOLD,
        "top_reasons": _top_reasons(df),
        "model_version": MODEL_PATH,
    }
