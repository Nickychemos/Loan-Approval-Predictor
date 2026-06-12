"""Reusable inference for the housing loan-approval model.

Loads the trained, calibrated model once (lazily) and turns an application dict
into a decision with a 0-100 score and SHAP-based reasons. Used by the Django
REST Framework view (the FastAPI service has been retired).
"""
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

THRESHOLD = float(os.getenv("APPROVAL_THRESHOLD", "0.465"))
_ROOT = Path(__file__).resolve().parents[2]   # repo root (.../loan_approval_predictor)
MODEL_CANDIDATES = [
    _ROOT / "models/loan_approval_model_tuned.joblib",
    _ROOT / "models/loan_approval_model_calibrated.joblib",
    _ROOT / "models/loan_approval_model.joblib",
]
NUMERIC = ["loan_amount", "loan_to_value_ratio", "property_value", "income",
           "loan_term", "dti", "loan_to_income"]
CATEGORICAL = ["loan_type", "loan_purpose", "lien_status", "occupancy_type",
               "preapproval", "conforming_loan_limit", "total_units",
               "construction_method", "derived_loan_product_type"]

_MODEL = None
_MODEL_PATH = None

# Human-readable labels for every base feature (one-hot suffixes are stripped).
FEATURE_LABELS = {
    "dti": "debt-to-income ratio",
    "loan_to_value_ratio": "loan-to-value ratio",
    "loan_to_income": "loan size relative to income",
    "property_value": "property (collateral) value",
    "income": "income",
    "loan_amount": "requested loan amount",
    "loan_term": "loan term",
    "loan_type": "loan type",
    "loan_purpose": "loan purpose",
    "lien_status": "lien status",
    "occupancy_type": "property occupancy type",
    "preapproval": "pre-approval status",
    "conforming_loan_limit": "conforming loan limit",
    "total_units": "number of property units",
    "construction_method": "construction method",
    "derived_loan_product_type": "loan product type",
}
# Value-aware phrasing for key numerics: (unfavourable, favourable).
NUM_PHRASE = {
    "dti": ("high debt-to-income ratio", "manageable debt-to-income ratio"),
    "loan_to_value_ratio": ("high loan-to-value ratio", "healthy loan-to-value ratio"),
    "loan_to_income": ("loan amount is large relative to income",
                       "loan amount is reasonable for the income"),
    "income": ("lower income", "strong income"),
    "loan_amount": ("large requested loan amount", "modest requested loan amount"),
    "property_value": ("lower collateral value", "strong collateral value"),
}


def _load():
    global _MODEL, _MODEL_PATH
    if _MODEL is None:
        for p in MODEL_CANDIDATES:
            if p.exists():
                _MODEL, _MODEL_PATH = joblib.load(p), str(p.relative_to(_ROOT))
                break
        else:
            raise RuntimeError("No model artifact found — train one first (loan_predictor.tune).")
    return _MODEL, _MODEL_PATH


def _to_frame(app: dict) -> pd.DataFrame:
    d = dict(app)
    if d.get("loan_to_income") is None and d.get("loan_amount") and d.get("income"):
        d["loan_to_income"] = d["loan_amount"] / d["income"] if d["income"] else None
    row = {c: d.get(c) for c in NUMERIC + CATEGORICAL}
    df = pd.DataFrame([row])
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Missing categoricals -> NaN so the pipeline's imputer fills the TRAINING
    # mode (passing None/"" through breaks the one-hot encoding and the score).
    for c in CATEGORICAL:
        df[c] = df[c].apply(lambda v: np.nan if v in (None, "") else str(v)).astype("object")
    return df


def _top_reasons(df: pd.DataFrame, model, k: int = 4):
    """Best-effort per-prediction explanation via XGBoost SHAP contributions."""
    try:
        cc = model.calibrated_classifiers_[0]
        pipe = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
        Xt = pre.transform(df)
        names = list(pre.get_feature_names_out())
        import xgboost as xgb
        contribs = clf.get_booster().predict(
            xgb.DMatrix(Xt, feature_names=names), pred_contribs=True)[0]
        pairs = sorted(zip(names, contribs[:-1]), key=lambda p: abs(p[1]), reverse=True)
        return [{"feature": n.split("__")[-1], "impact": round(float(v), 3),
                 "direction": "towards approval" if v > 0 else "towards denial"}
                for n, v in pairs[:k]]
    except Exception as e:
        return [{"note": f"reasons unavailable: {type(e).__name__}"}]


def _base_feature(feat: str) -> str:
    """Map a (possibly one-hot) feature name back to its base, e.g. loan_purpose_1 -> loan_purpose."""
    for b in sorted(FEATURE_LABELS, key=len, reverse=True):
        if feat == b or feat.startswith(b + "_"):
            return b
    return feat


def _phrase(reason: dict, application: dict) -> str:
    feat = reason.get("feature", "")
    base = _base_feature(feat)
    unfavourable = reason.get("direction") == "towards denial"
    if base in NUM_PHRASE:
        text = NUM_PHRASE[base][0 if unfavourable else 1]
        val = application.get(base)
        if base in ("dti", "loan_to_value_ratio") and val is not None:
            text += f" ({val:g}%)"
        return text[0].upper() + text[1:]
    label = FEATURE_LABELS.get(base, base.replace("_", " "))
    tail = "weighed against approval" if unfavourable else "supported approval"
    return f"{label.capitalize()} {tail}"


def _humanize(reasons: list, decision: str, application: dict, score: float):
    """Turn SHAP contributions into a one-line summary + principal-reason list.

    For a denial these are the adverse-action reasons (ECOA-style): the main
    factors that counted against the application."""
    want = "towards denial" if decision == "deny" else "towards approval"
    picked = [r for r in reasons if r.get("direction") == want][:4]
    if not picked:
        picked = [r for r in reasons if "feature" in r][:3]
    explanation = [p for p in (_phrase(r, application) for r in picked) if p]
    if decision == "deny":
        head = f"Application declined (score {score}/100)."
        summary = head + (" Principal reasons: " + "; ".join(explanation) + "."
                          if explanation else "")
    else:
        head = f"Application approved (score {score}/100)."
        summary = head + (" Key supporting factors: " + "; ".join(explanation) + "."
                          if explanation else "")
    return summary, explanation


def predict(application: dict) -> dict:
    """application: dict of feature -> value. Returns decision + score + reasons."""
    model, path = _load()
    df = _to_frame(application)
    proba = float(model.predict_proba(df)[0, 1])
    decision = "approve" if proba >= THRESHOLD else "deny"
    score = round(proba * 100, 1)
    reasons = _top_reasons(df, model)
    summary, explanation = _humanize(reasons, decision, application, score)
    return {
        "decision": decision,
        "score": score,
        "probability": round(proba, 4),
        "threshold": THRESHOLD,
        "top_reasons": reasons,          # structured SHAP contributions
        "summary": summary,              # one-line plain-English statement
        "explanation": explanation,      # principal reasons (adverse-action for denials)
        "model_version": path,
    }
