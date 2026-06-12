"""Hyperparameter tuning for the XGBoost loan-approval model.

Strategy:
  1. SEARCH cheaply on a 120k subsample of the TRAINING data with
     RandomizedSearchCV (3-fold, scored by PR-AUC). The held-out test set is
     never seen during the search.
  2. REFIT the best settings on the FULL training set.
  3. CALIBRATE (isotonic) so probabilities stay trustworthy.
  4. EVALUATE on the held-out test set and SAVE with joblib.

Run:
    python -m loan_predictor.tune --data data/processed/hmda_clean.parquet
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.stats import randint, uniform
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss,
    precision_recall_curve, classification_report, confusion_matrix,
)
from xgboost import XGBClassifier

from loan_predictor.train import NUMERIC, CATEGORICAL, build_preprocessor

SEARCH_SAMPLE = 120_000   # default (fast); override with --sample
N_ITER = 25               # default (fast); override with --n-iter

PARAM_DIST = {
    "clf__n_estimators": randint(200, 700),
    "clf__max_depth": randint(3, 10),
    "clf__learning_rate": uniform(0.01, 0.2),
    "clf__subsample": uniform(0.6, 0.4),        # 0.6 - 1.0
    "clf__colsample_bytree": uniform(0.6, 0.4),
    "clf__min_child_weight": randint(1, 10),
    "clf__gamma": uniform(0.0, 0.5),
    "clf__reg_lambda": uniform(0.5, 2.0),
}


def best_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(thr[np.argmax(f1[:-1])]) if len(thr) else 0.5


def main(data_path: str, sample: int = SEARCH_SAMPLE, n_iter: int = N_ITER,
         out_model: str = "models/loan_approval_model_tuned.joblib"):
    print(f"Loading {data_path} ...")
    df = pd.read_parquet(data_path)
    y = df["approved"].astype(int).values
    num = [c for c in NUMERIC if c in df.columns]
    cat = [c for c in CATEGORICAL if c in df.columns]
    X = df[num + cat]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pos = float((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1)

    # --- 1. search on a subsample of the training data ---
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X_tr), size=min(sample, len(X_tr)), replace=False)
    Xs, ys = X_tr.iloc[idx], y_tr[idx]
    print(f"Searching {n_iter} configs on {len(Xs):,} rows (3-fold, scoring=PR-AUC)...")

    pipe = Pipeline([
        ("pre", build_preprocessor(num, cat)),
        ("clf", XGBClassifier(scale_pos_weight=pos, eval_metric="aucpr",
                              tree_method="hist", n_jobs=-1, random_state=42)),
    ])
    search = RandomizedSearchCV(
        pipe, PARAM_DIST, n_iter=n_iter, scoring="average_precision",
        cv=3, n_jobs=1, random_state=42, verbose=2,
    )
    search.fit(Xs, ys)
    best = {k.replace("clf__", ""): (round(v, 4) if isinstance(v, float) else int(v))
            for k, v in search.best_params_.items()}
    print(f"\nBest CV PR-AUC (subsample): {search.best_score_:.4f}")
    print("Best params:", json.dumps(best, indent=2))

    # --- 2. refit best params on FULL training set ---
    print(f"\nRefitting best params on full training set ({len(X_tr):,} rows)...")
    tuned = XGBClassifier(scale_pos_weight=pos, eval_metric="aucpr",
                          tree_method="hist", n_jobs=-1, random_state=42, **best)
    full_pipe = Pipeline([("pre", build_preprocessor(num, cat)), ("clf", tuned)])

    # --- 3. calibrate (isotonic) ---
    calibrated = CalibratedClassifierCV(full_pipe, method="isotonic", cv=3)
    calibrated.fit(X_tr, y_tr)

    # --- 4. evaluate on held-out test + save ---
    p = calibrated.predict_proba(X_te)[:, 1]
    thr = best_threshold(y_te, p)
    preds = (p >= thr).astype(int)
    pr_auc = average_precision_score(y_te, p)
    print(f"\n=== TUNED + CALIBRATED on held-out test ===")
    print(f"PR-AUC {pr_auc:.4f}   ROC-AUC {roc_auc_score(y_te, p):.4f}   "
          f"Brier {brier_score_loss(y_te, p):.4f}   threshold {thr:.3f}")
    print(confusion_matrix(y_te, preds))
    print(classification_report(y_te, preds, target_names=["denied", "approved"], digits=3))

    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    joblib.dump(calibrated, out_model)
    with open("reports/best_params.json", "w") as f:
        json.dump({"best_params": best, "search_cv_pr_auc": round(search.best_score_, 4),
                   "test_pr_auc": round(pr_auc, 4), "search_rows": len(Xs), "n_iter": n_iter},
                  f, indent=2)
    print(f"\nSaved tuned model -> {out_model}")
    print("Best params       -> reports/best_params.json")
    print(f"(Previous calibrated PR-AUC was 0.9491 — compare against {pr_auc:.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/hmda_clean.parquet")
    ap.add_argument("--sample", type=int, default=SEARCH_SAMPLE,
                    help="rows used for the search (subsample of training data)")
    ap.add_argument("--n-iter", type=int, default=N_ITER,
                    help="number of random configurations to try")
    ap.add_argument("--out-model", default="models/loan_approval_model_tuned.joblib")
    args = ap.parse_args()
    main(args.data, args.sample, args.n_iter, args.out_model)
