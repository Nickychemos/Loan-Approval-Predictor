"""Rigorous evaluation: cross-validation + probability calibration.

Two stages:
  1. CROSS-VALIDATION — stratified 5-fold comparison of all 6 algorithms, so the
     ranking comes with a mean +/- std rather than a single lucky train/test split.
     Run on a common subsample (SVM/RF don't scale to ~780k x 5 folds), which is
     plenty for a stable *comparison*.
  2. CALIBRATION — retrain the winner on the FULL training set, wrap it in isotonic
     calibration so predicted probabilities are trustworthy (a 0.8 score really means
     ~80% approval), and report the Brier score before vs after + a reliability curve.

Run:
    python -m loan_predictor.evaluate --data data/processed/hmda_clean.parquet
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    make_scorer, recall_score, precision_score, f1_score, brier_score_loss,
    average_precision_score, roc_auc_score, precision_recall_curve,
    classification_report, confusion_matrix,
)

from loan_predictor.train import (
    NUMERIC, CATEGORICAL, build_preprocessor, make_models,
)

CV_SAMPLE = 20_000   # common subsample for the fair cross-validated comparison
N_FOLDS = 5


def cross_validate_models(X, y, num, cat):
    rng = np.random.RandomState(42)
    if len(X) > CV_SAMPLE:
        idx = rng.choice(len(X), size=CV_SAMPLE, replace=False)
        Xs, ys = X.iloc[idx], y[idx]
    else:
        Xs, ys = X, y
    print(f"Cross-validating on {len(Xs):,} rows, {N_FOLDS}-fold stratified.\n")

    pos = float((ys == 0).sum()) / max((ys == 1).sum(), 1)
    models = make_models(pos)
    # Per-class precision/recall/f1 are at each model's DEFAULT decision threshold,
    # i.e. exactly what classification_report shows, averaged across the 5 folds.
    scoring = {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "precision_denied": make_scorer(precision_score, pos_label=0, zero_division=0),
        "recall_denied": make_scorer(recall_score, pos_label=0),
        "f1_denied": make_scorer(f1_score, pos_label=0),
        "precision_approved": make_scorer(precision_score, pos_label=1, zero_division=0),
        "recall_approved": make_scorer(recall_score, pos_label=1),
        "f1_approved": make_scorer(f1_score, pos_label=1),
    }
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    rows = []
    for name, clf in models.items():
        pipe = Pipeline([("pre", build_preprocessor(num, cat)), ("clf", clf)])
        cv = cross_validate(pipe, Xs, ys, cv=skf, scoring=scoring, n_jobs=1)
        m = lambda k: round(cv[f"test_{k}"].mean(), 4)
        s = lambda k: round(cv[f"test_{k}"].std(), 4)
        row = {
            "model": name,
            "pr_auc_mean": m("pr_auc"), "pr_auc_std": s("pr_auc"),
            "roc_auc_mean": m("roc_auc"), "accuracy_mean": m("accuracy"),
            "precision_denied_mean": m("precision_denied"), "recall_denied_mean": m("recall_denied"),
            "recall_denied_std": s("recall_denied"), "f1_denied_mean": m("f1_denied"),
            "precision_approved_mean": m("precision_approved"), "recall_approved_mean": m("recall_approved"),
            "f1_approved_mean": m("f1_approved"),
        }
        rows.append(row)
        print(f"  {name}  (PR-AUC {row['pr_auc_mean']:.4f} +/- {row['pr_auc_std']:.4f}, "
              f"ROC-AUC {row['roc_auc_mean']:.4f}, accuracy {row['accuracy_mean']:.3f})")
        print(f"      {'class':<10}{'precision':>11}{'recall':>9}{'f1':>8}")
        print(f"      {'denied':<10}{row['precision_denied_mean']:>11.3f}"
              f"{row['recall_denied_mean']:>9.3f}{row['f1_denied_mean']:>8.3f}")
        print(f"      {'approved':<10}{row['precision_approved_mean']:>11.3f}"
              f"{row['recall_approved_mean']:>9.3f}{row['f1_approved_mean']:>8.3f}")

    rows.sort(key=lambda r: r["pr_auc_mean"], reverse=True)
    print("\n===================== CROSS-VALIDATED LEADERBOARD (5-fold means) =====================")
    print(f"{'model':<22}{'PR-AUC':>16}{'ROC-AUC':>9}{'acc':>7}"
          f"{'prec(den)':>11}{'rec(den)':>10}{'prec(app)':>11}{'rec(app)':>10}")
    for r in rows:
        print(f"{r['model']:<22}{r['pr_auc_mean']:>9.4f}+/-{r['pr_auc_std']:<6.4f}"
              f"{r['roc_auc_mean']:>9.3f}{r['accuracy_mean']:>7.3f}"
              f"{r['precision_denied_mean']:>11.3f}{r['recall_denied_mean']:>10.3f}"
              f"{r['precision_approved_mean']:>11.3f}{r['recall_approved_mean']:>10.3f}")
    return rows


def best_threshold(y_true, proba):
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(thr[np.argmax(f1[:-1])]) if len(thr) else 0.5


def calibrate_winner(winner_name, X, y, num, cat):
    print(f"\n=========== CALIBRATION: {winner_name} on full data ===========")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pos = float((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1)
    clf = make_models(pos)[winner_name]
    base = Pipeline([("pre", build_preprocessor(num, cat)), ("clf", clf)])

    # uncalibrated
    base.fit(X_tr, y_tr)
    p_raw = base.predict_proba(X_te)[:, 1]
    brier_raw = brier_score_loss(y_te, p_raw)

    # isotonic calibration (3-fold internal)
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
    calibrated.fit(X_tr, y_tr)
    p_cal = calibrated.predict_proba(X_te)[:, 1]
    brier_cal = brier_score_loss(y_te, p_cal)

    thr = best_threshold(y_te, p_cal)
    preds = (p_cal >= thr).astype(int)
    print(f"Brier score (lower=better):  raw {brier_raw:.4f}  ->  calibrated {brier_cal:.4f}")
    print(f"PR-AUC {average_precision_score(y_te, p_cal):.4f}   "
          f"ROC-AUC {roc_auc_score(y_te, p_cal):.4f}   threshold {thr:.3f}")
    print(confusion_matrix(y_te, preds))
    print(classification_report(y_te, preds, target_names=["denied", "approved"], digits=3))

    frac_pos, mean_pred = calibration_curve(y_te, p_cal, n_bins=10)
    report = {
        "winner": winner_name,
        "brier_raw": round(brier_raw, 4),
        "brier_calibrated": round(brier_cal, 4),
        "pr_auc": round(average_precision_score(y_te, p_cal), 4),
        "roc_auc": round(roc_auc_score(y_te, p_cal), 4),
        "threshold": round(thr, 3),
        "reliability_curve": {
            "mean_predicted": [round(float(v), 3) for v in mean_pred],
            "fraction_positive": [round(float(v), 3) for v in frac_pos],
        },
    }
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    joblib.dump(calibrated, "models/loan_approval_model_calibrated.joblib")
    with open("reports/calibration.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved calibrated model -> models/loan_approval_model_calibrated.joblib")
    print("Calibration report   -> reports/calibration.json")
    return report


def main(data_path: str, winner: str | None):
    print(f"Loading {data_path} ...")
    df = pd.read_parquet(data_path)
    y = df["approved"].astype(int).values
    num = [c for c in NUMERIC if c in df.columns]
    cat = [c for c in CATEGORICAL if c in df.columns]
    X = df[num + cat]
    print(f"  {len(df):,} rows | approved={y.mean()*100:.1f}%\n")

    cv_rows = cross_validate_models(X, y, num, cat)
    # Calibrate on FULL data, so only pick a model that scales (not KNN/SVM).
    not_scalable = {"KNN", "SVM (RBF)"}
    if winner:
        chosen = winner
    else:
        chosen = next(r["model"] for r in cv_rows if r["model"] not in not_scalable)
        if chosen != cv_rows[0]["model"]:
            print(f"\n(Note: top CV model {cv_rows[0]['model']} doesn't scale to full-data "
                  f"calibration; calibrating best scalable model: {chosen}.)")
    calibrate_winner(chosen, X, y, num, cat)

    with open("reports/cv_results.json", "w") as f:
        json.dump(cv_rows, f, indent=2)
    print("CV leaderboard       -> reports/cv_results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/hmda_clean.parquet")
    ap.add_argument("--winner", default=None,
                    help="Force which model to calibrate (default: top CV model). "
                         "Note: SVM/KNN don't scale to full-data calibration.")
    args = ap.parse_args()
    main(args.data, args.winner)
