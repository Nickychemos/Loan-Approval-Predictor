"""Train and compare several algorithms on cleaned HMDA data.

Model zoo: Logistic Regression, Decision Tree, Random Forest, KNN, SVM (RBF),
and XGBoost. We report the metrics that matter for imbalanced credit decisions
(PR-AUC, ROC-AUC, per-class precision/recall, confusion matrix) and tune the
decision threshold instead of trusting 0.5 / accuracy.

KNN and SVM don't scale to ~780k rows (KNN prediction and kernel-SVM training
are too expensive), so they train on a stratified SUBSAMPLE — clearly flagged
in the output. All models are evaluated on the SAME full test set.

Run:
    python -m loan_predictor.train --data data/processed/hmda_clean.parquet
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    classification_report, confusion_matrix,
)

NUMERIC = ["loan_amount", "loan_to_value_ratio", "property_value", "income",
           "loan_term", "dti", "loan_to_income"]
CATEGORICAL = ["loan_type", "loan_purpose", "lien_status", "occupancy_type",
               "preapproval", "conforming_loan_limit", "total_units",
               "construction_method", "derived_loan_product_type"]

# Cap training rows for algorithms that don't scale; None = use all rows.
SUBSAMPLE = {"KNN": 50_000, "SVM (RBF)": 20_000}


def build_preprocessor(num, cat):
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("oh", OneHotEncoder(handle_unknown="ignore", max_categories=20))])
    return ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)])


def make_models(pos_weight):
    """name -> classifier. pos_weight balances the minority (denied) class for XGBoost."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1),
        "SVM (RBF)": SVC(kernel="rbf", class_weight="balanced", probability=False, random_state=42),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, scale_pos_weight=pos_weight,
            eval_metric="aucpr", n_jobs=-1, random_state=42)
    except Exception as e:
        from sklearn.ensemble import HistGradientBoostingClassifier
        print(f"(xgboost unavailable: {e} -> using HistGradientBoosting)")
        models["HistGradientBoosting"] = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, random_state=42)
    return models


def get_scores(model, X):
    """Probability of the positive class, or decision_function score for SVM."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    s = model.decision_function(X)
    return (s - s.min()) / (s.max() - s.min() + 1e-12)  # scale to [0,1] for thresholding


def best_threshold(y_true, score):
    prec, rec, thr = precision_recall_curve(y_true, score)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(thr[np.argmax(f1[:-1])]) if len(thr) else 0.5


def evaluate(name, y_true, score, note=""):
    thr = best_threshold(y_true, score)
    preds = (score >= thr).astype(int)
    res = {
        "model": name,
        "trained_on": note or "full train set",
        "threshold": round(thr, 3),
        "pr_auc": round(average_precision_score(y_true, score), 4),
        "roc_auc": round(roc_auc_score(y_true, score), 4),
        "recall_denied": round(confusion_matrix(y_true, preds, normalize="true")[0, 0], 3),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
    }
    print(f"\n=== {name} {('['+note+']') if note else ''} ===")
    print(f"PR-AUC: {res['pr_auc']}   ROC-AUC: {res['roc_auc']}   "
          f"recall(denied): {res['recall_denied']}   thr: {res['threshold']}")
    print(classification_report(y_true, preds, target_names=["denied", "approved"], digits=3))
    return res


def main(data_path: str):
    print(f"Loading {data_path} ...")
    df = pd.read_parquet(data_path)
    y = df["approved"].astype(int).values
    num = [c for c in NUMERIC if c in df.columns]
    cat = [c for c in CATEGORICAL if c in df.columns]
    X = df[num + cat]
    print(f"  {len(df):,} rows | approved={y.mean()*100:.1f}% | {len(num)} num + {len(cat)} cat features")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pos = float((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1)
    models = make_models(pos)

    rng = np.random.RandomState(42)
    results, fitted = [], {}
    for name, clf in models.items():
        cap = SUBSAMPLE.get(name)
        if cap and cap < len(X_tr):
            idx = rng.choice(len(X_tr), size=cap, replace=False)
            Xf, yf = X_tr.iloc[idx], y_tr[idx]
            note = f"subsample {cap:,} rows"
        else:
            Xf, yf, note = X_tr, y_tr, ""
        pipe = Pipeline([("pre", build_preprocessor(num, cat)), ("clf", clf)])
        t0 = time.time()
        pipe.fit(Xf, yf)
        secs = time.time() - t0
        score = get_scores(pipe, X_te)
        res = evaluate(name, y_te, score, note)
        res["fit_seconds"] = round(secs, 1)
        print(f"   (fit time: {secs:.1f}s)")
        results.append(res)
        fitted[name] = pipe

    # leaderboard
    results.sort(key=lambda r: r["pr_auc"], reverse=True)
    print("\n================ LEADERBOARD (by PR-AUC) ================")
    print(f"{'model':<22}{'PR-AUC':>9}{'ROC-AUC':>9}{'rec(den)':>10}{'fit(s)':>9}  train")
    for r in results:
        print(f"{r['model']:<22}{r['pr_auc']:>9}{r['roc_auc']:>9}"
              f"{r['recall_denied']:>10}{r['fit_seconds']:>9}  {r['trained_on']}")

    best = results[0]
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    joblib.dump(fitted[best["model"]], "models/loan_approval_model.joblib")
    with open("reports/metrics.json", "w") as f:
        json.dump({"results": results, "best": best["model"]}, f, indent=2)
    print(f"\nBest: {best['model']} (PR-AUC {best['pr_auc']}) -> models/loan_approval_model.joblib")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/hmda_clean.parquet")
    args = ap.parse_args()
    main(args.data)
