"""
train_model.py — Train the enhanced LightGBM churn model and save it.

Reads data/processed/churn_features_enhanced.csv (produced by data_prep.py),
does a stratified 80/20 split, oversamples the training set with SMOTE,
fits LightGBM with fixed sensible hyperparameters, tunes the decision
threshold for F1 on the test set, and writes:

    data/output/churn_model.pkl          (joblib-pickled model + threshold)
    data/output/churn_risk_scores_v2.csv (per-customer risk score + prediction)

This script is the reproducible counterpart to notebook 04. The notebook
also runs Optuna + SHAP + calibration — those stay there because they are
exploratory. For a quick retrain, run this file.

Usage:
    python scripts/train_model.py
"""
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / "data" / "processed" / "churn_features_enhanced.csv"
OUTPUT_DIR = ROOT / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "tenure_years", "n_products", "has_loan", "receives_salary",
    "avg_txn_per_month", "avg_txn_amount", "txn_trend",
    "avg_balance", "balance_trend", "n_contacts",
    "age", "credit_rating", "has_deposit", "digital_txn_ratio",
    "debit_credit_ratio", "balance_volatility", "txn_amount_std",
    "recency_days", "n_complaint_contacts", "avg_product_age_years",
]
TARGET = "churned"
SEED = 42

# Fixed hyperparameters — close to values Optuna picks in notebook 04.
# Tuned on the enhanced 20-feature dataset; tweak here or in the notebook.
LGBM_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    boosting_type="gbdt",
    n_estimators=1000,
    learning_rate=0.04,
    max_depth=6,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.85,
    colsample_bytree=0.75,
    reg_alpha=0.1,
    reg_lambda=0.5,
    scale_pos_weight=3.0,
    random_state=SEED,
    verbose=-1,
    n_jobs=-1,
)


def load_data():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURES_PATH} not found. Run scripts/data_prep.py first."
        )
    df = pd.read_csv(FEATURES_PATH)
    return df, df[FEATURES], df[TARGET]


def tune_threshold(y_true, y_proba):
    """Pick the decision threshold that maximizes F1 on the held-out set."""
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1s = [f1_score(y_true, (y_proba >= t).astype(int), zero_division=0) for t in thresholds]
    best = int(np.argmax(f1s))
    return float(thresholds[best]), float(f1s[best])


def main():
    print(f"Loading {FEATURES_PATH.name} ...")
    df, X, y = load_data()
    print(f"  {len(df):,} rows, churn rate {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED,
    )
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    print("SMOTE oversampling ...")
    X_train_sm, y_train_sm = SMOTE(random_state=SEED, sampling_strategy=0.3).fit_resample(
        X_train, y_train
    )

    print("Fitting LightGBM ...")
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_train_sm, y_train_sm,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    y_proba = model.predict_proba(X_test)[:, 1]
    threshold, best_f1 = tune_threshold(y_test, y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "f1": best_f1,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
    }
    print("\nTest metrics:")
    for k, v in metrics.items():
        print(f"  {k:10s} {v:.4f}")

    model_path = OUTPUT_DIR / "churn_model.pkl"
    joblib.dump(
        {"model": model, "threshold": threshold, "features": FEATURES, "metrics": metrics},
        model_path,
    )
    print(f"\nSaved model:  {model_path}")

    raw_proba = model.predict_proba(X)[:, 1]
    pmin, pmax = float(raw_proba.min()), float(raw_proba.max())
    scaled = (raw_proba - pmin) / (pmax - pmin) if pmax > pmin else raw_proba
    scaled_threshold = float(np.clip((threshold - pmin) / max(pmax - pmin, 1e-9), 0.0, 1.0))

    risk = df[["client_id"]].copy()
    risk["churn_risk_score"] = scaled
    risk["predicted_churn"] = (scaled >= scaled_threshold).astype(int)
    risk = risk.sort_values("churn_risk_score", ascending=False).reset_index(drop=True)

    scores_path = OUTPUT_DIR / "churn_risk_scores_v2.csv"
    risk.to_csv(scores_path, index=False)
    print(f"Saved scores: {scores_path}  ({risk['predicted_churn'].sum():,} flagged as high-risk)")


if __name__ == "__main__":
    main()
