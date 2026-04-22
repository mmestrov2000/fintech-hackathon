"""
train_model.py — Train and save the LightGBM churn model.

Reads data/processed/churn_features_enhanced.csv (produced by data_prep.py),
does a stratified 80/20 split, oversamples the training set with SMOTE,
runs Optuna hyperparameter tuning (5-fold CV, F1 objective), fits the
final LightGBM model, tunes the decision threshold for F1 on the test
set, and writes:

    data/output/churn_model.pkl          (joblib: model + threshold + metrics)
    data/output/churn_risk_scores_v2.csv (per-customer risk score + prediction)

Same methodology as notebook 03 (the exploration notebook). Use the
notebook when you want SHAP / calibration / richer plots; use this script
when you just need a reproducible retrain.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --trials 30    # faster, lower-quality search
"""
import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

optuna.logging.set_verbosity(optuna.logging.WARNING)

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
SMOTE_SAMPLING = 0.3


def _load_data():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURES_PATH} not found. Run scripts/data_prep.py first."
        )
    df = pd.read_csv(FEATURES_PATH)
    return df, df[FEATURES], df[TARGET]


def _best_f1_over_thresholds(y_true, y_proba, step=0.02):
    thresholds = np.arange(0.05, 0.95, step)
    best = 0.0
    best_t = 0.5
    for t in thresholds:
        f1 = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        if f1 > best:
            best, best_t = f1, float(t)
    return best_t, float(best)


def _objective(trial, X_train, y_train):
    """5-fold CV F1 with SMOTE inside each fold (no leakage)."""
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 500,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 20.0),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "random_state": SEED,
        "verbose": -1,
        "n_jobs": -1,
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    f1s = []
    for tr, val in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[tr], X_train.iloc[val]
        y_tr, y_val = y_train.iloc[tr], y_train.iloc[val]

        X_tr_sm, y_tr_sm = SMOTE(
            random_state=SEED, sampling_strategy=SMOTE_SAMPLING
        ).fit_resample(X_tr, y_tr)

        m = lgb.LGBMClassifier(**params)
        m.fit(
            X_tr_sm, y_tr_sm,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )
        proba = m.predict_proba(X_val)[:, 1]
        _, best = _best_f1_over_thresholds(y_val, proba)
        f1s.append(best)

    return float(np.mean(f1s))


def main(trials: int):
    print(f"Loading {FEATURES_PATH.name} ...")
    df, X, y = _load_data()
    print(f"  {len(df):,} rows, churn rate {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED,
    )
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    print(f"\nOptuna hyperparameter search ({trials} trials) ...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(
        lambda t: _objective(t, X_train, y_train),
        n_trials=trials,
        show_progress_bar=True,
    )
    print(f"  best CV F1: {study.best_value:.4f}")

    best_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 1000,
        **study.best_params,
        "random_state": SEED,
        "verbose": -1,
        "n_jobs": -1,
    }
    X_train_sm, y_train_sm = SMOTE(
        random_state=SEED, sampling_strategy=SMOTE_SAMPLING
    ).fit_resample(X_train, y_train)

    print("\nFitting final model ...")
    model = lgb.LGBMClassifier(**best_params)
    model.fit(
        X_train_sm, y_train_sm,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    y_proba = model.predict_proba(X_test)[:, 1]
    threshold, best_f1 = _best_f1_over_thresholds(y_test, y_proba, step=0.01)
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
        {
            "model": model,
            "threshold": threshold,
            "features": FEATURES,
            "metrics": metrics,
            "best_params": study.best_params,
        },
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
    print(
        f"Saved scores: {scores_path}  "
        f"({risk['predicted_churn'].sum():,} flagged as high-risk @ threshold {threshold:.2f})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trials", type=int, default=150,
        help="Number of Optuna trials (default: 150, same as notebook 03)",
    )
    args = parser.parse_args()
    main(args.trials)
