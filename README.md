# Customer Churn Prediction — HPB Fintech Hackathon 2026

Predicting customer churn for a Croatian bank using client, product, transaction, balance, and contact-center data — and selecting the **optimal retention action** per customer.

This repo is a **research-style notebook project**. Each step of the pipeline lives in its own notebook or a small companion script; there is no application code to deploy. Use [notebooks/00_overview.ipynb](notebooks/00_overview.ipynb) as the single entry point.

## Approach

### Model 1 — Churn Prediction (LightGBM)

- **Feature window**: Apr–Dec 2025 (9 months of behavioral data)
- **Churn window**: Jan–Mar 2026
- **At-risk population**: clients with any active product on 2026-01-01
- **Churn definition**: client had a core product at reference date and lost all core products (accounts, loans, deposits) by 2026-03-31
- **Features**: 20 features (10 baseline + 10 added)
- **Model**: LightGBM with SMOTE oversampling, Optuna hyperparameter tuning, F1-optimal threshold, and SHAP explainability

### Model 2 — Contextual Bandit for Retention Actions

- **Architecture**: Ridge reward model per action ($\hat{\mu}_a(x) = \theta_a^\top x$)
- **Actions**: no action, push notification (€0.50), email (€8), call (€80)
- **Reward**: $R_i = (1 - Y_i) \cdot v_i - c(A_i)$ (retained revenue minus action cost)
- **Training**: 24-month cumulative Ridge refit with ε-greedy exploration
- **Key features**: cost-aware regularization, progressive call gating, HTE-aware context vectors

## Repository Structure

```
data/
  raw/              # Source CSVs (clients, products, transactions, balances, contacts) — gitignored
  processed/        # Engineered feature sets
  output/           # Risk scores, trained model, bandit training data, visualizations
docs/
  documentation/                # PDFs + LaTeX sources for the write-ups (churn, bandit, pitch)
  presentation/                 # Slide deck
  hackathon_statement/          # Original problem statement + ER diagram
notebooks/
  00_overview.ipynb             # Pipeline overview + links to each step (start here)
  01_eda.ipynb                  # EDA, churn labeling, initial feature engineering
  02_data_preparation.ipynb     # Missing-value handling and feature checks
  03_lightgbm_churn.ipynb       # LightGBM churn model — 20 features, Optuna + SHAP
  04_contextual_bandit.ipynb    # Contextual bandit for retention action selection
scripts/
  data_prep.py                  # Raw CSVs → processed feature datasets
  train_model.py                # Train + save the LightGBM churn model (Optuna + F1 threshold)
  generate_bandit_dataset.py    # Generates synthetic bandit training data from churn scores
requirements.txt
```

## How to Navigate

- **Want the big picture?** Open [notebooks/00_overview.ipynb](notebooks/00_overview.ipynb).
- **Want to see how features are built?** [01_eda.ipynb](notebooks/01_eda.ipynb) → [02_data_preparation.ipynb](notebooks/02_data_preparation.ipynb).
- **Want to iterate on the churn model?** [03_lightgbm_churn.ipynb](notebooks/03_lightgbm_churn.ipynb) (full narrative: Optuna, SHAP, calibration, plots).
- **Want to change retention actions / costs / rewards?** Edit [scripts/generate_bandit_dataset.py](scripts/generate_bandit_dataset.py), regenerate the dataset, then re-run [04_contextual_bandit.ipynb](notebooks/04_contextual_bandit.ipynb).

## Features (20)

**Baseline (10)** — `tenure_years`, `n_products`, `has_loan`, `receives_salary`, `avg_txn_per_month`, `avg_txn_amount`, `txn_trend`, `avg_balance`, `balance_trend`, `n_contacts`

**Added (10)** — `age`, `credit_rating`, `has_deposit`, `digital_txn_ratio`, `debit_credit_ratio`, `balance_volatility`, `txn_amount_std`, `recency_days`, `n_complaint_contacts`, `avg_product_age_years`

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running the pipeline

The repo offers **two interoperable entry points** that produce the same CSVs. Use whichever fits your task.

### Option A — reproducible scripts (recommended for re-runs)

```bash
# 1. Rebuild all processed feature CSVs from raw tables
python scripts/data_prep.py
#    → data/processed/churn_features_raw.csv        (10 features, pre-imputation)
#    → data/processed/churn_features_clean.csv      (10 features, imputed)
#    → data/processed/churn_features_enhanced.csv   (20 features, imputed)

# 2. Train the LightGBM churn model (Optuna + F1-optimal threshold)
python scripts/train_model.py                  # 150 Optuna trials (default)
python scripts/train_model.py --trials 30      # faster, lower-quality search
#    → data/output/churn_model.pkl              (joblib: model + threshold + metrics)
#    → data/output/churn_risk_scores_v2.csv     (per-customer risk score + prediction)

# 3. (Optional) Generate the contextual-bandit training dataset
python scripts/generate_bandit_dataset.py
#    → data/output/bandit_training_dataset.csv
```

`train_model.py` runs the same methodology as [notebook 03](notebooks/03_lightgbm_churn.ipynb) — Optuna hyperparameter search (5-fold CV, F1 objective) and F1-optimal threshold. The notebook additionally produces SHAP plots, calibration plots, and richer visualisations. Pick the notebook when you want to inspect a model; pick the script when you just need a fresh `.pkl`.

### Option B — notebooks (recommended for exploration / teaching)

Run the notebooks top-to-bottom:

`00_overview` (read first) → `01_eda` → `02_data_preparation` → `03_lightgbm_churn` → `04_contextual_bandit`

Each notebook writes to the same `data/processed/` and `data/output/` paths as the scripts, so you can mix and match freely (e.g. run `data_prep.py` once, then iterate in notebook 03).

### How notebooks fit into the workflow

- `00_overview.ipynb` — entry point: architecture diagram, pipeline walkthrough, full results table, what worked / what didn't / what to try next.
- `01_eda.ipynb` — EDA and visual feature construction (teaching version of `data_prep.py`).
- `02_data_preparation.ipynb` — distribution, missing-value, and correlation checks.
- `03_lightgbm_churn.ipynb` — the *narrative* version of `train_model.py`: Optuna, SHAP, calibration, full plots.
- `04_contextual_bandit.ipynb` — bandit training + evaluation against static baselines.

## Results

### Churn model — [notebook 03](notebooks/03_lightgbm_churn.ipynb) / [scripts/train_model.py](scripts/train_model.py)

| Metric | Value |
|--------|-------|
| ROC AUC | 0.942 |
| PR AUC | 0.367 |
| Precision | 0.636 |
| Recall | 0.538 |
| F1 Score | 0.583 |

Top drivers (gain-based importance): `avg_balance`, `age`, `tenure_years`, `credit_rating`, `avg_product_age_years`, `recency_days`, `balance_volatility`.

### Contextual Bandit — [notebook 04](notebooks/04_contextual_bandit.ipynb)

The learned policy outperforms all static baselines on holdout (last 6 months):

- Beats "always no action" — targeted interventions save revenue
- Beats "always call" — avoids wasting €80 on low-value customers
- Calls concentrate on high-value, high-risk customers (D10 call rate ≫ D1)
- Model learns progressively: call fraction rises from ~0% (months 1–6) to steady state (months 19–24)

## Outputs

- `data/output/churn_model.pkl` — trained LightGBM + F1-optimal threshold + features + metrics (joblib)
- `data/output/churn_risk_scores_v2.csv` — per-customer risk score + prediction at the saved threshold
- `data/output/bandit_training_dataset.csv` — bandit training data (generated)
- `data/output/bandit_*.png` — bandit visualizations (learning curves, heatmaps, policy comparison)

The churn pipeline supports threshold tuning depending on campaign economics:

- lower threshold → broader, recall-oriented retention campaigns
- higher threshold → precision-oriented, expensive interventions

The contextual bandit selects the cost-optimal retention action per customer, balancing intervention cost against churn-reduction benefit.
