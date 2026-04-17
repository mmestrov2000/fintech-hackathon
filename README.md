# Customer Churn Prediction — HPB Fintech Hackathon 2026

Predicting customer churn for a Croatian bank using client, product, transaction, balance, and contact-center data.

## Approach

- **Feature window**: Apr-Dec 2025 (9 months of behavioral data)
- **Churn window**: Jan-Mar 2026
- **At-risk population**: clients with any active product on 2026-01-01
- **Churn definition**: client had a core product at reference date and lost all core products (accounts, loans, deposits) by 2026-03-31
- **Model**: LightGBM with SMOTE oversampling, Optuna hyperparameter tuning, threshold optimization, and SHAP explainability

The work evolved in two modeling stages:

- **Baseline model**: 10 interpretable features
- **Enhanced model**: 20 features total, adding 10 new signals from unused raw columns

## Repository Structure

```
data/
  raw/              # Source CSVs (clients, products, transactions, balances, contacts)
  processed/        # Engineered feature sets
  output/           # Risk scores and model outputs
docs/
  fintech_hackathon.pdf     # Presentation / final document
notebooks/
  01_eda.ipynb                     # EDA, churn labeling, initial feature engineering
  02_data_preparation.ipynb        # Missing value handling and feature checks
  03_model_lightgbm.ipynb          # Baseline LightGBM model with 10 features
  04_enhanced_features_model.ipynb # Enhanced LightGBM model with 20 features
requirements.txt
```

## Feature Sets

### Original 10 Features

- `tenure_years`
- `n_products`
- `has_loan`
- `receives_salary`
- `avg_txn_per_month`
- `avg_txn_amount`
- `txn_trend`
- `avg_balance`
- `balance_trend`
- `n_contacts`

### Added 10 Features

- `age`
- `credit_rating`
- `has_deposit`
- `digital_txn_ratio`
- `debit_credit_ratio`
- `balance_volatility`
- `txn_amount_std`
- `recency_days`
- `n_complaint_contacts`
- `avg_product_age_years`

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run the notebooks in order:

`01_eda` -> `02_data_preparation` -> `03_model_lightgbm` -> `04_enhanced_features_model`

## Results

### Baseline Model: Notebook 03

| Metric | Value |
|--------|-------|
| F1 Score | 0.386 |

### Enhanced Model: Notebook 04

| Metric | Value |
|--------|-------|
| ROC AUC | 0.942 |
| PR AUC | 0.367 |
| Precision | 0.636 |
| Recall | 0.538 |
| F1 Score | 0.583 |

This improves F1 from **0.386** to **0.583**, a **+51.1%** gain over the baseline.

## Most Impactful Features

Based on the enhanced LightGBM model, the strongest churn drivers were:

1. `avg_balance`
2. `age`
3. `tenure_years`

Other strong new signals included `credit_rating`, `avg_product_age_years`, `recency_days`, and `balance_volatility`.

## Outputs

- `data/output/churn_risk_scores.csv` — baseline model scores
- `data/output/churn_risk_scores_v2.csv` — enhanced model scores

The final pipeline supports threshold tuning depending on campaign economics:

- lower threshold for broader recall-oriented retention campaigns
- higher threshold for precision-oriented expensive interventions
