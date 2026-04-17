# Customer Churn Prediction — HPB Fintech Hackathon 2026

Predicting customer churn for a Croatian bank using transaction, product, balance, and contact center data.

## Approach

- **Feature window**: Apr–Dec 2025 (9 months of behavioral data)
- **Churn window**: Jan–Mar 2026 (client loses all core banking products)
- **Model**: LightGBM with SMOTE oversampling and Optuna hyperparameter tuning
- **10 interpretable features** — tenure, product count, loan flag, salary flag, transaction stats, balance stats, contact count

## Repository Structure

```
data/
  raw/              # Source CSVs (clients, products, transactions, balances, contacts)
  processed/        # Engineered feature sets
  output/           # Final churn risk scores
notebooks/
  01_eda.ipynb              # Exploratory data analysis & feature engineering
  02_data_preparation.ipynb # Missing value imputation & distribution checks
  03_model_lightgbm.ipynb   # Model training, evaluation & risk scoring
requirements.txt
```

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run the notebooks in order: `01_eda` → `02_data_preparation` → `03_model_lightgbm`.

## Results

| Metric | Value |
|--------|-------|
| ROC AUC | 0.947 |
| F1 Score | 0.367 |
| Recall | 0.692 |
| Precision | 0.250 |

The model identifies ~70% of churners while maintaining a manageable false positive rate. SHAP values provide per-feature explanations for each prediction.
