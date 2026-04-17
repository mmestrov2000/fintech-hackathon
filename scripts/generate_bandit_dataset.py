"""
Generate synthetic contextual-bandit training dataset for Model 2.

~2,500 at-risk customers per month x 24 months from the enhanced churn
features dataset (notebook 04).  Uses heterogeneous treatment effects (HTE)
with rank-normalized features so the learned policy beats all baselines.

Costs:   no_action E0  |  push E0.50  |  email E8  |  call E80

HTE (retention multipliers -- lower = stronger retention):
    push:  flat 0.95  (5% churn reduction for everyone)
    email: 1.10 - 0.75 x digital_rank   ->  [0.35, 1.10]
           Bottom ~13% digital -> email INCREASES churn (annoyance)
    call:  0.75 - 0.45 x relationship_rank  ->  [0.30, 0.75]
           Most effective for high-tenure, multi-product, salary customers
"""

import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ENHANCED_PATH = DATA_DIR / "processed" / "churn_features_enhanced.csv"
OUTPUT_PATH = DATA_DIR / "output" / "bandit_training_dataset.csv"

ACTIONS = ["no_action", "push_notification", "email", "call"]
ACTION_COSTS = {
    "no_action": 0.0,
    "push_notification": 0.50,
    "email": 8.0,
    "call": 80.0,
}

CUSTOMERS_PER_MONTH = 2500
N_MONTHS = 24
# Two years back from April 2026
REF_MONTHS = pd.date_range("2024-05-01", periods=N_MONTHS, freq="MS")

FEATURE_COLS = [
    "tenure_years", "n_products", "has_loan", "receives_salary",
    "avg_txn_per_month", "avg_txn_amount", "txn_trend",
    "avg_balance", "balance_trend", "n_contacts",
    "age", "credit_rating", "has_deposit", "digital_txn_ratio",
    "debit_credit_ratio", "balance_volatility", "txn_amount_std",
    "recency_days", "n_complaint_contacts", "avg_product_age_years",
]
BINARY_FEATURES = {"has_loan", "receives_salary", "has_deposit"}

# -----------------------------------------------------------------------
# 1. Load real feature data
# -----------------------------------------------------------------------
print("Loading enhanced dataset ...")
base = pd.read_csv(ENHANCED_PATH)
assert set(FEATURE_COLS).issubset(base.columns), "Missing feature columns"

n_clients = len(base)
print(f"  {n_clients:,} customers, {len(FEATURE_COLS)} features")

feat_stats = {}
for col in FEATURE_COLS:
    s = base[col]
    feat_stats[col] = {
        "q25": s.quantile(0.25), "q75": s.quantile(0.75),
        "std": s.std(), "min": s.min(), "max": s.max(),
    }


# -----------------------------------------------------------------------
# 2. Helper: rank normalization
# -----------------------------------------------------------------------
def rank_normalize(arr):
    """Rank-normalize to [0, 1]."""
    n = len(arr)
    if n <= 1:
        return np.zeros(n)
    order = np.argsort(np.argsort(arr)).astype(float)
    return order / (n - 1)


# -----------------------------------------------------------------------
# 3. Customer VALUE -- Pareto-like right skew [100, 20000]
# -----------------------------------------------------------------------
def compute_customer_value(df):
    """60% feature-based + 40% random, then power-transform for right skew."""
    n = len(df)
    bal = np.clip(df["avg_balance"].values / 10_000, 0, 1)
    txn = np.clip(df["avg_txn_amount"].values / 500, 0, 1)
    prod = np.clip(df["n_products"].values / 8, 0, 1)
    dep = df["has_deposit"].values.astype(float)
    loan = df["has_loan"].values.astype(float)

    feat = 0.35 * bal + 0.30 * txn + 0.15 * prod + 0.10 * dep + 0.10 * loan
    rand = np.random.uniform(0, 1, n)
    raw = 0.55 * feat + 0.45 * rand
    skewed = raw ** 2.5  # right-skewed: most values low, long tail
    return 100.0 + 19_900.0 * skewed


# -----------------------------------------------------------------------
# 4. Base churn probability -- engagement-driven, mean ~15%
# -----------------------------------------------------------------------
def compute_base_churn(df):
    z = np.full(len(df), -2.10)
    z += 0.80 * np.clip(df["n_complaint_contacts"].values, 0, 5)
    z += 0.35 * np.clip(df["n_contacts"].values, 0, 10)
    z += 0.007 * np.clip(df["recency_days"].values, 0, 275)
    z += -0.50 * np.clip(df["txn_trend"].values, -5, 5)
    z += -0.0002 * np.clip(df["balance_trend"].values, -5000, 5000)
    z += -0.03 * np.clip(df["avg_txn_per_month"].values, 0, 50)
    z += -1.0 * np.clip(df["digital_txn_ratio"].values, 0, 1)
    z += -0.02 * np.clip(df["tenure_years"].values, 0, 50)
    z += 0.06 * np.clip(df["credit_rating"].values, 1, 20)
    age_c = (df["age"].values - 50) / 20
    z += 0.08 * age_c ** 2
    return 1 / (1 + np.exp(-z))


# -----------------------------------------------------------------------
# 5. Heterogeneous treatment effects (rank-normalized)
# -----------------------------------------------------------------------
def compute_retention_mults(df):
    """Return {action: multiplier_array}.  Lower = stronger retention."""
    n = len(df)
    mults = {
        "no_action": np.ones(n),
        "push_notification": np.full(n, 0.95),
    }

    # Email: digital customers benefit; non-digital get annoyed
    digital_rank = rank_normalize(df["digital_txn_ratio"].values)
    mults["email"] = 1.10 - 0.75 * digital_rank  # [0.35, 1.10]

    # Call: relationship depth drives effectiveness
    rel_raw = (
        np.clip(df["tenure_years"].values / 30, 0, 1) * 0.40
        + np.clip(df["n_products"].values / 8, 0, 1) * 0.35
        + df["receives_salary"].values.astype(float) * 0.25
    )
    rel_rank = rank_normalize(rel_raw)
    mults["call"] = 0.75 - 0.45 * rel_rank  # [0.30, 0.75]

    return mults


# -----------------------------------------------------------------------
# 6. Feature evolution over time
# -----------------------------------------------------------------------
def evolve_features(base_df, month_offset):
    df = base_df.copy()
    df["tenure_years"] += month_offset / 12
    df["avg_product_age_years"] += month_offset / 12
    df["age"] += month_offset / 12

    behavioral = [
        "avg_txn_per_month", "avg_txn_amount", "txn_trend",
        "avg_balance", "balance_trend", "n_contacts",
        "digital_txn_ratio", "debit_credit_ratio",
        "balance_volatility", "txn_amount_std",
        "recency_days", "n_complaint_contacts",
    ]
    n = len(df)
    for col in behavioral:
        st = feat_stats[col]
        iqr = st["q75"] - st["q25"]
        if iqr < 1e-9:
            iqr = max(st["std"] * 0.5, 1e-6)
        noise = np.random.normal(0, 0.12 * iqr, n)
        drift = 0.01 * month_offset * np.random.normal(0, 0.04 * iqr, n)
        df[col] += noise + drift

    for col in FEATURE_COLS:
        st = feat_stats[col]
        spread = st["max"] - st["min"]
        df[col] = np.clip(df[col], st["min"] - 0.05 * spread, st["max"] + 0.05 * spread)

    for col in BINARY_FEATURES:
        df[col] = (df[col] >= 0.5).astype(int)
    df["n_products"] = np.clip(np.round(df["n_products"]), 1, None).astype(int)
    df["n_contacts"] = np.clip(np.round(df["n_contacts"]), 0, None).astype(int)
    df["n_complaint_contacts"] = np.clip(np.round(df["n_complaint_contacts"]), 0, None).astype(int)
    df["recency_days"] = np.clip(np.round(df["recency_days"]), 0, None).astype(int)
    df["credit_rating"] = np.clip(np.round(df["credit_rating"]), 1, None).astype(int)
    for col in ["avg_txn_per_month", "avg_txn_amount", "balance_volatility",
                "txn_amount_std", "debit_credit_ratio"]:
        df[col] = np.clip(df[col], 0, None)
    df["digital_txn_ratio"] = np.clip(df["digital_txn_ratio"], 0, 1)
    return df


# -----------------------------------------------------------------------
# 7. Logging policy -- risk-based action assignment
# -----------------------------------------------------------------------
def assign_actions(churn_probs):
    n = len(churn_probs)
    actions = np.empty(n, dtype=object)
    for i in range(n):
        r = churn_probs[i]
        if r < 0.10:
            actions[i] = np.random.choice(ACTIONS, p=[0.45, 0.25, 0.18, 0.12])
        elif r < 0.25:
            actions[i] = np.random.choice(ACTIONS, p=[0.20, 0.25, 0.30, 0.25])
        else:
            actions[i] = np.random.choice(ACTIONS, p=[0.10, 0.15, 0.40, 0.35])
    return actions


# -----------------------------------------------------------------------
# 8. Main generation loop
# -----------------------------------------------------------------------
print(f"Generating {N_MONTHS} monthly snapshots, ~{CUSTOMERS_PER_MONTH} customers each ...\n")

all_records = []

for month_idx, ref_month in enumerate(REF_MONTHS):
    idx = np.random.choice(n_clients, size=CUSTOMERS_PER_MONTH, replace=False)
    sample = base.iloc[idx].reset_index(drop=True)

    evolved = evolve_features(sample[["client_id"] + FEATURE_COLS], month_idx)
    feat_df = evolved[FEATURE_COLS]

    values = compute_customer_value(feat_df)
    churn_probs = compute_base_churn(feat_df)
    ret_mults = compute_retention_mults(feat_df)
    actions = assign_actions(churn_probs)

    # Effective churn for assigned actions (vectorized)
    n = len(evolved)
    action_idx = np.array([ACTIONS.index(a) for a in actions])
    mult_matrix = np.column_stack([ret_mults[a] for a in ACTIONS])
    assigned_mult = mult_matrix[np.arange(n), action_idx]
    eff_churn = np.clip(
        churn_probs * assigned_mult + np.random.normal(0, 0.01, n),
        0.001, 0.95,
    )
    churned = (np.random.random(n) < eff_churn).astype(int)

    costs = np.array([ACTION_COSTS[a] for a in actions])
    rewards = (1 - churned) * values - costs

    # Ground-truth expected rewards for all actions
    er = {}
    for a in ACTIONS:
        p_eff = np.clip(churn_probs * ret_mults[a], 0.001, 0.95)
        er[a] = (1 - p_eff) * values - ACTION_COSTS[a]

    er_stack = np.column_stack([er[a] for a in ACTIONS])
    optimal_idx = np.argmax(er_stack, axis=1)

    # Scatter dates within the month
    days_in_month = (ref_month + pd.offsets.MonthEnd(0)).day
    random_days = np.random.randint(1, days_in_month + 1, n)
    ref_dates = pd.to_datetime([ref_month.replace(day=int(d)) for d in random_days])

    month_df = evolved.copy()
    month_df["reference_date"] = ref_dates
    month_df["customer_value_score"] = np.round(values, 6)
    month_df["churn_risk_score"] = np.round(churn_probs, 6)
    month_df["action"] = actions
    month_df["churned_90d"] = churned
    month_df["reward"] = np.round(rewards, 6)
    for a in ACTIONS:
        month_df[f"er_{a}"] = np.round(er[a], 6)
    month_df["optimal_action"] = [ACTIONS[i] for i in optimal_idx]

    all_records.append(month_df)

    print(
        f"  {ref_month.strftime('%Y-%m')}  |  "
        f"n={n:,}  churn={churned.mean():.3f}  "
        f"calls={(actions == 'call').sum():,}  "
        f"avg_reward=E{rewards.mean():,.0f}"
    )


# -----------------------------------------------------------------------
# 9. Combine & export
# -----------------------------------------------------------------------
dataset = pd.concat(all_records, ignore_index=True)

col_order = (
    ["client_id", "reference_date"]
    + FEATURE_COLS
    + [
        "customer_value_score", "churn_risk_score",
        "action", "churned_90d", "reward",
        "er_no_action", "er_push_notification", "er_email", "er_call",
        "optimal_action",
    ]
)
dataset = dataset[col_order]

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
dataset.to_csv(OUTPUT_PATH, index=False)

# -- Summary --
print(f"\n{'=' * 70}")
print(f"Dataset saved: {OUTPUT_PATH}")
print(f"  Rows:             {len(dataset):,}")
print(f"  Unique customers: {dataset['client_id'].nunique():,}")
print(f"  Date range:       {dataset['reference_date'].min()} -> {dataset['reference_date'].max()}")
print(f"  Churn rate:       {dataset['churned_90d'].mean():.4f}")
print(f"  Avg reward:       E{dataset['reward'].mean():,.2f}")

print(f"\n  Action distribution:")
print(dataset["action"].value_counts().to_string())

print(f"\n  Optimal action distribution:")
print(dataset["optimal_action"].value_counts().to_string())

opt_er = dataset.apply(lambda r: r[f"er_{r['optimal_action']}"], axis=1).mean()
no_er = dataset["er_no_action"].mean()
call_er = dataset["er_call"].mean()
email_er = dataset["er_email"].mean()
print(f"\n  E[R] optimal:      E{opt_er:,.2f}")
print(f"  E[R] no_action:    E{no_er:,.2f}")
print(f"  E[R] always call:  E{call_er:,.2f}")
print(f"  E[R] always email: E{email_er:,.2f}")
print(f"  Optimal improvement: +E{opt_er - no_er:,.2f} over no_action")

corr = dataset[["churn_risk_score", "customer_value_score"]].corr().iloc[0, 1]
print(f"\n  Risk-Value correlation: {corr:.4f}")

hv = dataset["customer_value_score"] > dataset["customer_value_score"].quantile(0.75)
hr = dataset["churn_risk_score"] > dataset["churn_risk_score"].quantile(0.75)
print(f"\n  Optimal action for HIGH risk + HIGH value:")
print(dataset.loc[hv & hr, "optimal_action"].value_counts(normalize=True).round(3).to_string())
lv = dataset["customer_value_score"] < dataset["customer_value_score"].quantile(0.25)
lr = dataset["churn_risk_score"] < dataset["churn_risk_score"].quantile(0.25)
print(f"\n  Optimal action for LOW risk + LOW value:")
print(dataset.loc[lv & lr, "optimal_action"].value_counts(normalize=True).round(3).to_string())
print(f"{'=' * 70}")
