"""
Generate a synthetic contextual-bandit training dataset for Model 2 (retention action model).

Uses the real enhanced churn features (20-feature dataset from notebook 04) as the base
distribution, then extends it across multiple monthly reference dates per customer.

Output schema matches Section 6 of the methodology report:
    (x_i, a_i, v_i, r_i) plus reference_date, churn_risk_score, and churned_90d.

Actions:  {no_action, push_notification, email, call}
Costs:    {0.00,      0.05,              0.10,  0.90}

Section 7 asymmetry:
    Pr(Y=0 | A=call, x)  >>  Pr(Y=0 | A ∈ {no_action, push, email}, x)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ENHANCED_PATH = DATA_DIR / "processed" / "churn_features_enhanced.csv"
OUTPUT_PATH = DATA_DIR / "output" / "bandit_training_dataset.csv"

# Action set and costs (Table 1 in the report)
ACTIONS = ["no_action", "push_notification", "email", "call"]
ACTION_COSTS = {"no_action": 0.00, "push_notification": 0.05, "email": 0.10, "call": 0.90}

# Cost weight λ  (reward = (1-Y)*v - λ*c(a))
LAMBDA = 1.0

# Time grid: 24 monthly reference dates
REF_DATES = pd.date_range("2024-01-01", periods=24, freq="MS")

# Fraction of the customer base sampled per month (not every customer
# enters the retention pipeline every month)
MONTHLY_SAMPLE_FRAC = 0.35

# 20 features used in the churn model (same order as notebook 04)
FEATURE_COLS = [
    "tenure_years", "n_products", "has_loan", "receives_salary",
    "avg_txn_per_month", "avg_txn_amount", "txn_trend",
    "avg_balance", "balance_trend", "n_contacts",
    "age", "credit_rating", "has_deposit", "digital_txn_ratio",
    "debit_credit_ratio", "balance_volatility", "txn_amount_std",
    "recency_days", "n_complaint_contacts", "avg_product_age_years",
]

# Features that stay constant or change deterministically over time
STATIC_FEATURES = {"has_loan", "receives_salary", "has_deposit", "credit_rating"}
BINARY_FEATURES = {"has_loan", "receives_salary", "has_deposit"}

# ──────────────────────────────────────────────────────────────────────
# 1. Load real data & compute per-feature statistics
# ──────────────────────────────────────────────────────────────────────
print("Loading enhanced dataset …")
base = pd.read_csv(ENHANCED_PATH)
assert set(FEATURE_COLS).issubset(base.columns), "Missing expected feature columns"

# Per-feature median & IQR (for bounded noise)
feat_stats = {}
for col in FEATURE_COLS:
    s = base[col]
    feat_stats[col] = {
        "mean": s.mean(),
        "std": s.std(),
        "median": s.median(),
        "q25": s.quantile(0.25),
        "q75": s.quantile(0.75),
        "min": s.min(),
        "max": s.max(),
    }

client_ids = base["client_id"].values
n_clients = len(client_ids)
print(f"  {n_clients:,} customers, {len(FEATURE_COLS)} features")

# ──────────────────────────────────────────────────────────────────────
# 2. Helper: compute customer value score  (normalised 0–1)
# ──────────────────────────────────────────────────────────────────────
def compute_customer_value(df: pd.DataFrame) -> np.ndarray:
    """
    Heuristic customer value based on real banking signals.
    Higher tenure, more products, salary reception, higher balances,
    more transaction volume, and lower credit risk all increase value.
    """
    # Normalise each component to roughly 0–1 using the base distribution
    tenure_norm = np.clip(df["tenure_years"].values / 30.0, 0, 1)
    prod_norm = np.clip(df["n_products"].values / 8.0, 0, 1)
    salary = df["receives_salary"].values.astype(float)
    bal_norm = np.clip(df["avg_balance"].values / 10_000.0, 0, 1)
    txn_norm = np.clip(df["avg_txn_per_month"].values / 30.0, 0, 1)
    # Credit rating: lower is better (1=best), invert
    cr_norm = np.clip(1.0 - (df["credit_rating"].values - 1) / 10.0, 0, 1)
    deposit = df["has_deposit"].values.astype(float)
    loan = df["has_loan"].values.astype(float)

    raw = (
        0.20 * tenure_norm
        + 0.15 * prod_norm
        + 0.15 * salary
        + 0.15 * bal_norm
        + 0.10 * txn_norm
        + 0.10 * cr_norm
        + 0.10 * deposit
        + 0.05 * loan
    )
    # Scale to [1, 15] — abstract value units comparable to action costs.
    # This ensures calls (cost 0.90) can be justified for high-value
    # customers where retention uplift × value > cost.
    v_min, v_max = raw.min(), raw.max()
    if v_max - v_min < 1e-9:
        return np.full(len(raw), 8.0)
    scaled = 1.0 + 14.0 * (raw - v_min) / (v_max - v_min)
    return scaled


# ──────────────────────────────────────────────────────────────────────
# 3. Helper: base churn probability from features
# ──────────────────────────────────────────────────────────────────────
def compute_base_churn_prob(df: pd.DataFrame) -> np.ndarray:
    """
    Logistic-style churn probability driven by realistic risk factors.

    Calibrated with a wider spread: most customers 3–10%, but a
    meaningful high-risk tail at 20–40%.  This creates the economic
    region where calls (cost 0.90) become justified for high-value
    customers (Section 7 proof-of-concept).
    """
    z = np.zeros(len(df))

    # Baseline intercept (raised to ~-1.8 so the mean is ~15%)
    z += -1.8

    # ── risk-increasing factors (wider coefficients for spread) ──

    # Tenure: short tenure strongly increases risk
    z += -0.05 * np.clip(df["tenure_years"].values, 0, 50)

    # Products: fewer products → higher risk
    z += -0.25 * np.clip(df["n_products"].values, 1, 10)

    # Salary: receiving salary is very protective
    z += -0.8 * df["receives_salary"].values

    # Transaction activity: lower → higher risk
    z += -0.05 * np.clip(df["avg_txn_per_month"].values, 0, 50)

    # Transaction trend: declining → higher risk
    z += -0.5 * np.clip(df["txn_trend"].values, -5, 5)

    # Balance: lower → higher risk
    z += -0.00008 * np.clip(df["avg_balance"].values, 0, 50000)

    # Balance trend: declining → higher risk
    z += -0.0002 * np.clip(df["balance_trend"].values, -5000, 5000)

    # Contacts: more contacts → higher risk
    z += 0.35 * np.clip(df["n_contacts"].values, 0, 10)

    # Credit rating: higher number → higher risk
    z += 0.12 * np.clip(df["credit_rating"].values, 1, 20)

    # Recency: longer gap → higher risk
    z += 0.006 * np.clip(df["recency_days"].values, 0, 275)

    # Digital engagement: higher → protective
    z += -1.5 * np.clip(df["digital_txn_ratio"].values, 0, 1)

    # Complaints: strong risk signal
    z += 0.7 * np.clip(df["n_complaint_contacts"].values, 0, 5)

    # Deposits/loans protective
    z += -0.5 * df["has_deposit"].values
    z += -0.5 * df["has_loan"].values

    # Age: U-shape (young & old higher risk)
    age_centered = (df["age"].values - 50) / 20.0
    z += 0.15 * age_centered ** 2

    prob = 1.0 / (1.0 + np.exp(-z))
    return prob


# ──────────────────────────────────────────────────────────────────────
# 4. Helper: action effect on churn (Section 7 asymmetry)
# ──────────────────────────────────────────────────────────────────────
def retention_multiplier(action: str, churn_prob: float, value: float) -> float:
    """
    Return a multiplier ∈ (0, 1] applied to churn probability.
    Lower multiplier = stronger retention effect.

    Section 7: call has high retention effect; push/email low.
    Effect also depends on customer profile — high-value customers
    respond better to interventions.
    """
    if action == "no_action":
        return 1.0

    # Base retention effect per action
    base_effect = {
        "push_notification": 0.92,  # 8% churn reduction
        "email": 0.82,              # 18% churn reduction
        "call": 0.35,               # 65% churn reduction (strong)
    }
    mult = base_effect[action]

    # Interaction: high-value customers respond more to call
    if action == "call":
        # Normalise value to [0,1] since it's now on [1,15] scale
        v_norm = np.clip((value - 1) / 14.0, 0, 1)
        value_boost = np.clip(0.12 * v_norm, 0, 0.12)
        mult -= value_boost

    # For high-risk customers, interventions are more impactful
    risk_boost = np.clip(0.08 * (churn_prob - 0.05), 0, 0.10)
    mult -= risk_boost

    return max(mult, 0.10)  # floor: never fully eliminate churn


# ──────────────────────────────────────────────────────────────────────
# 5. Evolve features over time
# ──────────────────────────────────────────────────────────────────────
def evolve_features(base_df: pd.DataFrame, month_offset: int) -> pd.DataFrame:
    """
    Create a time-shifted version of the feature set.
    Deterministic changes (tenure, product age, age) + bounded noise
    on behavioral features.
    """
    df = base_df.copy()

    # Deterministic time shift
    df["tenure_years"] = df["tenure_years"] + month_offset / 12.0
    df["avg_product_age_years"] = df["avg_product_age_years"] + month_offset / 12.0
    df["age"] = df["age"] + month_offset / 12.0

    n = len(df)

    # Add bounded Gaussian noise to behavioral features
    behavioral = [
        "avg_txn_per_month", "avg_txn_amount", "txn_trend",
        "avg_balance", "balance_trend", "n_contacts",
        "digital_txn_ratio", "debit_credit_ratio",
        "balance_volatility", "txn_amount_std",
        "recency_days", "n_complaint_contacts",
    ]
    for col in behavioral:
        st = feat_stats[col]
        iqr = st["q75"] - st["q25"]
        if iqr < 1e-9:
            iqr = max(st["std"] * 0.5, 1e-6)
        # Noise scale = 15% of IQR
        noise = np.random.normal(0, 0.15 * iqr, size=n)

        # Add a small time-varying drift (mean-reverting)
        drift = 0.01 * month_offset * np.random.normal(0, 0.05 * iqr, size=n)
        df[col] = df[col] + noise + drift

    # Clip to reasonable ranges
    for col in FEATURE_COLS:
        st = feat_stats[col]
        lower = st["min"]
        upper = st["max"]
        # Extend range slightly to allow natural spread
        spread = upper - lower
        df[col] = np.clip(df[col], lower - 0.05 * spread, upper + 0.05 * spread)

    # Re-enforce binary constraints
    for col in BINARY_FEATURES:
        df[col] = (df[col] >= 0.5).astype(int)

    # Integer-like features
    df["n_products"] = np.clip(np.round(df["n_products"]), 1, None).astype(int)
    df["n_contacts"] = np.clip(np.round(df["n_contacts"]), 0, None).astype(int)
    df["n_complaint_contacts"] = np.clip(np.round(df["n_complaint_contacts"]), 0, None).astype(int)
    df["recency_days"] = np.clip(np.round(df["recency_days"]), 0, None).astype(int)
    df["credit_rating"] = np.clip(np.round(df["credit_rating"]), 1, None).astype(int)

    # Ensure non-negative where appropriate
    for col in ["avg_txn_per_month", "avg_txn_amount", "balance_volatility",
                "txn_amount_std", "digital_txn_ratio", "debit_credit_ratio"]:
        df[col] = np.clip(df[col], 0, None)

    df["digital_txn_ratio"] = np.clip(df["digital_txn_ratio"], 0, 1)

    return df


# ──────────────────────────────────────────────────────────────────────
# 6. Action assignment policy (evolves over time to mimic logged data)
# ──────────────────────────────────────────────────────────────────────
def assign_actions(churn_probs: np.ndarray, values: np.ndarray,
                   month_idx: int, n_months: int) -> np.ndarray:
    """
    Simulate a realistic logged action policy that evolves over time.

    Early months: mostly uniform random (exploration phase).
    Later months: increasingly informed — higher-risk, higher-value
    customers get calls; lower-risk get lighter or no action.

    This ensures the dataset reflects a time-based learning process.
    """
    n = len(churn_probs)
    actions = np.empty(n, dtype=object)

    # Fraction of "informed" decisions grows linearly from 0.1 to 0.7
    informed_frac = 0.1 + 0.6 * (month_idx / max(n_months - 1, 1))

    for i in range(n):
        if np.random.random() > informed_frac:
            # Random exploration
            actions[i] = np.random.choice(ACTIONS)
        else:
            # Informed policy: use risk & value
            risk = churn_probs[i]
            val = values[i]

            if risk < 0.05:
                # Very low risk → no action or light touch
                actions[i] = np.random.choice(
                    ["no_action", "push_notification"], p=[0.7, 0.3]
                )
            elif risk < 0.15:
                if val > 0.5:
                    # Moderate risk + high value → email or push
                    actions[i] = np.random.choice(
                        ["push_notification", "email", "call"], p=[0.3, 0.5, 0.2]
                    )
                else:
                    actions[i] = np.random.choice(
                        ["no_action", "push_notification", "email"], p=[0.3, 0.4, 0.3]
                    )
            else:
                if val > 0.5:
                    # High risk + high value → call preferred
                    actions[i] = np.random.choice(
                        ["email", "call"], p=[0.25, 0.75]
                    )
                else:
                    # High risk + low value → email or push (call too expensive)
                    actions[i] = np.random.choice(
                        ["push_notification", "email", "call"], p=[0.3, 0.5, 0.2]
                    )
    return actions


# ──────────────────────────────────────────────────────────────────────
# 7. Main generation loop
# ──────────────────────────────────────────────────────────────────────
print(f"Generating {len(REF_DATES)} monthly snapshots …")

all_records = []
n_months = len(REF_DATES)

for month_idx, ref_date in enumerate(REF_DATES):
    # Sample a subset of customers for this month
    sample_size = int(n_clients * MONTHLY_SAMPLE_FRAC)
    sampled_idx = np.random.choice(n_clients, size=sample_size, replace=False)
    base_sample = base.iloc[sampled_idx].reset_index(drop=True)

    # Evolve features
    evolved = evolve_features(base_sample[["client_id"] + FEATURE_COLS], month_idx)

    # Build a temporary DataFrame for scoring
    score_df = evolved[FEATURE_COLS].copy()

    # Compute customer value & churn risk
    values = compute_customer_value(score_df)
    churn_probs = compute_base_churn_prob(score_df)

    # Assign actions (logged policy)
    actions = assign_actions(churn_probs, values, month_idx, n_months)

    # Simulate churn outcome per customer
    churned_90d = np.zeros(len(evolved), dtype=int)
    for i in range(len(evolved)):
        # Adjust churn prob based on action retention effect
        mult = retention_multiplier(actions[i], churn_probs[i], values[i])
        effective_prob = churn_probs[i] * mult

        # Add slight randomness (individual variation)
        noise = np.random.normal(0, 0.02)
        effective_prob = np.clip(effective_prob + noise, 0.001, 0.95)

        churned_90d[i] = int(np.random.random() < effective_prob)

    # Compute reward: R = (1 - Y) * v - λ * c(a)
    costs = np.array([ACTION_COSTS[a] for a in actions])
    rewards = (1 - churned_90d) * values - LAMBDA * costs

    # Build record
    month_df = evolved.copy()
    month_df["reference_date"] = ref_date
    month_df["customer_value_score"] = np.round(values, 6)
    month_df["churn_risk_score"] = np.round(churn_probs, 6)
    month_df["action"] = actions
    month_df["churned_90d"] = churned_90d
    month_df["reward"] = np.round(rewards, 6)

    all_records.append(month_df)

    # Progress
    churn_rate = churned_90d.mean()
    n_calls = (actions == "call").sum()
    avg_reward = rewards.mean()
    print(
        f"  {ref_date.strftime('%Y-%m')}  |  "
        f"n={len(evolved):,}  "
        f"churn={churn_rate:.3f}  "
        f"calls={n_calls:,}  "
        f"avg_reward={avg_reward:.4f}"
    )

# ──────────────────────────────────────────────────────────────────────
# 8. Combine & export
# ──────────────────────────────────────────────────────────────────────
dataset = pd.concat(all_records, ignore_index=True)

# Reorder columns: client_id, reference_date, features, value, risk, action, outcome, reward
col_order = (
    ["client_id", "reference_date"]
    + FEATURE_COLS
    + ["customer_value_score", "churn_risk_score", "action", "churned_90d", "reward"]
)
dataset = dataset[col_order]

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
dataset.to_csv(OUTPUT_PATH, index=False)

print(f"\n{'='*70}")
print(f"Dataset saved: {OUTPUT_PATH}")
print(f"  Rows:            {len(dataset):,}")
print(f"  Unique customers: {dataset['client_id'].nunique():,}")
print(f"  Date range:       {dataset['reference_date'].min()} → {dataset['reference_date'].max()}")
print(f"  Columns:          {len(dataset.columns)}")
print(f"\n  Churn rate:       {dataset['churned_90d'].mean():.4f}")
print(f"  Avg reward:       {dataset['reward'].mean():.4f}")
print(f"\n  Action distribution:")
print(dataset["action"].value_counts().to_string())
print(f"\n  Avg reward by action:")
print(dataset.groupby("action")["reward"].mean().round(4).to_string())
print(f"\n  Churn rate by action:")
print(dataset.groupby("action")["churned_90d"].mean().round(4).to_string())
print(f"\n  Rows per customer (median): {dataset.groupby('client_id').size().median():.0f}")
print(f"{'='*70}")
