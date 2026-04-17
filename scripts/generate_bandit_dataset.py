"""
Generate a synthetic contextual-bandit training dataset for Model 2 (retention action model).

Uses the real enhanced churn features (20-feature dataset from notebook 04) as the base
distribution, then extends it across 24 monthly reference dates per customer.

Key design principles:
  1. Customer VALUE and churn RISK are independent — high-value at-risk customers exist.
  2. Action effects are calibrated so the optimal policy BEATS every naive baseline.
  3. Calls are expensive but very effective → optimal only for high risk × high value.
  4. Phase 1 (months 1–12): logging policy uses only {no_action, push, email}.
     Phase 2 (months 13–24): logging policy begins to include calls.
  5. Ground-truth expected rewards for all actions are saved for exact evaluation.

Actions:  {no_action, push_notification, email, call}
Costs:    {  0 EUR,      20 EUR,         100 EUR,  1200 EUR}

Retention multipliers (applied to base churn probability):
    no_action ........... 1.00  (0% reduction)
    push_notification ... 0.95  (5% reduction — minor nudge)
    email ............... 0.85  (15% reduction — moderate campaign)
    call ................ 0.30  (70% reduction — personal outreach with incentives)
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

# Action set and costs (in EUR)
ACTIONS = ["no_action", "push_notification", "email", "call"]
ACTION_COSTS = {"no_action": 0.0, "push_notification": 20.0, "email": 100.0, "call": 1200.0}

# Retention multipliers — lower = stronger retention effect
RETENTION_MULT = {"no_action": 1.00, "push_notification": 0.95, "email": 0.85, "call": 0.30}

# Cost weight λ  (reward = (1−Y)·v − λ·c(a))
LAMBDA = 1.0

# Time grid: 24 monthly reference dates
REF_DATES = pd.date_range("2024-01-01", periods=24, freq="MS")

# Fraction of the customer base sampled per month
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

STATIC_FEATURES = {"has_loan", "receives_salary", "has_deposit", "credit_rating"}
BINARY_FEATURES = {"has_loan", "receives_salary", "has_deposit"}

# ──────────────────────────────────────────────────────────────────────
# 1. Load real data & compute per-feature statistics
# ──────────────────────────────────────────────────────────────────────
print("Loading enhanced dataset …")
base = pd.read_csv(ENHANCED_PATH)
assert set(FEATURE_COLS).issubset(base.columns), "Missing expected feature columns"

feat_stats = {}
for col in FEATURE_COLS:
    s = base[col]
    feat_stats[col] = {
        "mean": s.mean(), "std": s.std(), "median": s.median(),
        "q25": s.quantile(0.25), "q75": s.quantile(0.75),
        "min": s.min(), "max": s.max(),
    }

client_ids = base["client_id"].values
n_clients = len(client_ids)
print(f"  {n_clients:,} customers, {len(FEATURE_COLS)} features")


# ──────────────────────────────────────────────────────────────────────
# 2. Customer VALUE — independent of churn-risk features
# ──────────────────────────────────────────────────────────────────────
def compute_customer_value(df: pd.DataFrame) -> np.ndarray:
    """
    Customer annual revenue in EUR.
    
    60% feature-based (monetary signals) + 40% random noise.
    The random component ensures risk-value independence while keeping
    feature-driven ordering strong enough for clean decile behavior.
    
    Range: roughly €500 – €15,000 annual revenue.
    """
    bal_norm = np.clip(df["avg_balance"].values / 10_000.0, 0, 1)
    txn_amt_norm = np.clip(df["avg_txn_amount"].values / 500.0, 0, 1)
    prod_norm = np.clip(df["n_products"].values / 8.0, 0, 1)
    deposit = df["has_deposit"].values.astype(float)
    loan = df["has_loan"].values.astype(float)

    feature_component = (
        0.35 * bal_norm
        + 0.30 * txn_amt_norm
        + 0.15 * prod_norm
        + 0.10 * deposit
        + 0.10 * loan
    )

    # Mix: 60% features + 40% random → clean decile ordering with independence
    random_component = np.random.uniform(0, 1, size=len(df))
    raw = 0.60 * feature_component + 0.40 * random_component

    # Scale to EUR [500, 15000] annual revenue
    v_min, v_max = raw.min(), raw.max()
    if v_max - v_min < 1e-9:
        return np.full(len(raw), 5000.0)
    scaled = 500.0 + 14500.0 * (raw - v_min) / (v_max - v_min)
    return scaled


# ──────────────────────────────────────────────────────────────────────
# 3. Base churn probability — driven by ENGAGEMENT/BEHAVIORAL signals
# ──────────────────────────────────────────────────────────────────────
def compute_base_churn_prob(df: pd.DataFrame) -> np.ndarray:
    """
    Churn probability driven by engagement & behavioral signals that are
    largely orthogonal to the monetary value signals.

    Key risk factors: complaints, contacts, low activity, declining trends,
    short tenure, low digital engagement, recent inactivity.
    """
    z = np.zeros(len(df))

    # Baseline intercept → calibrate mean churn ~15%
    z += -1.6

    # ── Risk-increasing factors (behavioral/engagement) ──
    z += 0.90 * np.clip(df["n_complaint_contacts"].values, 0, 5)
    z += 0.40 * np.clip(df["n_contacts"].values, 0, 10)
    z += 0.008 * np.clip(df["recency_days"].values, 0, 275)
    z += -0.60 * np.clip(df["txn_trend"].values, -5, 5)
    z += -0.0003 * np.clip(df["balance_trend"].values, -5000, 5000)
    z += -0.04 * np.clip(df["avg_txn_per_month"].values, 0, 50)
    z += -1.2 * np.clip(df["digital_txn_ratio"].values, 0, 1)

    # ── Mild protective factors ──
    z += -0.03 * np.clip(df["tenure_years"].values, 0, 50)
    z += 0.08 * np.clip(df["credit_rating"].values, 1, 20)
    age_centered = (df["age"].values - 50) / 20.0
    z += 0.10 * age_centered ** 2

    # NOTE: avg_balance, n_products, has_deposit, has_loan, receives_salary
    # are NOT included here — they drive VALUE, not risk.

    prob = 1.0 / (1.0 + np.exp(-z))
    return prob


# ──────────────────────────────────────────────────────────────────────
# 4. Action effect on churn — clean, deterministic multipliers
# ──────────────────────────────────────────────────────────────────────
def effective_churn_prob(action: str, base_prob: float) -> float:
    return base_prob * RETENTION_MULT[action]


def expected_reward(action: str, base_prob: float, value: float) -> float:
    p_churn = effective_churn_prob(action, base_prob)
    return (1.0 - p_churn) * value - LAMBDA * ACTION_COSTS[action]


# ──────────────────────────────────────────────────────────────────────
# 5. Evolve features over time
# ──────────────────────────────────────────────────────────────────────
def evolve_features(base_df: pd.DataFrame, month_offset: int) -> pd.DataFrame:
    df = base_df.copy()
    df["tenure_years"] = df["tenure_years"] + month_offset / 12.0
    df["avg_product_age_years"] = df["avg_product_age_years"] + month_offset / 12.0
    df["age"] = df["age"] + month_offset / 12.0

    n = len(df)
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
        noise = np.random.normal(0, 0.15 * iqr, size=n)
        drift = 0.01 * month_offset * np.random.normal(0, 0.05 * iqr, size=n)
        df[col] = df[col] + noise + drift

    for col in FEATURE_COLS:
        st = feat_stats[col]
        lower, upper = st["min"], st["max"]
        spread = upper - lower
        df[col] = np.clip(df[col], lower - 0.05 * spread, upper + 0.05 * spread)

    for col in BINARY_FEATURES:
        df[col] = (df[col] >= 0.5).astype(int)

    df["n_products"] = np.clip(np.round(df["n_products"]), 1, None).astype(int)
    df["n_contacts"] = np.clip(np.round(df["n_contacts"]), 0, None).astype(int)
    df["n_complaint_contacts"] = np.clip(np.round(df["n_complaint_contacts"]), 0, None).astype(int)
    df["recency_days"] = np.clip(np.round(df["recency_days"]), 0, None).astype(int)
    df["credit_rating"] = np.clip(np.round(df["credit_rating"]), 1, None).astype(int)

    for col in ["avg_txn_per_month", "avg_txn_amount", "balance_volatility",
                "txn_amount_std", "digital_txn_ratio", "debit_credit_ratio"]:
        df[col] = np.clip(df[col], 0, None)
    df["digital_txn_ratio"] = np.clip(df["digital_txn_ratio"], 0, 1)

    return df


# ──────────────────────────────────────────────────────────────────────
# 6. Logging policy — TWO PHASES
# ──────────────────────────────────────────────────────────────────────
def assign_actions(churn_probs: np.ndarray, values: np.ndarray,
                   month_idx: int) -> np.ndarray:
    """
    Logging policy: all 4 actions available from month 1.
    Action probabilities depend on risk level (reasonable heuristic).
    Calls are assigned ~15% of the time overall — enough to learn from.
    """
    n = len(churn_probs)
    actions = np.empty(n, dtype=object)

    for i in range(n):
        risk = churn_probs[i]

        if risk < 0.10:
            # Low risk: mostly no_action, some push
            actions[i] = np.random.choice(
                ACTIONS, p=[0.50, 0.25, 0.15, 0.10])
        elif risk < 0.25:
            # Medium risk: mix of push/email, some calls
            actions[i] = np.random.choice(
                ACTIONS, p=[0.20, 0.30, 0.30, 0.20])
        else:
            # High risk: mostly email/call
            actions[i] = np.random.choice(
                ACTIONS, p=[0.10, 0.15, 0.45, 0.30])

    return actions


# ──────────────────────────────────────────────────────────────────────
# 7. Main generation loop
# ──────────────────────────────────────────────────────────────────────
print(f"Generating {len(REF_DATES)} monthly snapshots …")

all_records = []

for month_idx, ref_date in enumerate(REF_DATES):
    sample_size = int(n_clients * MONTHLY_SAMPLE_FRAC)
    sampled_idx = np.random.choice(n_clients, size=sample_size, replace=False)
    base_sample = base.iloc[sampled_idx].reset_index(drop=True)

    evolved = evolve_features(base_sample[["client_id"] + FEATURE_COLS], month_idx)
    score_df = evolved[FEATURE_COLS].copy()

    values = compute_customer_value(score_df)
    churn_probs = compute_base_churn_prob(score_df)

    actions = assign_actions(churn_probs, values, month_idx)

    # Simulate churn outcome
    churned_90d = np.zeros(len(evolved), dtype=int)
    for i in range(len(evolved)):
        p_eff = effective_churn_prob(actions[i], churn_probs[i])
        noise = np.random.normal(0, 0.01)
        p_eff = np.clip(p_eff + noise, 0.001, 0.95)
        churned_90d[i] = int(np.random.random() < p_eff)

    costs = np.array([ACTION_COSTS[a] for a in actions])
    rewards = (1 - churned_90d) * values - LAMBDA * costs

    # Ground truth expected rewards for all actions
    er_no = np.array([expected_reward("no_action", churn_probs[i], values[i])
                      for i in range(len(evolved))])
    er_push = np.array([expected_reward("push_notification", churn_probs[i], values[i])
                        for i in range(len(evolved))])
    er_email = np.array([expected_reward("email", churn_probs[i], values[i])
                         for i in range(len(evolved))])
    er_call = np.array([expected_reward("call", churn_probs[i], values[i])
                        for i in range(len(evolved))])

    month_df = evolved.copy()
    month_df["reference_date"] = ref_date
    month_df["customer_value_score"] = np.round(values, 6)
    month_df["churn_risk_score"] = np.round(churn_probs, 6)
    month_df["action"] = actions
    month_df["churned_90d"] = churned_90d
    month_df["reward"] = np.round(rewards, 6)
    month_df["er_no_action"] = np.round(er_no, 6)
    month_df["er_push_notification"] = np.round(er_push, 6)
    month_df["er_email"] = np.round(er_email, 6)
    month_df["er_call"] = np.round(er_call, 6)
    month_df["optimal_action"] = [
        ACTIONS[np.argmax([er_no[i], er_push[i], er_email[i], er_call[i]])]
        for i in range(len(evolved))
    ]

    all_records.append(month_df)

    churn_rate = churned_90d.mean()
    n_calls = (actions == "call").sum()
    avg_reward = rewards.mean()
    print(
        f"  {ref_date.strftime('%Y-%m')}  |  "
        f"n={len(evolved):,}  "
        f"churn={churn_rate:.3f}  "
        f"calls={n_calls:,}  "
        f"avg_reward=€{avg_reward:,.0f}")


# ──────────────────────────────────────────────────────────────────────
# 8. Combine & export
# ──────────────────────────────────────────────────────────────────────
dataset = pd.concat(all_records, ignore_index=True)

col_order = (
    ["client_id", "reference_date"]
    + FEATURE_COLS
    + ["customer_value_score", "churn_risk_score",
       "action", "churned_90d", "reward",
       "er_no_action", "er_push_notification", "er_email", "er_call",
       "optimal_action"]
)
dataset = dataset[col_order]

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
dataset.to_csv(OUTPUT_PATH, index=False)

# ── Summary ──
print(f"\n{'='*70}")
print(f"Dataset saved: {OUTPUT_PATH}")
print(f"  Rows:             {len(dataset):,}")
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

print(f"\n  Optimal action distribution (ground truth):")
print(dataset["optimal_action"].value_counts().to_string())

# Verify economics: optimal beats baselines
opt_er = dataset.apply(lambda r: r[f"er_{r['optimal_action']}"], axis=1).mean()
no_er = dataset["er_no_action"].mean()
call_er = dataset["er_call"].mean()
print(f"\n  Expected reward verification:")
print(f"    Optimal policy:      {opt_er:.4f}")
print(f"    Always no_action:    {no_er:.4f}")
print(f"    Always call:         {call_er:.4f}")
print(f"    Optimal improvement: +{opt_er - no_er:.4f} over no_action ({(opt_er-no_er)/no_er*100:.1f}%)")

# Verify risk-value independence
corr = dataset[["churn_risk_score", "customer_value_score"]].corr().iloc[0, 1]
print(f"\n  Risk-Value correlation:  {corr:.4f}  (should be near 0)")

# Verify calls are optimal for high-risk high-value
hv = dataset["customer_value_score"] > dataset["customer_value_score"].quantile(0.75)
hr = dataset["churn_risk_score"] > dataset["churn_risk_score"].quantile(0.75)
print(f"\n  Optimal action for HIGH risk + HIGH value customers:")
print(dataset.loc[hv & hr, "optimal_action"].value_counts(normalize=True).round(3).to_string())
print(f"\n  Optimal action for LOW risk + LOW value customers:")
lv = dataset["customer_value_score"] < dataset["customer_value_score"].quantile(0.25)
lr = dataset["churn_risk_score"] < dataset["churn_risk_score"].quantile(0.25)
print(dataset.loc[lv & lr, "optimal_action"].value_counts(normalize=True).round(3).to_string())
print(f"{'='*70}")