"""
data_prep.py — Load raw tables and build the churn feature sets.

Runs the same feature engineering as notebooks 01 + 02 + 04, in one script,
without plotting or exploration. Produces three CSVs in data/processed/:

    churn_features_raw.csv        10 features, missing values untouched
    churn_features_clean.csv      10 features, imputed
    churn_features_enhanced.csv   20 features (10 original + 10 added), imputed

Usage:
    python scripts/data_prep.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

CORE_DOMAINS = {"ACCOUNTS", "LOANS", "DEPOSITS"}
PLACEHOLDER_DATE = "1/1/2023/"

FEAT_START = pd.Timestamp("2025-04-01")
FEAT_END = pd.Timestamp("2025-12-31")
REF = pd.Timestamp("2026-01-01")
CHURN_END = pd.Timestamp("2026-03-31")

BASELINE_FEATURES = [
    "tenure_years", "n_products", "has_loan", "receives_salary",
    "avg_txn_per_month", "avg_txn_amount", "txn_trend",
    "avg_balance", "balance_trend", "n_contacts",
]
ADDED_FEATURES = [
    "age", "credit_rating", "has_deposit", "digital_txn_ratio",
    "debit_credit_ratio", "balance_volatility", "txn_amount_std",
    "recency_days", "n_complaint_contacts", "avg_product_age_years",
]
ALL_FEATURES = BASELINE_FEATURES + ADDED_FEATURES

DIGITAL_CHANNELS = {"Retail internet banking", "SEPA Instant"}
COMPLAINT_TYPES = {"Prigovor", "Naplata"}


# ---------------------------------------------------------------------------
# EU-format parsing helpers (CSVs use dd/mm/yyyy and comma decimals)
# ---------------------------------------------------------------------------
def parse_eu_date(s):
    return pd.to_datetime(s.str.strip().str.rstrip("/"), format="%d/%m/%Y", errors="coerce")


def parse_eu_datetime(s):
    return pd.to_datetime(
        s.str.strip().str.replace("/ ", " ", regex=False),
        format="%d/%m/%Y %H:%M:%S", errors="coerce",
    )


def eu_to_float(s):
    return pd.to_numeric(
        s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
        errors="coerce",
    )


# ---------------------------------------------------------------------------
# Load raw tables
# ---------------------------------------------------------------------------
def load_raw():
    clients = pd.read_csv(RAW / "clients.csv", dtype=str)
    clients["age"] = pd.to_numeric(clients["age"], errors="coerce")
    clients["first_relationship_date"] = parse_eu_date(clients["first_relationship_date"])

    products = pd.read_csv(RAW / "products.csv", dtype=str)
    products.loc[products["closing_date"] == PLACEHOLDER_DATE, "closing_date"] = np.nan
    products["opening_date"] = parse_eu_date(products["opening_date"])
    products["closing_date"] = parse_eu_date(products["closing_date"])

    transactions = pd.read_csv(RAW / "transactions.csv", dtype=str)
    transactions = transactions.loc[:, transactions.columns.notna()]
    transactions["date"] = parse_eu_datetime(transactions["txn_datetime"])
    transactions["amount"] = eu_to_float(transactions["amount"])

    balances = pd.read_csv(RAW / "balances.csv", dtype=str)
    balances["balance"] = eu_to_float(balances["balance"])
    balances["valid_from"] = parse_eu_date(balances["valid_from"])

    contacts = pd.read_csv(RAW / "contacts.csv", dtype=str)
    contacts["date"] = parse_eu_datetime(contacts["created_at"])

    return clients, products, transactions, balances, contacts


# ---------------------------------------------------------------------------
# Population and churn label
# ---------------------------------------------------------------------------
def build_labels(products):
    """At-risk = any active product at REF. Churned = had core at REF, lost all core by CHURN_END."""
    products_core = products[products["product_domain"].isin(CORE_DOMAINS)]

    prods_at_ref = products[
        (products["opening_date"] <= REF)
        & (products["closing_date"].isna() | (products["closing_date"] > REF))
    ]
    risk_ids = set(prods_at_ref["client_id"].unique())

    core_at_ref = products_core[
        (products_core["opening_date"] <= REF)
        & (products_core["closing_date"].isna() | (products_core["closing_date"] > REF))
    ]
    had_core = set(core_at_ref["client_id"].unique())

    core_at_end = products_core[
        (products_core["opening_date"] <= CHURN_END)
        & (products_core["closing_date"].isna() | (products_core["closing_date"] > CHURN_END))
    ]
    core_active_end = set(core_at_end["client_id"].unique())
    churned = (risk_ids & had_core) - core_active_end

    feat = pd.DataFrame({"client_id": list(risk_ids)})
    feat["churned"] = feat["client_id"].isin(churned).astype(int)
    return feat, prods_at_ref


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def _slope(grp):
    """Least-squares slope of grp['mo'] vs grp['y']."""
    if len(grp) < 2:
        return 0.0
    x = grp["mo"].values.astype(float)
    y = grp["y"].values.astype(float)
    x = x - x.mean()
    d = (x ** 2).sum()
    return (x * (y - y.mean())).sum() / d if d else 0.0


def build_baseline_features(feat, clients, products, prods_at_ref, transactions, balances, contacts):
    risk_ids = set(feat["client_id"])
    prod_client = products[["product_id", "client_id"]].drop_duplicates()

    # 1. tenure_years
    kl = clients[["client_id", "first_relationship_date"]].copy()
    kl["tenure_years"] = (REF - kl["first_relationship_date"]).dt.days / 365.25
    feat = feat.merge(kl[["client_id", "tenure_years"]], on="client_id", how="left")

    # 2. n_products
    n_pr = prods_at_ref.groupby("client_id").size().reset_index(name="n_products")
    feat = feat.merge(n_pr, on="client_id", how="left")

    # 3. has_loan
    loan_ids = set(prods_at_ref[prods_at_ref["product_domain"] == "LOANS"]["client_id"])
    feat["has_loan"] = feat["client_id"].isin(loan_ids).astype(int)

    # 4. receives_salary
    feat = feat.merge(clients[["client_id", "receives_salary_at_bank"]], on="client_id", how="left")
    feat["receives_salary"] = (feat["receives_salary_at_bank"] == "YES").astype(int)
    feat.drop(columns=["receives_salary_at_bank"], inplace=True)

    # Transaction window scoped to at-risk population
    txn_w = transactions[(transactions["date"] >= FEAT_START) & (transactions["date"] <= FEAT_END)]
    txn_c = txn_w.merge(prod_client, on="product_id", how="left")
    txn_c = txn_c[txn_c["client_id"].isin(risk_ids)]

    # 5-6. avg_txn_per_month, avg_txn_amount
    txn_agg = txn_c.groupby("client_id").agg(
        n_txn=("amount", "size"),
        avg_txn_amount=("amount", "mean"),
        n_months=("date", lambda x: x.dt.to_period("M").nunique()),
    ).reset_index()
    txn_agg["avg_txn_per_month"] = txn_agg["n_txn"] / txn_agg["n_months"].clip(lower=1)
    feat = feat.merge(
        txn_agg[["client_id", "avg_txn_per_month", "avg_txn_amount"]],
        on="client_id", how="left",
    )

    # 7. txn_trend (slope of monthly counts)
    txn_mc = (
        txn_c.groupby(["client_id", txn_c["date"].dt.to_period("M")])
        .size().reset_index(name="y")
    )
    txn_mc["mo"] = txn_mc["date"].apply(lambda p: p.year * 12 + p.month)
    feat = feat.merge(
        txn_mc.groupby("client_id").apply(_slope).reset_index(name="txn_trend"),
        on="client_id", how="left",
    )

    # 8. avg_balance (ACCOUNT_BALANCE only)
    bal_w = balances[
        (balances["balance_type"] == "ACCOUNT_BALANCE")
        & (balances["valid_from"] >= FEAT_START) & (balances["valid_from"] <= FEAT_END)
    ]
    bal_c = bal_w.merge(prod_client, on="product_id", how="left")
    bal_c = bal_c[bal_c["client_id"].isin(risk_ids)]
    feat = feat.merge(
        bal_c.groupby("client_id")["balance"].mean().reset_index(name="avg_balance"),
        on="client_id", how="left",
    )

    # 9. balance_trend (slope of monthly median balance)
    bal_c = bal_c.copy()
    bal_c["mo"] = bal_c["valid_from"].dt.year * 12 + bal_c["valid_from"].dt.month
    bal_mo = bal_c.groupby(["client_id", "mo"])["balance"].median().reset_index(name="y")
    feat = feat.merge(
        bal_mo.groupby("client_id").apply(_slope).reset_index(name="balance_trend"),
        on="client_id", how="left",
    )

    # 10. n_contacts
    con_w = contacts[(contacts["date"] >= FEAT_START) & (contacts["date"] <= FEAT_END)]
    con_w = con_w[con_w["client_id"].isin(risk_ids)]
    feat = feat.merge(
        con_w.groupby("client_id").size().reset_index(name="n_contacts"),
        on="client_id", how="left",
    )

    # Intermediate frames reused by enhanced features
    return feat, txn_c, bal_c, con_w


def build_added_features(feat, clients, prods_at_ref, txn_c, bal_c, con_w):
    # 1. age
    feat = feat.merge(clients[["client_id", "age"]], on="client_id", how="left")

    # 2. credit_rating
    cr = clients[["client_id", "credit_rating"]].copy()
    cr["credit_rating"] = pd.to_numeric(cr["credit_rating"], errors="coerce")
    feat = feat.merge(cr, on="client_id", how="left")

    # 3. has_deposit
    deposit_ids = set(prods_at_ref[prods_at_ref["product_domain"] == "DEPOSITS"]["client_id"])
    feat["has_deposit"] = feat["client_id"].isin(deposit_ids).astype(int)

    # 4. digital_txn_ratio
    t = txn_c.copy()
    t["is_digital"] = t["channel"].isin(DIGITAL_CHANNELS).astype(int)
    chan = t.groupby("client_id").agg(
        n_total=("is_digital", "size"), n_digital=("is_digital", "sum"),
    ).reset_index()
    chan["digital_txn_ratio"] = chan["n_digital"] / chan["n_total"]
    feat = feat.merge(chan[["client_id", "digital_txn_ratio"]], on="client_id", how="left")

    # 5. debit_credit_ratio
    dir_counts = txn_c.groupby(["client_id", "direction"]).size().unstack(fill_value=0).reset_index()
    for c in ("D", "C"):
        if c not in dir_counts.columns:
            dir_counts[c] = 0
    dir_counts["debit_credit_ratio"] = dir_counts["D"] / (dir_counts["C"] + 1)
    feat = feat.merge(dir_counts[["client_id", "debit_credit_ratio"]], on="client_id", how="left")

    # 6. balance_volatility (std of monthly median balances)
    bal_vol = bal_c.groupby(["client_id", "mo"])["balance"].median().reset_index()
    feat = feat.merge(
        bal_vol.groupby("client_id")["balance"].std().reset_index(name="balance_volatility"),
        on="client_id", how="left",
    )

    # 7. txn_amount_std
    feat = feat.merge(
        txn_c.groupby("client_id")["amount"].std().reset_index(name="txn_amount_std"),
        on="client_id", how="left",
    )

    # 8. recency_days
    last_txn = txn_c.groupby("client_id")["date"].max().reset_index(name="last_txn_date")
    last_txn["recency_days"] = (REF - last_txn["last_txn_date"]).dt.days
    feat = feat.merge(last_txn[["client_id", "recency_days"]], on="client_id", how="left")

    # 9. n_complaint_contacts
    comp = con_w[con_w["case_type"].isin(COMPLAINT_TYPES)]
    feat = feat.merge(
        comp.groupby("client_id").size().reset_index(name="n_complaint_contacts"),
        on="client_id", how="left",
    )

    # 10. avg_product_age_years
    pa = prods_at_ref.copy()
    pa["product_age_days"] = (REF - pa["opening_date"]).dt.days
    avg_age = pa.groupby("client_id")["product_age_days"].mean().reset_index()
    avg_age["avg_product_age_years"] = avg_age["product_age_days"] / 365.25
    feat = feat.merge(
        avg_age[["client_id", "avg_product_age_years"]], on="client_id", how="left",
    )

    return feat


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------
def impute(feat, features):
    """Fill NaN in place. Behavioral → 0, recency → max window, demographic → median."""
    behavioral = [
        "avg_txn_per_month", "avg_txn_amount", "txn_trend",
        "avg_balance", "balance_trend", "n_contacts",
        "digital_txn_ratio", "debit_credit_ratio",
        "balance_volatility", "txn_amount_std", "n_complaint_contacts",
    ]
    for col in behavioral:
        if col in feat.columns:
            feat[col] = feat[col].fillna(0)

    if "recency_days" in feat.columns:
        feat["recency_days"] = feat["recency_days"].fillna((REF - FEAT_START).days)

    for col in ["tenure_years", "age", "credit_rating", "avg_product_age_years"]:
        if col in feat.columns and feat[col].isnull().any():
            feat[col] = feat[col].fillna(feat[col].median())

    return feat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading raw tables from {RAW} ...")
    clients, products, transactions, balances, contacts = load_raw()

    print("Building labels ...")
    feat, prods_at_ref = build_labels(products)
    print(f"  at-risk: {len(feat):,}  churn rate: {feat['churned'].mean():.1%}")

    print("Building 10 baseline features ...")
    feat, txn_c, bal_c, con_w = build_baseline_features(
        feat, clients, products, prods_at_ref, transactions, balances, contacts
    )

    # 1) raw (pre-imputation) — used by notebook 02 for EDA on missingness
    raw_path = PROCESSED / "churn_features_raw.csv"
    feat[["client_id", *BASELINE_FEATURES, "churned"]].to_csv(raw_path, index=False)
    print(f"  wrote {raw_path}")

    # 2) clean (imputed) — used by notebook 03 baseline model
    clean_feat = impute(feat.copy(), BASELINE_FEATURES)
    clean_path = PROCESSED / "churn_features_clean.csv"
    clean_feat[["client_id", *BASELINE_FEATURES, "churned"]].to_csv(clean_path, index=False)
    print(f"  wrote {clean_path}")

    print("Building 10 added features ...")
    feat = build_added_features(feat, clients, prods_at_ref, txn_c, bal_c, con_w)

    # 3) enhanced (imputed, 20 features) — used by notebook 04 and the bandit
    feat = impute(feat, ALL_FEATURES)
    enh_path = PROCESSED / "churn_features_enhanced.csv"
    feat[["client_id", *ALL_FEATURES, "churned"]].to_csv(enh_path, index=False)
    print(f"  wrote {enh_path}")

    print(f"\nDone. {len(feat):,} rows x {len(ALL_FEATURES)} features.")


if __name__ == "__main__":
    main()
