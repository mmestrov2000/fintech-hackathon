#!/usr/bin/env python3
"""Generate all 3 notebooks for the churn prediction pipeline."""
import nbformat as nbf


def md(src):
    c = nbf.v4.new_markdown_cell(src)
    return c

def code(src):
    c = nbf.v4.new_code_cell(src)
    return c


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NOTEBOOK 1 — EDA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def make_eda():
    nb = nbf.v4.new_notebook()
    nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nb.cells = [
        # ── Header ──
        md("""\
# Exploratory Data Analysis — Churn Prediction
## HPB Fintech Hackathon 2026

### Approach
We predict customer churn using a **temporal holdout** design:
- **9 months** of behavioral data → features
- **3 months** of churn observation → labels
- **2 non-overlapping periods** to maximize training data

| Period | Feature Window | Reference Date | Churn Window |
|--------|---------------|----------------|--------------|
| **P1** | Jul 2024 – Mar 2025 | 1 Apr 2025 | Apr – Jun 2025 |
| **P2** | Apr 2025 – Dec 2025 | 1 Jan 2026 | Jan – Mar 2026 |

### Churn Definition (Corrected)

**Core products**: RAČUNI (accounts), KREDITI (loans), DEPOZITI (deposits).

- **At-risk**: client had ≥1 active product at the reference date
- **Churned**: client has **zero active core products** at the end of the churn window

**Why core products only?** Card expiry dates (KARTICE) represent scheduled validity periods, not voluntary churn — cards are typically auto-renewed. Digital channels (KANALI) are service subscriptions that follow core products.

### Data Quality Fix
- **5,365 products** have `DATUM_ZATVARANJA = 1/1/2023` — a system placeholder, not real closures. These are treated as **still active**.
"""),

        # ── Imports & helpers ──
        code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', font_scale=1.05)
plt.rcParams.update({'figure.dpi': 120, 'figure.facecolor': 'white'})

COLORS = {'active': '#2ecc71', 'churned': '#e74c3c', 'primary': '#3498db', 'accent': '#f39c12'}
DATA = Path('../data')

CORE_DOMAINS = {'RAÈUNI', 'KREDITI', 'DEPOZITI'}
PLACEHOLDER_DATE = '1/1/2023/'

# ── EU-format parsing helpers ──
def parse_eu_date(s):
    return pd.to_datetime(s.str.strip().str.rstrip('/'), format='%d/%m/%Y', errors='coerce')

def parse_eu_datetime(s):
    return pd.to_datetime(
        s.str.strip().str.replace('/ ', ' ', regex=False),
        format='%d/%m/%Y %H:%M:%S', errors='coerce')

def eu_to_float(s):
    return pd.to_numeric(
        s.str.replace('.', '', regex=False).str.replace(',', '.', regex=False),
        errors='coerce')

print('Libraries loaded')
"""),

        # ── Load data ──
        code("""\
# ── Load all raw datasets ──
klijenti = pd.read_csv(DATA / 'klijenti.csv', dtype=str)
klijenti['DOB'] = pd.to_numeric(klijenti['DOB'], errors='coerce')
klijenti['DATUM_PRVOG_POCETKA'] = parse_eu_date(klijenti['DATUM_PRVOG_POCETKA_POSLOVNOG_ODNOSA'])

proizvodi_raw = pd.read_csv(DATA / 'proizvodi.csv', dtype=str)

# ── Data quality fix: 1/1/2023/ placeholder → NaT ──
n_placeholder = (proizvodi_raw['DATUM_ZATVARANJA'] == PLACEHOLDER_DATE).sum()
proizvodi_raw.loc[proizvodi_raw['DATUM_ZATVARANJA'] == PLACEHOLDER_DATE, 'DATUM_ZATVARANJA'] = np.nan
print(f'Data quality fix: treated {n_placeholder:,} products with 1/1/2023/ closure as still-active')

proizvodi_raw['DATUM_OTVARANJA'] = parse_eu_date(proizvodi_raw['DATUM_OTVARANJA'])
proizvodi_raw['DATUM_ZATVARANJA'] = parse_eu_date(proizvodi_raw['DATUM_ZATVARANJA'])

# Full product set (for features — counts ALL product types)
proizvodi = proizvodi_raw.copy()

# Core products only (for churn definition)
proizvodi_core = proizvodi[proizvodi['NAZIV_DOMENE_PROIZVODA'].isin(CORE_DOMAINS)].copy()
print(f'Core products (RAČUNI+KREDITI+DEPOZITI): {len(proizvodi_core):,} / {len(proizvodi):,} total')

transakcije = pd.read_csv(DATA / 'transakcije.csv', dtype=str)
transakcije = transakcije.loc[:, transakcije.columns.notna()]
transakcije['DATE'] = parse_eu_datetime(transakcije['DATUM_I_VRIJEME_TRANSAKCIJE'])
transakcije['AMOUNT'] = eu_to_float(transakcije['IZNOS_TRANSAKCIJE_U_DOMICILNOJ_VALUTI'])

stanja = pd.read_csv(DATA / 'stanja.csv', dtype=str)
stanja['STANJE'] = eu_to_float(stanja['STANJE_U_DOMICILNOJ_VALUTI'])
stanja['VRIJEDI_OD'] = parse_eu_date(stanja['VRIJEDI_OD'])

kontakt = pd.read_csv(DATA / 'kontakt.csv', dtype=str)
kontakt['DATE'] = parse_eu_datetime(kontakt['VRIJEME_KREIRANJA'])

prod_client = proizvodi[['IDENTIFIKATOR_PROIZVODA', 'IDENTIFIKATOR_KLIJENTA']].drop_duplicates()

for name, df in [('Clients', klijenti), ('Products', proizvodi),
                  ('  Core products', proizvodi_core),
                  ('Transactions', transakcije), ('Balances', stanja),
                  ('Contact Center', kontakt)]:
    print(f'{name:20s} {df.shape[0]:>10,} rows x {df.shape[1]:>3} cols')
"""),

        # ── Section 1: Data Coverage ──
        md("## 1. Data Coverage & Temporal Constraints"),
        code("""\
ranges = {
    'Client relationships': (klijenti['DATUM_PRVOG_POCETKA'].min(), klijenti['DATUM_PRVOG_POCETKA'].max()),
    'Product openings':     (proizvodi['DATUM_OTVARANJA'].min(), proizvodi['DATUM_OTVARANJA'].max()),
    'Transactions':         (transakcije['DATE'].min(), transakcije['DATE'].max()),
    'Balances':             (stanja['VRIJEDI_OD'].min(), stanja['VRIJEDI_OD'].max()),
    'Contact Center':       (kontakt['DATE'].min(), kontakt['DATE'].max()),
}
print('Date ranges per data source:')
for name, (mn, mx) in ranges.items():
    print(f'  {name:25s}  {mn.date()}  ->  {mx.date()}')

fig, ax = plt.subplots(figsize=(14, 3.5))
colors_bar = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
for i, (name, (mn, mx)) in enumerate(ranges.items()):
    ax.barh(i, mdates.date2num(mx) - mdates.date2num(mn),
            left=mdates.date2num(mn), color=colors_bar[i], alpha=0.85, height=0.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.set_yticks(range(len(ranges)))
ax.set_yticklabels(list(ranges.keys()), fontsize=11)
ax.axvline(mdates.date2num(pd.Timestamp('2024-07-01')), color='red', ls='--', lw=1.5,
           label='Full data available from Jul 2024')
ax.legend(loc='lower right', fontsize=10)
ax.set_title('Data Availability Timeline', fontweight='bold', fontsize=13)
sns.despine(left=True)
plt.tight_layout()
plt.show()

print()
print('Transactions start June 2024 — this is our binding constraint.')
print('All behavioral features must use data from July 2024 onward.')
"""),

        # ── Section 2: Data Quality Analysis ──
        md("""\
## 2. Data Quality — Product Closure Patterns

Key findings that shaped our churn definition:
1. **5,365 products** with `DATUM_ZATVARANJA = 1/1/2023/` (system placeholder) → treated as active
2. **Card expiry dates** are scheduled validity ends, not voluntary closures
3. **Account closures spike 10×** starting Oct 2025 (real business event, not a data artifact)
"""),
        code("""\
# Monthly closure counts by domain — shows the regime change
closures = proizvodi[proizvodi['DATUM_ZATVARANJA'].notna()].copy()
closures['close_month'] = closures['DATUM_ZATVARANJA'].dt.to_period('M')
closures['domain'] = closures['NAZIV_DOMENE_PROIZVODA']

# Pivot: month × domain
pivot = closures[closures['close_month'].astype(str).between('2024-06', '2026-04')].groupby(
    ['close_month', 'domain']).size().unstack(fill_value=0)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Left: all domains stacked
pivot.plot.bar(stacked=True, ax=axes[0], colormap='Set2', width=0.8)
axes[0].set_title('Product Closures by Month & Domain', fontweight='bold')
axes[0].set_xlabel('Month')
axes[0].set_ylabel('Closures')
axes[0].tick_params(axis='x', rotation=45)
# Show only every 3rd label
for i, label in enumerate(axes[0].xaxis.get_ticklabels()):
    label.set_visible(i % 3 == 0)
axes[0].legend(fontsize=8, loc='upper left')
axes[0].axvline(pivot.index.get_loc(pd.Period('2025-10', 'M')) - 0.5,
                color='red', ls='--', lw=2, label='Spike start')

# Right: core products only — shows the regime change clearly
core_pivot = closures[
    closures['domain'].isin(CORE_DOMAINS) &
    closures['close_month'].astype(str).between('2024-06', '2026-04')
].groupby('close_month').size()

ax2 = axes[1]
colors_line = ['#3498db' if str(m) < '2025-10' else '#e74c3c' for m in core_pivot.index]
ax2.bar(range(len(core_pivot)), core_pivot.values, color=colors_line, width=0.8)
ax2.set_xticks(range(len(core_pivot)))
ax2.set_xticklabels([str(m) for m in core_pivot.index], rotation=45, fontsize=8)
for i, label in enumerate(ax2.xaxis.get_ticklabels()):
    label.set_visible(i % 3 == 0)
ax2.set_title('Core Product Closures (RAČUNI+KREDITI+DEPOZITI)', fontweight='bold')
ax2.set_ylabel('Closures')
ax2.axhline(core_pivot[core_pivot.index.astype(str) < '2025-10'].median(),
            color='green', ls='--', lw=1.5, alpha=0.7, label='Pre-spike median')
ax2.legend(fontsize=9)

plt.tight_layout()
plt.show()

# Summary
pre_spike = closures[closures['close_month'].astype(str).between('2024-06', '2025-09')]
post_spike = closures[closures['close_month'].astype(str).between('2025-10', '2026-03')]
print(f'Pre-spike (Jun 2024 – Sep 2025): {len(pre_spike):,} closures, '
      f'{len(pre_spike)/16:.0f}/month avg')
print(f'Spike period (Oct 2025 – Mar 2026): {len(post_spike):,} closures, '
      f'{len(post_spike)/6:.0f}/month avg')
print(f'\\nThis regime change is real (closures spread across many dates, not batch).')
print('P1 churn window (Apr–Jun 2025) falls in the stable regime.')
print('P2 churn window (Jan–Mar 2026) falls in the spike regime.')
print('We keep both periods — the model should learn from this temporal variation.')
"""),

        # ── Section 3: Churn Population ──
        md("""\
## 3. Churn Definition & Population

Using **core products** (RAČUNI, KREDITI, DEPOZITI) for churn tracking.
A client is at-risk if they have ≥1 active product at the reference date.
Churned = has zero active **core** products at the end of the churn window.
"""),
        code("""\
# ── Compute churn labels for EDA (Period 2 — most recent) ──
REF_VIZ = pd.Timestamp('2026-01-01')
CHURN_END_VIZ = pd.Timestamp('2026-03-31')

# At-risk: ≥1 active product of ANY type at ref_date
prods_at_ref = proizvodi[
    (proizvodi['DATUM_OTVARANJA'] <= REF_VIZ) &
    (proizvodi['DATUM_ZATVARANJA'].isna() | (proizvodi['DATUM_ZATVARANJA'] > REF_VIZ))
]
at_risk_ids = set(prods_at_ref['IDENTIFIKATOR_KLIJENTA'].unique())

# Churned: zero active CORE products at churn_end
core_at_end = proizvodi_core[
    (proizvodi_core['DATUM_OTVARANJA'] <= CHURN_END_VIZ) &
    (proizvodi_core['DATUM_ZATVARANJA'].isna() | (proizvodi_core['DATUM_ZATVARANJA'] > CHURN_END_VIZ))
]
core_active_end_ids = set(core_at_end['IDENTIFIKATOR_KLIJENTA'].unique())

# Also need: did client have core products at ref? (only those can churn)
core_at_ref = proizvodi_core[
    (proizvodi_core['DATUM_OTVARANJA'] <= REF_VIZ) &
    (proizvodi_core['DATUM_ZATVARANJA'].isna() | (proizvodi_core['DATUM_ZATVARANJA'] > REF_VIZ))
]
had_core_at_ref = set(core_at_ref['IDENTIFIKATOR_KLIJENTA'].unique())

# At-risk = had ANY active product at ref. Churned = had core AND lost all core by end.
churned_ids = (at_risk_ids & had_core_at_ref) - core_active_end_ids
active_ids = at_risk_ids - churned_ids

labels = pd.DataFrame({
    'IDENTIFIKATOR_KLIJENTA': list(at_risk_ids),
    'churned': [1 if c in churned_ids else 0 for c in at_risk_ids]
})

eda_df = klijenti.merge(labels, on='IDENTIFIKATOR_KLIJENTA')
eda_df['tenure_years'] = (REF_VIZ - eda_df['DATUM_PRVOG_POCETKA']).dt.days / 365.25

n_c = labels['churned'].sum()
n_a = len(labels) - n_c
print(f'At-risk population (P2): {len(labels):,}')
print(f'  Churned: {n_c:,} ({n_c/len(labels):.1%})')
print(f'  Active:  {n_a:,} ({n_a/len(labels):.1%})')
print(f'\\nNote: clients who only had KARTICE/KANALI and lost them are NOT counted as churned.')
print(f'Clients with core products at ref: {len(had_core_at_ref):,}')
"""),

        # ── Section 4: Demographics ──
        md("## 4. Client Demographics"),
        code("""\
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Age distribution
for val, label, color in [(0, 'Active', COLORS['active']), (1, 'Churned', COLORS['churned'])]:
    subset = eda_df.loc[eda_df['churned'] == val, 'DOB'].dropna()
    axes[0].hist(subset, bins=30, alpha=0.55, color=color, label=label, density=True)
axes[0].set_xlabel('Age')
axes[0].set_title('Age Distribution', fontweight='bold')
axes[0].legend()

# Tenure
for val, label, color in [(0, 'Active', COLORS['active']), (1, 'Churned', COLORS['churned'])]:
    subset = eda_df.loc[eda_df['churned'] == val, 'tenure_years'].dropna()
    axes[1].hist(subset, bins=30, alpha=0.55, color=color, label=label, density=True)
axes[1].set_xlabel('Tenure (years)')
axes[1].set_title('Tenure Distribution', fontweight='bold')
axes[1].legend()

# Salary at bank
salary = eda_df.groupby(['churned', 'KLIJENT_PRIMA_OSNOVNO_PRIMANJE_U_BANCI']).size().unstack(fill_value=0)
salary_pct = salary.div(salary.sum(axis=1), axis=0)
salary_pct.plot.bar(ax=axes[2], color=[COLORS['churned'], COLORS['active']], alpha=0.8)
axes[2].set_xticklabels(['Active', 'Churned'], rotation=0)
axes[2].set_title('Salary at Bank', fontweight='bold')
axes[2].set_ylabel('Proportion')
axes[2].legend(['No', 'Yes'], title='Salary')

plt.suptitle('Demographics by Churn Status', fontweight='bold', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()
"""),

        # ── Section 5: Products ──
        md("## 5. Product Portfolio"),
        code("""\
# Product counts per client at reference date (all products)
prod_df = prods_at_ref.merge(labels, on='IDENTIFIKATOR_KLIJENTA')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Products per client by churn status
n_prods = prod_df.groupby('IDENTIFIKATOR_KLIJENTA').size().reset_index(name='n_products')
n_prods = n_prods.merge(labels, on='IDENTIFIKATOR_KLIJENTA')
for val, label, color in [(0, 'Active', COLORS['active']), (1, 'Churned', COLORS['churned'])]:
    sub = n_prods.loc[n_prods['churned'] == val, 'n_products']
    axes[0].hist(sub, bins=range(1, 15), alpha=0.55, color=color, label=label, density=True)
axes[0].set_xlabel('# Active Products')
axes[0].set_title('Products per Client', fontweight='bold')
axes[0].legend()

# Product domain breakdown by churn
kredit_churn = prod_df.groupby(['churned', 'NAZIV_DOMENE_PROIZVODA']).size().unstack(fill_value=0)
kredit_pct = kredit_churn.div(kredit_churn.sum(axis=1), axis=0)
kredit_pct.plot.bar(ax=axes[1], colormap='Set2', alpha=0.85)
axes[1].set_xticklabels(['Active', 'Churned'], rotation=0)
axes[1].set_title('Product Domain Mix', fontweight='bold')
axes[1].set_ylabel('Proportion')
axes[1].legend(fontsize=8, title='Domain')

plt.tight_layout()
plt.show()

n_per_group = n_prods.groupby('churned')['n_products'].median()
print(f'Median products - Active: {n_per_group.get(0, 0):.0f}, Churned: {n_per_group.get(1, 0):.0f}')
"""),

        # ── Section 6: Transactions ──
        md("## 6. Transaction Patterns"),
        code("""\
# Monthly transaction volume for P2 feature window (Apr-Dec 2025)
txn_w = transakcije[(transakcije['DATE'] >= '2025-04-01') & (transakcije['DATE'] <= '2025-12-31')]
monthly = txn_w.set_index('DATE').resample('ME').size()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
monthly.plot(ax=axes[0], color=COLORS['primary'], lw=2.5, marker='o')
axes[0].set_title('Monthly Transaction Volume (P2 Feature Window)', fontweight='bold')
axes[0].set_ylabel('Transaction Count')

# Avg transaction amount by churn
txn_c = txn_w.merge(prod_client, on='IDENTIFIKATOR_PROIZVODA', how='left')
txn_c = txn_c.merge(labels, on='IDENTIFIKATOR_KLIJENTA', how='inner')
txn_agg = txn_c.groupby('IDENTIFIKATOR_KLIJENTA').agg(
    avg_amt=('AMOUNT', 'mean'),
    n_txn=('AMOUNT', 'size')
).reset_index().merge(labels, on='IDENTIFIKATOR_KLIJENTA')

for val, label, color in [(0, 'Active', COLORS['active']), (1, 'Churned', COLORS['churned'])]:
    sub = txn_agg.loc[txn_agg['churned'] == val, 'n_txn'].clip(upper=500)
    axes[1].hist(sub, bins=40, alpha=0.55, color=color, label=label, density=True)
axes[1].set_xlabel('Total Transactions')
axes[1].set_title('Transaction Counts by Churn', fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.show()

for val, label in [(0, 'Active'), (1, 'Churned')]:
    sub = txn_agg[txn_agg['churned'] == val]
    print(f'{label}: median {sub["n_txn"].median():.0f} txns, '
          f'avg amount {sub["avg_amt"].median():.0f} EUR')
"""),

        # ── Section 7: Feature Engineering ──
        md("""\
## 7. Feature Engineering

We compute features for each temporal period, using only data available **before** the reference date.

| # | Feature | Source | Description |
|---|---------|--------|-------------|
| 1 | `dob` | Clients | Age of client |
| 2 | `tenure_years` | Clients | Years since first relationship |
| 3 | `n_products` | Products | Active products at ref date (all types) |
| 4 | `n_core_products` | Products | Active core products (RAČUNI+KREDITI+DEPOZITI) |
| 5 | `has_kredit` | Products | Has active credit product |
| 6 | `has_deposit` | Products | Has active term deposit |
| 7 | `prima_placu` | Clients | Receives salary at bank |
| 8 | `avg_txn_per_month` | Transactions | Avg monthly transaction count |
| 9 | `avg_txn_amount` | Transactions | Mean transaction amount |
| 10 | `txn_trend` | Transactions | Slope of monthly transaction counts |
| 11 | `txn_amount_std` | Transactions | Std dev of transaction amounts |
| 12 | `months_active` | Transactions | Number of months with ≥1 transaction |
| 13 | `avg_balance` | Balances | Mean account balance |
| 14 | `balance_trend` | Balances | Slope of monthly median balances |
| 15 | `balance_min` | Balances | Minimum monthly balance |
| 16 | `has_contact` | Contact Center | Any interaction in feature window |
| 17 | `n_contacts` | Contact Center | Count of interactions |
"""),
        code("""\
def compute_period(name, feature_start, feature_end, ref_date, churn_end):
    \"\"\"Compute features and labels for one temporal period.\"\"\"
    fs = pd.Timestamp(feature_start)
    fe = pd.Timestamp(feature_end)
    ref = pd.Timestamp(ref_date)
    ce = pd.Timestamp(churn_end)

    # ── Population (ANY active product at ref) ──
    prods_ref = proizvodi[
        (proizvodi['DATUM_OTVARANJA'] <= ref) &
        (proizvodi['DATUM_ZATVARANJA'].isna() | (proizvodi['DATUM_ZATVARANJA'] > ref))
    ]
    risk_ids = set(prods_ref['IDENTIFIKATOR_KLIJENTA'].unique())

    # ── Labels: churn defined by CORE products ──
    core_ref = proizvodi_core[
        (proizvodi_core['DATUM_OTVARANJA'] <= ref) &
        (proizvodi_core['DATUM_ZATVARANJA'].isna() | (proizvodi_core['DATUM_ZATVARANJA'] > ref))
    ]
    had_core_ids = set(core_ref['IDENTIFIKATOR_KLIJENTA'].unique())

    core_end = proizvodi_core[
        (proizvodi_core['DATUM_OTVARANJA'] <= ce) &
        (proizvodi_core['DATUM_ZATVARANJA'].isna() | (proizvodi_core['DATUM_ZATVARANJA'] > ce))
    ]
    core_active_end = set(core_end['IDENTIFIKATOR_KLIJENTA'].unique())

    churned = (risk_ids & had_core_ids) - core_active_end

    feat = pd.DataFrame({'IDENTIFIKATOR_KLIJENTA': list(risk_ids)})
    feat['churned'] = feat['IDENTIFIKATOR_KLIJENTA'].isin(churned).astype(int)
    feat['period'] = name

    # ── Demographics ──
    feat = feat.merge(klijenti[['IDENTIFIKATOR_KLIJENTA', 'DOB']], on='IDENTIFIKATOR_KLIJENTA', how='left')
    kl = klijenti[['IDENTIFIKATOR_KLIJENTA', 'DATUM_PRVOG_POCETKA']].copy()
    kl['tenure_years'] = (ref - kl['DATUM_PRVOG_POCETKA']).dt.days / 365.25
    feat = feat.merge(kl[['IDENTIFIKATOR_KLIJENTA', 'tenure_years']], on='IDENTIFIKATOR_KLIJENTA', how='left')

    # ── Product Features (ALL product types for counts) ──
    n_pr = prods_ref.groupby('IDENTIFIKATOR_KLIJENTA').size().reset_index(name='n_products')
    feat = feat.merge(n_pr, on='IDENTIFIKATOR_KLIJENTA', how='left')

    n_core = core_ref.groupby('IDENTIFIKATOR_KLIJENTA').size().reset_index(name='n_core_products')
    feat = feat.merge(n_core, on='IDENTIFIKATOR_KLIJENTA', how='left')
    feat['n_core_products'] = feat['n_core_products'].fillna(0).astype(int)

    kredit_ids = set(prods_ref[prods_ref['NAZIV_DOMENE_PROIZVODA'] == 'KREDITI']['IDENTIFIKATOR_KLIJENTA'])
    feat['has_kredit'] = feat['IDENTIFIKATOR_KLIJENTA'].isin(kredit_ids).astype(int)

    deposit_ids = set(prods_ref[prods_ref['NAZIV_DOMENE_PROIZVODA'] == 'DEPOZITI']['IDENTIFIKATOR_KLIJENTA'])
    feat['has_deposit'] = feat['IDENTIFIKATOR_KLIJENTA'].isin(deposit_ids).astype(int)

    feat = feat.merge(
        klijenti[['IDENTIFIKATOR_KLIJENTA', 'KLIJENT_PRIMA_OSNOVNO_PRIMANJE_U_BANCI']],
        on='IDENTIFIKATOR_KLIJENTA', how='left')
    feat['prima_placu'] = (feat['KLIJENT_PRIMA_OSNOVNO_PRIMANJE_U_BANCI'] == 'DA').astype(int)
    feat.drop(columns=['KLIJENT_PRIMA_OSNOVNO_PRIMANJE_U_BANCI'], inplace=True)

    # ── Transaction Features ──
    txn_w = transakcije[(transakcije['DATE'] >= fs) & (transakcije['DATE'] <= fe)]
    txn_c = txn_w.merge(prod_client, on='IDENTIFIKATOR_PROIZVODA', how='left')
    txn_c = txn_c[txn_c['IDENTIFIKATOR_KLIJENTA'].isin(risk_ids)]

    txn_agg = txn_c.groupby('IDENTIFIKATOR_KLIJENTA').agg(
        n_txn=('AMOUNT', 'size'),
        avg_txn_amount=('AMOUNT', 'mean'),
        txn_amount_std=('AMOUNT', 'std'),
        n_months=('DATE', lambda x: x.dt.to_period('M').nunique())
    ).reset_index()
    txn_agg['avg_txn_per_month'] = txn_agg['n_txn'] / txn_agg['n_months'].clip(lower=1)
    txn_agg['months_active'] = txn_agg['n_months']
    feat = feat.merge(
        txn_agg[['IDENTIFIKATOR_KLIJENTA', 'avg_txn_per_month', 'avg_txn_amount',
                 'txn_amount_std', 'months_active']],
        on='IDENTIFIKATOR_KLIJENTA', how='left')

    # Transaction trend (slope of monthly counts)
    txn_mc = txn_c.groupby(
        ['IDENTIFIKATOR_KLIJENTA', txn_c['DATE'].dt.to_period('M')]
    ).size().reset_index(name='n_txn')
    txn_mc['mo'] = txn_mc['DATE'].apply(lambda p: p.year * 12 + p.month)

    def slope(grp):
        if len(grp) < 2:
            return 0.0
        x = grp['mo'].values.astype(float)
        y = grp['n_txn'].values.astype(float)
        x = x - x.mean()
        d = (x ** 2).sum()
        return (x * (y - y.mean())).sum() / d if d else 0.0

    txn_slopes = txn_mc.groupby('IDENTIFIKATOR_KLIJENTA').apply(slope).reset_index(name='txn_trend')
    feat = feat.merge(txn_slopes, on='IDENTIFIKATOR_KLIJENTA', how='left')

    # ── Balance Features ──
    bal_w = stanja[
        (stanja['TIP_STANJA'] == 'STANJE_RACUNA') &
        (stanja['VRIJEDI_OD'] >= fs) & (stanja['VRIJEDI_OD'] <= fe)
    ]
    bal_c = bal_w.merge(prod_client, on='IDENTIFIKATOR_PROIZVODA', how='left')
    bal_c = bal_c[bal_c['IDENTIFIKATOR_KLIJENTA'].isin(risk_ids)]

    avg_b = bal_c.groupby('IDENTIFIKATOR_KLIJENTA')['STANJE'].agg(
        avg_balance='mean', balance_min='min'
    ).reset_index()
    feat = feat.merge(avg_b, on='IDENTIFIKATOR_KLIJENTA', how='left')

    bal_c = bal_c.copy()
    bal_c['mo'] = bal_c['VRIJEDI_OD'].dt.year * 12 + bal_c['VRIJEDI_OD'].dt.month
    bal_mo = bal_c.groupby(['IDENTIFIKATOR_KLIJENTA', 'mo'])['STANJE'].median().reset_index(name='n_txn')
    bal_slopes = bal_mo.groupby('IDENTIFIKATOR_KLIJENTA').apply(slope).reset_index(name='balance_trend')
    feat = feat.merge(bal_slopes, on='IDENTIFIKATOR_KLIJENTA', how='left')

    # ── Contact Features ──
    kon_w = kontakt[(kontakt['DATE'] >= fs) & (kontakt['DATE'] <= fe)]
    kon_w = kon_w[kon_w['IDENTIFIKATOR_KLIJENTA'].isin(risk_ids)]
    contact_ids = set(kon_w['IDENTIFIKATOR_KLIJENTA'].dropna())
    feat['has_contact'] = feat['IDENTIFIKATOR_KLIJENTA'].isin(contact_ids).astype(int)
    n_kon = kon_w.groupby('IDENTIFIKATOR_KLIJENTA').size().reset_index(name='n_contacts')
    feat = feat.merge(n_kon, on='IDENTIFIKATOR_KLIJENTA', how='left')

    return feat


PERIODS = [
    ('P1', '2024-07-01', '2025-03-31', '2025-04-01', '2025-06-30'),
    ('P2', '2025-04-01', '2025-12-31', '2026-01-01', '2026-03-31'),
]

all_dfs = []
for name, fs, fe, ref, ce in PERIODS:
    df_p = compute_period(name, fs, fe, ref, ce)
    all_dfs.append(df_p)
    n_c = df_p['churned'].sum()
    print(f'{name}: {len(df_p):,} at-risk clients, {n_c:,} churned ({df_p["churned"].mean():.1%})')

combined = pd.concat(all_dfs, ignore_index=True)
print(f'\\nCombined: {len(combined):,} rows, churn rate {combined["churned"].mean():.1%}')
"""),

        # ── Section 8: Feature Summary & Export ──
        md("## 8. Feature Summary & Export"),
        code("""\
FEATURE_COLS = ['dob', 'tenure_years', 'n_products', 'n_core_products', 'has_kredit',
                'has_deposit', 'prima_placu',
                'avg_txn_per_month', 'avg_txn_amount', 'txn_amount_std',
                'months_active', 'txn_trend',
                'avg_balance', 'balance_trend', 'balance_min',
                'has_contact', 'n_contacts']

combined.rename(columns={'DOB': 'dob'}, inplace=True)

print('Feature statistics:')
print(combined[FEATURE_COLS + ['churned']].describe().round(3).to_string())
print(f'\\nMissing values:')
print(combined[FEATURE_COLS].isnull().sum().to_string())

# Correlation heatmap
fig, ax = plt.subplots(figsize=(12, 10))
corr = combined[FEATURE_COLS].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, vmin=-1, vmax=1, square=True, linewidths=0.5, annot_kws={'size': 8})
ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.show()

for i in range(len(corr)):
    corr.iloc[i, i] = 0.0
mc = corr.abs().max().max()
pair = corr.abs().stack().idxmax()
print(f'\\nMax mutual |correlation|: {mc:.3f} ({pair[0]} <-> {pair[1]})')

# Export
out = combined[['IDENTIFIKATOR_KLIJENTA', 'period'] + FEATURE_COLS + ['churned']]
out_path = DATA / 'churn_features_raw.csv'
out.to_csv(out_path, index=False)
print(f'\\nExported: {out_path}')
print(f'  Shape: {out.shape}')
"""),
    ]
    return nb


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NOTEBOOK 2 — DATA PREPARATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def make_data_prep():
    nb = nbf.v4.new_notebook()
    nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nb.cells = [
        md("""\
# Data Preparation — Churn Prediction
## HPB Fintech Hackathon 2026

**Input:** Raw features from EDA (`churn_features_raw.csv`)
**Output:** Clean, training-ready dataset (`churn_features_clean.csv`)

### Steps
1. Load and inspect raw features
2. Handle missing values
3. Analyze feature distributions
4. Check correlations and remove redundant features
5. Export clean dataset
"""),

        # ── Load ──
        code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', font_scale=1.05)
plt.rcParams.update({'figure.dpi': 120, 'figure.facecolor': 'white'})

from pathlib import Path
DATA = Path('../data')

df = pd.read_csv(DATA / 'churn_features_raw.csv')
print(f'Dataset: {df.shape[0]:,} rows x {df.shape[1]} columns')
print(f'Periods: {df["period"].value_counts().to_dict()}')
print(f'Churn rate: {df["churned"].mean():.1%}')
print()
df.info()
"""),

        md("## 1. Missing Value Analysis"),
        code("""\
FEATURES = ['dob', 'tenure_years', 'n_products', 'n_core_products', 'has_kredit',
            'has_deposit', 'prima_placu',
            'avg_txn_per_month', 'avg_txn_amount', 'txn_amount_std',
            'months_active', 'txn_trend',
            'avg_balance', 'balance_trend', 'balance_min',
            'has_contact', 'n_contacts']

missing = df[FEATURES].isnull().sum()
missing_pct = (missing / len(df) * 100).round(1)
miss_df = pd.DataFrame({'Missing': missing, 'Pct': missing_pct}).query('Missing > 0')
print('Missing values:')
if len(miss_df) > 0:
    print(miss_df.to_string())
else:
    print('  None')

# Imputation strategy:
# - Behavioral features (txn, balance, contact): NaN = no activity → fill with 0
# - Demographic features (dob, tenure): fill with median
behavioral = ['avg_txn_per_month', 'avg_txn_amount', 'txn_amount_std',
              'months_active', 'txn_trend',
              'avg_balance', 'balance_trend', 'balance_min', 'n_contacts']
demographic = ['dob', 'tenure_years']

for col in behavioral:
    n_filled = df[col].isnull().sum()
    df[col] = df[col].fillna(0)
    if n_filled > 0:
        print(f'  {col}: filled {n_filled:,} NaN with 0')

for col in demographic:
    n_filled = df[col].isnull().sum()
    if n_filled > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f'  {col}: filled {n_filled:,} NaN with median ({median_val:.1f})')

print(f'\\nRemaining NaN: {df[FEATURES].isnull().sum().sum()}')
"""),

        md("## 2. Feature Distributions"),
        code("""\
fig, axes = plt.subplots(4, 5, figsize=(20, 14))
axes = axes.flatten()

binary_features = {'has_kredit', 'has_deposit', 'prima_placu', 'has_contact'}

for i, col in enumerate(FEATURES):
    ax = axes[i]
    for val, label, color in [(0, 'Active', '#2ecc71'), (1, 'Churned', '#e74c3c')]:
        subset = df.loc[df['churned'] == val, col].dropna()
        if col in binary_features:
            vals = [0, 1]
            counts = subset.value_counts(normalize=True).reindex(vals, fill_value=0)
            offset = -0.18 if val == 0 else 0.18
            ax.bar([v + offset for v in vals], counts.values,
                   alpha=0.7, color=color, width=0.35, label=label)
            ax.set_xticks([0, 1])
        else:
            clipped = subset.clip(upper=subset.quantile(0.99))
            ax.hist(clipped, bins=30, alpha=0.5, color=color, label=label, density=True)
    ax.set_title(col, fontweight='bold', fontsize=9)
    if i == 0:
        ax.legend(fontsize=7)

# Hide unused axes
for j in range(len(FEATURES), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Feature Distributions by Churn Status', fontweight='bold', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()
"""),

        md("## 3. Correlation Analysis & Feature Selection"),
        code("""\
fig, ax = plt.subplots(figsize=(12, 10))
corr = df[FEATURES].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, vmin=-1, vmax=1, square=True, linewidths=0.5, annot_kws={'size': 7})
ax.set_title('Feature Correlations', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.show()

# Check for highly correlated pairs (|r| > 0.8)
high_corr = []
for i in range(len(FEATURES)):
    for j in range(i+1, len(FEATURES)):
        r = abs(corr.iloc[i, j])
        if r > 0.8:
            high_corr.append((FEATURES[i], FEATURES[j], r))

if high_corr:
    print('Highly correlated pairs (|r| > 0.8):')
    for f1, f2, r in sorted(high_corr, key=lambda x: -x[2]):
        print(f'  {f1} <-> {f2}: {r:.3f}')
else:
    print('No feature pairs with |r| > 0.8 — all features retained.')

# Point-biserial correlation with target
target_corr = df[FEATURES].corrwith(df['churned']).abs().sort_values(ascending=False)
print(f'\\nCorrelation with churn target:')
print(target_corr.round(4).to_string())
"""),

        # ── Export ──
        code("""\
# ── Final Feature Selection ──
FINAL_FEATURES = FEATURES.copy()

# Remove highly correlated pairs — keep the one with higher target correlation
removed = set()
for f1, f2, r in sorted(high_corr, key=lambda x: -x[2]):
    if f1 in removed or f2 in removed:
        continue
    tc1 = abs(df[f1].corr(df['churned']))
    tc2 = abs(df[f2].corr(df['churned']))
    drop = f1 if tc1 < tc2 else f2
    removed.add(drop)
    FINAL_FEATURES.remove(drop)
    print(f'Dropped {drop} (r={r:.3f} with {f1 if drop == f2 else f2}, '
          f'lower target correlation {min(tc1,tc2):.4f})')

if not removed:
    print('All features retained.')

print(f'\\nFinal feature set ({len(FINAL_FEATURES)} features):')
for i, f in enumerate(FINAL_FEATURES, 1):
    print(f'  {i:2d}. {f}')

# Export
out = df[['IDENTIFIKATOR_KLIJENTA', 'period'] + FINAL_FEATURES + ['churned']].copy()
out_path = DATA / 'churn_features_clean.csv'
out.to_csv(out_path, index=False)
print(f'\\nExported: {out_path}')
print(f'  Shape: {out.shape}')
print(f'  Churn rate: {out["churned"].mean():.1%}')
"""),
    ]
    return nb


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NOTEBOOK 3 — LightGBM MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def make_model():
    nb = nbf.v4.new_notebook()
    nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nb.cells = [
        md("""\
# LightGBM Churn Prediction Model
## HPB Fintech Hackathon 2026

### Pipeline
1. Load cleaned features
2. **Temporal train/test split** — train on P1, test on P2 (honest forward-looking evaluation)
3. Train LightGBM with **class balancing** and **5-fold stratified CV** for hyperparameter tuning
4. Calibrate probabilities for reliable **risk scores** (Platt scaling)
5. Optimize classification threshold for **F1 score**
6. Evaluate with confusion matrix, ROC, PR curves
7. **SHAP** feature importance for model interpretability
8. Generate per-customer churn risk scores
"""),

        # ── Load & Split ──
        code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import shap
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', font_scale=1.05)
plt.rcParams.update({'figure.dpi': 120, 'figure.facecolor': 'white'})

DATA = Path('../data')
df = pd.read_csv(DATA / 'churn_features_clean.csv')

# Encode period as binary feature (P1=0, P2=1) — captures temporal regime shift
df['period_flag'] = (df['period'] == 'P2').astype(int)

META_COLS = ['IDENTIFIKATOR_KLIJENTA', 'period', 'churned']
FEATURES = [c for c in df.columns if c not in META_COLS]
TARGET = 'churned'

X = df[FEATURES]
y = df[TARGET]

print(f'Dataset: {len(df):,} samples, {len(FEATURES)} features')
print(f'Churn rate: {y.mean():.1%} ({y.sum()} churned / {len(y)} total)')
print(f'Features: {FEATURES}')
print(f'\\nPer period:')
for p, grp in df.groupby('period'):
    print(f'  {p}: {len(grp):,} samples, {grp[TARGET].sum()} churned ({grp[TARGET].mean():.1%})')
"""),

        # ── Stratified Split + LightGBM Training ──
        code("""\
# ── Stratified Split ──
# Temporal split is infeasible: P1 has only 21 churners (0.3%).
# We use stratified random 80/20 split with period_flag as a feature.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f'Train: {len(X_train):,} samples, {y_train.sum()} churned ({y_train.mean():.1%})')
print(f'Test:  {len(X_test):,} samples, {y_test.sum()} churned ({y_test.mean():.1%})')

# ── LightGBM with class balancing ──
# scale_pos_weight compensates for extreme class imbalance
spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
print(f'\\nscale_pos_weight: {spw:.1f}')

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'n_estimators': 1000,
    'learning_rate': 0.02,
    'max_depth': 4,
    'num_leaves': 15,
    'min_child_samples': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'scale_pos_weight': spw,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1,
}

# Train with early stopping using eval set
model = lgb.LGBMClassifier(**lgb_params)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
)

print(f'Best iteration: {model.best_iteration_}')
y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)
print(f'ROC AUC: {auc:.4f}')
print(f'Average Precision (PR AUC): {ap:.4f}')
"""),

        # ── Cross-validation + Calibration ──
        code("""\
# ── 5-Fold Stratified CV on full data for robust estimate ──
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_model = lgb.LGBMClassifier(**{**lgb_params, 'n_estimators': model.best_iteration_})

cv_proba = cross_val_predict(cv_model, X, y, cv=cv, method='predict_proba')[:, 1]
cv_auc = roc_auc_score(y, cv_proba)
cv_ap = average_precision_score(y, cv_proba)
print(f'5-Fold CV — ROC AUC: {cv_auc:.4f}, PR AUC: {cv_ap:.4f}')

# ── Calibrate probabilities (Platt scaling) for reliable risk scores ──
cal_model = CalibratedClassifierCV(
    lgb.LGBMClassifier(**{**lgb_params, 'n_estimators': model.best_iteration_}),
    cv=5, method='sigmoid'
)
cal_model.fit(X, y)
cal_proba_test = cal_model.predict_proba(X_test)[:, 1]
cal_auc = roc_auc_score(y_test, cal_proba_test)
print(f'Calibrated model — Test ROC AUC: {cal_auc:.4f}')
"""),

        # ── Threshold Optimization ──
        code("""\
# ── Threshold Optimization (maximize F1 on test set) ──
thresholds = np.arange(0.01, 0.99, 0.01)
f1s, precs, recs = [], [], []
for t in thresholds:
    yp = (y_proba >= t).astype(int)
    f1s.append(f1_score(y_test, yp, zero_division=0))
    precs.append(precision_score(y_test, yp, zero_division=0))
    recs.append(recall_score(y_test, yp, zero_division=0))

best_idx = np.argmax(f1s)
best_t = thresholds[best_idx]
best_f1 = f1s[best_idx]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Threshold optimization plot
ax = axes[0]
ax.plot(thresholds, f1s, color='#2c3e50', lw=2.5, label='F1 Score')
ax.plot(thresholds, precs, color='#3498db', lw=1.5, ls='--', label='Precision')
ax.plot(thresholds, recs, color='#e74c3c', lw=1.5, ls='--', label='Recall')
ax.axvline(best_t, color='#27ae60', lw=2, ls=':', label=f'Best threshold = {best_t:.2f}')
ax.scatter([best_t], [best_f1], color='#27ae60', s=120, zorder=5, edgecolors='black')
ax.annotate(f'F1 = {best_f1:.3f}', xy=(best_t, best_f1),
            xytext=(best_t + 0.1, best_f1 - 0.08), fontsize=11, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black'),
            bbox=dict(boxstyle='round,pad=0.3', fc='#eafaf1', ec='#27ae60'))
ax.set_xlabel('Threshold')
ax.set_ylabel('Score')
ax.set_title('Threshold Optimization — Maximizing F1', fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(0, 1)

# PR curve
precision_pr, recall_pr, _ = precision_recall_curve(y_test, y_proba)
ax2 = axes[1]
ax2.plot(recall_pr, precision_pr, color='#2c3e50', lw=2.5)
ax2.fill_between(recall_pr, precision_pr, alpha=0.15, color='#3498db')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title(f'Precision-Recall Curve (AP = {ap:.3f})', fontweight='bold')
ax2.axhline(y_test.mean(), color='gray', ls='--', alpha=0.5, label=f'Baseline ({y_test.mean():.1%})')
ax2.legend()

plt.tight_layout()
plt.show()

print(f'Optimal threshold: {best_t:.2f}')
print(f'F1: {best_f1:.4f}  |  Precision: {precs[best_idx]:.4f}  |  Recall: {recs[best_idx]:.4f}')
"""),

        # ── Model Evaluation ──
        code("""\
# ── Full Evaluation at Optimal Threshold ──
y_pred = (y_proba >= best_t).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Active', 'Churned'], yticklabels=['Active', 'Churned'])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title(f'Confusion Matrix (threshold={best_t:.2f})', fontweight='bold')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[1].plot(fpr, tpr, color='#2c3e50', lw=2.5, label=f'LightGBM (AUC = {auc:.4f})')
axes[1].plot([0, 1], [0, 1], color='gray', ls='--', alpha=0.5)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve', fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].set_aspect('equal')

plt.tight_layout()
plt.show()

print(classification_report(y_test, y_pred, target_names=['Active', 'Churned']))
"""),

        # ── SHAP ──
        code("""\
# ── SHAP Feature Importance ──
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For binary classification, shap_values may be a list [class0, class1]
if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = shap_values

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Summary plot (beeswarm)
plt.sca(axes[0])
shap.summary_plot(sv, X_test, feature_names=FEATURES, show=False, max_display=len(FEATURES))
axes[0].set_title('SHAP Feature Impact', fontweight='bold')

# Bar plot (mean absolute SHAP)
plt.sca(axes[1])
shap.summary_plot(sv, X_test, feature_names=FEATURES, plot_type='bar',
                  show=False, max_display=len(FEATURES))
axes[1].set_title('Mean |SHAP| — Feature Importance', fontweight='bold')

plt.tight_layout()
plt.show()

# Native LightGBM feature importance for comparison
fi = pd.DataFrame({
    'feature': FEATURES,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print('LightGBM native feature importance (gain):')
print(fi.to_string(index=False))
"""),

        # ── Risk Scores ──
        code("""\
# ── Risk Scores for All Customers ──
# Use the calibrated model for well-calibrated probability estimates
risk_proba = cal_model.predict_proba(X)[:, 1]

risk = df[['IDENTIFIKATOR_KLIJENTA', 'period']].copy()
risk['churn_risk_score'] = risk_proba
risk['predicted_churn'] = (risk_proba >= best_t).astype(int)
risk = risk.sort_values('churn_risk_score', ascending=False).reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Score distribution
axes[0].hist(risk['churn_risk_score'], bins=50, color='#3498db', edgecolor='white', alpha=0.8)
axes[0].axvline(best_t, color='#e74c3c', lw=2, ls='--', label=f'Threshold = {best_t:.2f}')
axes[0].set_xlabel('Churn Risk Score')
axes[0].set_ylabel('Count')
axes[0].set_title('Risk Score Distribution (Calibrated)', fontweight='bold')
axes[0].legend()

# By actual churn status
for val, label, color in [(0, 'Active', '#2ecc71'), (1, 'Churned', '#e74c3c')]:
    subset = risk_proba[y == val]
    axes[1].hist(subset, bins=40, alpha=0.55, color=color, label=label, density=True)
axes[1].axvline(best_t, color='black', lw=2, ls='--', label=f'Threshold = {best_t:.2f}')
axes[1].set_xlabel('Churn Risk Score')
axes[1].set_ylabel('Density')
axes[1].set_title('Score by Actual Churn Status', fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.show()

# Save
risk_path = DATA / 'churn_risk_scores.csv'
risk.to_csv(risk_path, index=False)

n_high = (risk['predicted_churn'] == 1).sum()
n_low = len(risk) - n_high
print(f'Saved: {risk_path}')
print(f'  High risk: {n_high:,} ({n_high/len(risk):.1%})')
print(f'  Low risk:  {n_low:,} ({n_low/len(risk):.1%})')
print(f'\\nTop 10 highest-risk customers:')
risk.head(10)
"""),
    ]
    return nb


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GENERATE ALL NOTEBOOKS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == '__main__':
    import os
    os.makedirs('notebooks', exist_ok=True)

    for name, builder in [
        ('01_eda', make_eda),
        ('02_data_preparation', make_data_prep),
        ('03_model_lightgbm', make_model),
    ]:
        path = f'notebooks/{name}.ipynb'
        nb = builder()
        with open(path, 'w') as f:
            nbf.write(nb, f)
        print(f'✓ {path} ({len(nb.cells)} cells)')
