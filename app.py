# app.py
# Credit Risk Intelligence — polished UI + robust calculations + print-friendly Report Mode
# Requires: streamlit, pandas, numpy, numpy-financial, matplotlib

import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
from pathlib import Path
# import math, datetime ---- if needed later, uncomment these

# Matplotlib (single backend set for entire app)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO

APP_VER = "v1.3.0"

# -------------------------
# Theme / Page
# -------------------------
PRIMARY = "#0E7490"
BG = "#F8FAFC"
SURFACE = "#FFFFFF"
TEXT = "#0F172A"
MUTED = "#6B7280"
BORDER = "#E6E8EB"

# -------------------------
# Branding and author info
# -------------------------
AUTHOR_FULL  = "Aurokrishnaa R L"
AUTHOR_SHORT = "Auro"
AUTHOR_TAGLINE = "MS Finance (Quant) · MBA Finance"

st.set_page_config(page_title="Credit Risk Intelligence", layout="wide")

st.markdown(
    f"""
    <style>
      :root {{
        --primary: {PRIMARY}; --bg: {BG}; --surface: {SURFACE};
        --text: {TEXT}; --muted: {MUTED}; --border: {BORDER};
      }}
      body {{ background: var(--bg); }}
      .block-container {{ max-width: 1200px; padding-top: .6rem; padding-bottom: 2.6rem; }}
      .cr-hero {{ background: linear-gradient(180deg, rgba(14,116,144,.06), rgba(14,116,144,0));
        border:1px solid var(--border); border-radius:16px; padding:22px 24px; margin:0 0 12px 0; text-align:center; }}
      .cr-hero h1 {{ margin:0; font-size:1.9rem; color:var(--text); }}
      .cr-hero p {{ margin:.35rem 0 0 0; color:var(--muted); }}

      /* NEW: subtle byline + stack chips under the hero */
      .cr-hero .cr-byline {{ margin-top:.35rem; color:var(--muted); font-size:.95rem; }}
      .cr-hero .cr-byline b {{ color:var(--text); }}
      .cr-hero .cr-chipstack {{ margin-top:.4rem; }}

      .cr-card {{ border:1px solid var(--border); border-radius:14px; padding:16px 18px; background:var(--surface);
        box-shadow:0 1px 2px rgba(0,0,0,.03); margin-bottom:.75rem; }}
      .cr-ribbon {{ background:#f6f8fb; border:1px solid var(--border); border-radius:12px; padding:10px 14px; margin:.6rem 0 1rem; }}
      .cr-pill {{ display:inline-block; background:var(--surface); border:1px solid #e3e7ee; border-radius:999px; padding:4px 10px; margin:4px 6px 0 0; font-size:.9rem; }}
      .stButton > button, .stDownloadButton > button {{ border-radius:10px!important; border:1px solid var(--border)!important; padding:.5rem .75rem!important; }}
      .stButton > button[kind="primary"] {{ background:var(--primary)!important; color:white!important; border-color:rgba(0,0,0,.06)!important; }}
      [data-testid="stMetric"] {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:8px 12px; box-shadow:0 1px 2px rgba(0,0,0,.03); }}
      .cr-brandbar {{ display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap; background:var(--surface); border:1px solid var(--border);
        border-radius:12px; padding:10px 12px; margin-top:2px; }}
      .cr-footer {{ margin-top:10px; border-top:1px dashed var(--border); padding-top:8px; color:var(--muted); }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="cr-hero">
      <h1>Credit Risk Intelligence Platform</h1>
      <p>Stress, assess, and price credit risk with explainable metrics and reporting.</p>
      <div class="cr-byline">by <b>{AUTHOR_SHORT}</b> <span style="opacity:.6;">•</span> {AUTHOR_TAGLINE}</div>
      <div class="cr-chipstack">
        <span class="cr-pill">Python</span>
        <span class="cr-pill">Pandas</span>
        <span class="cr-pill">NumPy</span>
        <span class="cr-pill">Streamlit</span>
        <span class="cr-pill">CECL / IFRS</span>
        <span class="cr-pill">Stress testing</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Subtle contact chips below hero
LINKEDIN_URL = "https://www.linkedin.com/in/aurokrishnaa"
WEBSITE_URL  = "https://www.aurokrishnaa.me"
RESUME_URL   = globals().get("RESUME_URL", None)  # set to your resume link when ready

chips = f'''
<div style="display:flex; gap:8px; justify-content:center; margin:.25rem 0 .5rem 0; flex-wrap:wrap;">
  <a class="cr-pill" href="{WEBSITE_URL}" target="_blank" rel="noopener"> Website</a>
  <a class="cr-pill" href="{LINKEDIN_URL}" target="_blank" rel="noopener"> LinkedIn</a>
  {f'<a class="cr-pill" href="{RESUME_URL}" target="_blank" rel="noopener"> Résumé</a>' if RESUME_URL else ''}
</div>
'''
st.markdown(chips, unsafe_allow_html=True)

st.session_state.setdefault("disc_rate_pct", 6.00)

# -------------------------
# Sidebar: Scenario / Covenants
# -------------------------
st.sidebar.header("Scenario setup")
preset = st.sidebar.radio("Scenario preset", ["Base", "Adverse", "Severe", "Custom"], index=0)
if preset == "Custom":
    rate_bps = st.sidebar.slider("Rate shock (bps)", -300, 300, 0, step=25)
    unemp_pp = st.sidebar.slider("Unemployment shock (pp)", 0, 10, 0, step=1)
    coll_drop = st.sidebar.slider("Collateral drop (%)", 0, 60, 0, step=5)
else:
    presets = {
        "Base":    {"rate_bps": 0,   "unemp_pp": 0, "coll_drop": 0},
        "Adverse": {"rate_bps": 100, "unemp_pp": 2, "coll_drop": 10},
        "Severe":  {"rate_bps": 250, "unemp_pp": 5, "coll_drop": 25},
    }
    rate_bps  = presets[preset]["rate_bps"]
    unemp_pp  = presets[preset]["unemp_pp"]
    coll_drop = presets[preset]["coll_drop"]

with st.sidebar.expander("Covenant thresholds", expanded=False):
    dscr_min = st.number_input("DSCR >=", value=1.20, step=0.05, format="%.2f")
    icr_min  = st.number_input("ICR >=",  value=2.00, step=0.25, format="%.2f")
    ltv_max  = st.number_input("LTV <=",  value=0.85, step=0.01, format="%.2f")
    dti_max  = st.number_input("DTI <=",  value=0.45, step=0.01, format="%.2f")
    dscr_amber = dscr_min + 0.15; icr_amber = icr_min + 0.25
    ltv_amber  = max(0.0, ltv_max - 0.05); dti_amber = max(0.0, dti_max - 0.05)

st.sidebar.caption("Tip: Presets fill shocks for you. Switch to Custom to fine-tune.")

# Keep scenario label available everywhere (Reports uses this)
st.session_state["preset"] = preset

# -------------------------

ROOT = Path(__file__).parent if "__file__" in globals() else Path.cwd()
DATA_DIR = ROOT / "data"; DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = ROOT / "reports"; REPORT_DIR.mkdir(parents=True, exist_ok=True)

template_path = DATA_DIR / "loan_template.csv"
sample_path   = DATA_DIR / "sample_loans.csv"

# -------------------------
# Helpers (robust & shared)
# -------------------------
def to_decimal_rate(r):
    if pd.isna(r): return np.nan
    return r / 100.0 if r > 1.0 else r

def clean_interest_rate(x):
    """Accepts 0.075, '7.5%', '7,5' → returns decimal rate or NaN."""
    if pd.isna(x): return np.nan
    if isinstance(x, str):
        s = x.strip().replace("%", "").replace(",", ".")
        try: x = float(s)
        except: return np.nan
    try:
        x = float(x)
    except:
        return np.nan
    return to_decimal_rate(x)

def monthly_payment(P, annual_rate, n_months):
    if pd.isna(P) or pd.isna(annual_rate) or pd.isna(n_months) or n_months <= 0: return np.nan
    r = annual_rate / 12.0
    if abs(r) < 1e-12: return float(P / n_months)
    return float(npf.pmt(r, n_months, -P))

def outstanding_balance(P, annual_rate, n_months, k):
    if pd.isna(P) or pd.isna(annual_rate) or pd.isna(n_months): return np.nan
    if k <= 0: return float(P)
    if k >= n_months: return 0.0
    r = annual_rate / 12.0
    if abs(r) < 1e-12: return float(max(P - (P / n_months) * k, 0.0))
    A = monthly_payment(P, annual_rate, n_months)
    return float(P*(1+r)**k - A*((1+r)**k - 1)/r)

def first_year_interest(P, annual_rate, n_months):
    if pd.isna(P) or pd.isna(annual_rate) or pd.isna(n_months) or n_months <= 0: return np.nan
    r = annual_rate / 12.0
    if abs(r) < 1e-12: return 0.0
    A = monthly_payment(P, annual_rate, n_months)
    bal = P; months = int(min(12, n_months)); total_int = 0.0
    for _ in range(months):
        interest = bal * r
        principal = A - interest
        bal = max(bal - principal, 0.0)
        total_int += interest
    return float(total_int)

def choose_collateral_value(row):
    cmv = row.get("CurrentMarketValue", np.nan)
    if pd.notna(cmv):
        try:
            cmv = float(cmv)
            if cmv > 0:
                return cmv
        except:
            pass
    cv = row.get("CollateralValue", np.nan)
    if pd.notna(cv):
        try:
            cv = float(cv)
            return cv if cv > 0 else np.nan
        except:
            return np.nan
    return np.nan

def validate_rows(df):
    problems = []
    for idx, r in df.iterrows():
        row_issues = []
        if pd.isna(r.get("LoanAmount")) or r["LoanAmount"] <= 0: row_issues.append("LoanAmount<=0 or missing")
        if pd.isna(r.get("TermMonths")) or r["TermMonths"] <= 0: row_issues.append("TermMonths<=0 or missing")
        if pd.isna(r.get("InterestRate_clean")) or r["InterestRate_clean"] < 0: row_issues.append("Invalid InterestRate")
        if "CreditScore" in df.columns:
            cs = r.get("CreditScore")
            if pd.notna(cs) and (cs < 300 or cs > 850): row_issues.append("CreditScore out of [300,850]")
        coll = r.get("CollateralBase", np.nan)
        if pd.isna(coll) or coll <= 0: row_issues.append("No collateral value")
        if row_issues:
            problems.append({"LoanID": r.get("LoanID", idx), "Issues": ", ".join(row_issues)})
    return pd.DataFrame(problems)

def pd_scorecard(cs, dti, ltv, sector):
    """
    Heuristic 12-month PD based on credit score + leverage (DTI/LTV) + sector.
    Returns a PD in [0.2%, 35%] for demo purposes.
    """
    if pd.isna(cs): 
        base = 0.03
    elif cs >= 760: 
        base = 0.005
    elif cs >= 720: 
        base = 0.008
    elif cs >= 680: 
        base = 0.015
    elif cs >= 640: 
        base = 0.030
    elif cs >= 600: 
        base = 0.060
    else: 
        base = 0.100

    if pd.isna(dti): 
        dti_f = 1.2
    elif dti <= 0.20: 
        dti_f = 1.0
    elif dti <= 0.35: 
        dti_f = 1.2
    elif dti <= 0.50: 
        dti_f = 1.6
    else: 
        dti_f = 2.2

    if pd.isna(ltv): 
        ltv_f = 1.2
    elif ltv <= 0.70: 
        ltv_f = 1.0
    elif ltv <= 0.85: 
        ltv_f = 1.2
    elif ltv <= 1.00: 
        ltv_f = 1.5
    else: 
        ltv_f = 2.0

    sector = (sector or "").strip().lower()
    sector_factors = {
        "construction": 1.30, "retail": 1.25, "realestate": 1.20, "real estate": 1.20,
        "hospitality": 1.25, "manufacturing": 1.10, "healthcare": 1.10, "transport": 1.15
    }
    sec_f = sector_factors.get(sector, 1.0)
    return float(np.clip(base * dti_f * ltv_f * sec_f, 0.002, 0.35))

def lgd_from_collateral(loan_amount, collateral_base, coll_type):
    if pd.isna(loan_amount) or loan_amount <= 0 or pd.isna(collateral_base) or collateral_base <= 0:
        return 0.60
    ctype = (coll_type or "").strip().lower()
    haircuts = {
        "realestate": 0.30, "real estate": 0.30, "equipment": 0.45, "inventory": 0.60,
        "ar": 0.60, "accounts receivable": 0.60, "securities": 0.10
    }
    h = haircuts.get(ctype, 0.50)
    recovery_cost = 0.10
    effective_collateral = max(collateral_base, 0) * (1.0 - h)
    recovery_rate = min(effective_collateral / loan_amount, 1.0) * (1.0 - recovery_cost)
    return float(np.clip(1.0 - recovery_rate, 0.05, 0.95))

def ead_12m_avg(loan_amount, bal_12m):
    if pd.isna(loan_amount): 
        return np.nan
    if pd.isna(bal_12m): 
        return float(loan_amount)
    return float(0.5 * (loan_amount + max(0.0, bal_12m)))

def _build_template_df():
    # Numeric InterestRate values are intentional (no "%" strings) to avoid parsing errors.
    return pd.DataFrame([
        {
            "LoanID":"L-001","Borrower":"Example Corp","Sector":"Manufacturing","Geography":"NY",
            "LoanAmount":250000,"InterestRate":7.5,"TermMonths":60,"CreditScore":720,
            "MonthlyIncome":18000,"CollateralType":"Equipment","CollateralValue":200000,
            "CurrentMarketValue":190000,"ExistingDTI":0.35,"NOI":"","EBITDA":""
        },
        {
            "LoanID":"L-002","Borrower":"Beacon Realty","Sector":"Real Estate","Geography":"CA",
            "LoanAmount":600000,"InterestRate":6.9,"TermMonths":120,"CreditScore":770,
            "MonthlyIncome":50000,"CollateralType":"RealEstate","CollateralValue":900000,
            "CurrentMarketValue":880000,"ExistingDTI":0.20,"NOI":"","EBITDA":""
        },
        {
            "LoanID":"L-003","Borrower":"Urban Hotels","Sector":"Hospitality","Geography":"NV",
            "LoanAmount":450000,"InterestRate":10.5,"TermMonths":96,"CreditScore":605,
            "MonthlyIncome":24000,"CollateralType":"Real Estate","CollateralValue":420000,
            "CurrentMarketValue":380000,"ExistingDTI":0.50,"NOI":"","EBITDA":""
        }
    ])

# CECL helpers (single source used by both tab & report)
def monthly_default_prob(pd_12m):
    if pd.isna(pd_12m) or pd_12m <= 0: return 0.0
    pd_12m = float(np.clip(pd_12m, 1e-9, 0.99))
    return 1.0 - (1.0 - pd_12m) ** (1.0 / 12.0)

def lifetime_el_row(P, annual_rate, term_m, pd_12m, lgd, disc_rate_annual):
    if pd.isna(P) or pd.isna(annual_rate) or pd.isna(term_m) or term_m <= 0 or pd.isna(pd_12m) or pd.isna(lgd): return np.nan
    P = float(P); annual_rate = float(annual_rate); term_m = int(term_m)
    lgd = float(np.clip(lgd, 0.0, 1.0)); pm = monthly_default_prob(float(pd_12m)); r_m = float(disc_rate_annual)/12.0
    A = monthly_payment(P, annual_rate, term_m); bal = P; survival = 1.0; el_pv = 0.0
    for m in range(1, term_m + 1):
        interest = 0.0 if abs(annual_rate/12.0) < 1e-12 else bal*(annual_rate/12.0)
        principal = A - interest; ead_m = bal; q_m = survival * pm; loss_m = q_m * lgd * ead_m
        el_pv += loss_m / ((1.0 + r_m) ** m) if r_m > -0.9999 else loss_m
        survival *= (1.0 - pm); bal = max(bal - principal, 0.0)
        if bal <= 1e-8: break
    return float(el_pv)

# ---- Percentile-based risk tiering (base PD distribution) ----
def _risk_cuts_from_base(pd_series, min_gap=0.003):
    """
    Compute Low/Med/High cutpoints from BASE PD using ~33/66 percentiles.
    Adds a fallback for tiny samples.
    """
    x = pd.to_numeric(pd_series, errors="coerce").astype(float).clip(lower=1e-5)
    x_nonan = x.dropna()
    if x_nonan.empty or len(x_nonan) < 6:
        return 0.02, 0.06  # stable fallback for small samples
    q_low, q_high = np.nanpercentile(x_nonan, [33, 66])
    if (q_high - q_low) < min_gap:
        mid = (q_low + q_high) / 2
        q_low, q_high = mid - min_gap/2, mid + min_gap/2
    return float(q_low), float(q_high)

def _assign_tiers_from_pd(pd_series, cuts):
    q_low, q_high = cuts
    x = pd.to_numeric(pd_series, errors="coerce").astype(float).clip(lower=1e-5)
    conds = [(x <= q_low), (x > q_low) & (x <= q_high), (x > q_high)]
    return np.select(conds, ["Low", "Medium", "High"], default="Medium")

# (Legacy) threshold-style tiering kept for compatibility
def risk_tier(pd12, ltv, dti, dscr):
    if (pd12 is not None and pd12 >= 0.06) or (ltv is not None and ltv > 0.90) or (dscr is not None and dscr < 1.20) or (dti is not None and dti > 0.45):
        return "High"
    if (pd12 is not None and pd12 >= 0.02) or (ltv is not None and ltv > 0.80) or (dscr is not None and dscr < 1.50) or (dti is not None and dti > 0.35):
        return "Medium"
    return "Low"

def badge_from_covenants(dscr, icr, ltv, dti,
                         dscr_min=1.20, icr_min=2.00, ltv_max=0.85, dti_max=0.45,
                         dscr_amber=1.35, icr_amber=2.25, ltv_amber=0.80, dti_amber=0.40):
    def lt(a, b): return (pd.notna(a) and pd.notna(b) and a < b)
    def gt(a, b): return (pd.notna(a) and pd.notna(b) and a > b)
    if lt(dscr, dscr_min) or (pd.notna(icr) and lt(icr, icr_min)) or gt(ltv, ltv_max) or gt(dti, dti_max):
        return "Red"
    if lt(dscr, dscr_amber) or (pd.notna(icr) and lt(icr, icr_amber)) or gt(ltv, ltv_amber) or gt(dti, dti_amber):
        return "Amber"
    return "Green"

# -------------------------
# Get started
# -------------------------
with st.container():
    st.markdown('<div class="cr-card">', unsafe_allow_html=True)
    st.subheader("Get started")
    st.markdown(
        "Download the CSV template, replace the sample rows (keep the **same columns**), then upload your file."
    )

    # Always build a filled template in memory (and try to refresh the on-disk copy too)
    try:
        tmpl_df = _build_template_df()  # you added this helper earlier
    except NameError:
        # ultra-safe fallback if helper isn't present
        tmpl_df = pd.DataFrame([{
            "LoanID":"L-001","Borrower":"Example Corp","Sector":"Manufacturing","Geography":"NY",
            "LoanAmount":250000,"InterestRate":7.5,"TermMonths":60,"CreditScore":720,
            "MonthlyIncome":18000,"CollateralType":"Equipment","CollateralValue":200000,
            "CurrentMarketValue":190000,"ExistingDTI":0.35,"NOI":"","EBITDA":""
        }])
    try:
        tmpl_df.to_csv(template_path, index=False)  # best-effort refresh
    except Exception:
        pass

    tmpl_csv = tmpl_df.to_csv(index=False).encode("utf-8")

    c0, c1, c2 = st.columns([1, 1, 2])
    with c0:
        st.download_button(
            "Download CSV template",
            data=tmpl_csv,
            file_name="loan_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c1:
        use_sample = st.button("Use sample dataset", use_container_width=True)
    with c2:
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        st.caption(
            "Format: UTF-8 CSV · **Columns must match the template** · "
            "**InterestRate is numeric** — use `0.075` (meaning 7.5%) **or** `7.5` (meaning 7.5%). "
            "Do **not** include a % sign."
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Quick glossary so users know what’s expected
    with st.expander("Column glossary", expanded=False):
        st.markdown(
            "- **LoanAmount** *(USD)* — Original principal.\n"
            "- **InterestRate** — Numeric only. Use `0.075` or `7.5` for 7.5%.\n"
            "- **TermMonths** — Remaining term in months.\n"
            "- **MonthlyIncome** — If **NOI** is not provided, DSCR uses MonthlyIncome×12.\n"
            "- **NOI / EBITDA** *(optional)* — If provided, DSCR/ICR will use them.\n"
            "- **CollateralType / CollateralValue / CurrentMarketValue** — Used for LTV & LGD.\n"
            "- **CreditScore** — 300–850; improves PD calibration.\n"
            "- **Sector / Geography** — For concentration views and sector PD factors."
        )

def _load_sample_df():
    """Balanced sample: Low, Medium, High risk mixes. InterestRate is numeric (no % strings)."""
    df = pd.DataFrame([
        # Low risk (prime, low leverage)
        {"LoanID":"L-001","Borrower":"Alpha Co","Sector":"Manufacturing","Geography":"NY","LoanAmount":250000,"InterestRate":6.5,"TermMonths":60,"CreditScore":780,"MonthlyIncome":22000,"CollateralType":"Equipment","CollateralValue":350000,"CurrentMarketValue":340000,"ExistingDTI":0.18},
        {"LoanID":"L-002","Borrower":"Beacon Realty","Sector":"Real Estate","Geography":"CA","LoanAmount":600000,"InterestRate":6.9,"TermMonths":120,"CreditScore":770,"MonthlyIncome":50000,"CollateralType":"RealEstate","CollateralValue":900000,"CurrentMarketValue":880000,"ExistingDTI":0.20},
        {"LoanID":"L-003","Borrower":"Sovereign Fund","Sector":"Healthcare","Geography":"MA","LoanAmount":180000,"InterestRate":5.8,"TermMonths":48,"CreditScore":765,"MonthlyIncome":20000,"CollateralType":"Securities","CollateralValue":250000,"CurrentMarketValue":245000,"ExistingDTI":0.16},

        # Medium risk
        {"LoanID":"L-004","Borrower":"Gamma Inc","Sector":"Healthcare","Geography":"TX","LoanAmount":120000,"InterestRate":9.9,"TermMonths":36,"CreditScore":680,"MonthlyIncome":12000,"CollateralType":"AR","CollateralValue":90000,"CurrentMarketValue":90000,"ExistingDTI":0.34},
        {"LoanID":"L-005","Borrower":"Delta Retail","Sector":"Retail","Geography":"FL","LoanAmount":200000,"InterestRate":8.2,"TermMonths":72,"CreditScore":700,"MonthlyIncome":16000,"CollateralType":"Inventory","CollateralValue":150000,"CurrentMarketValue":140000,"ExistingDTI":0.35},
        {"LoanID":"L-006","Borrower":"Omega Logistics","Sector":"Transport","Geography":"IL","LoanAmount":320000,"InterestRate":8.2,"TermMonths":84,"CreditScore":710,"MonthlyIncome":30000,"CollateralType":"Equipment","CollateralValue":280000,"CurrentMarketValue":270000,"ExistingDTI":0.33},

        # High risk (subprime / high leverage)
        {"LoanID":"L-007","Borrower":"Beta LLC","Sector":"Construction","Geography":"CA","LoanAmount":500000,"InterestRate":9.5,"TermMonths":84,"CreditScore":640,"MonthlyIncome":35000,"CollateralType":"RealEstate","CollateralValue":600000,"CurrentMarketValue":540000,"ExistingDTI":0.42},
        {"LoanID":"L-008","Borrower":"Urban Hotels","Sector":"Hospitality","Geography":"NV","LoanAmount":450000,"InterestRate":10.5,"TermMonths":96,"CreditScore":605,"MonthlyIncome":24000,"CollateralType":"Real Estate","CollateralValue":420000,"CurrentMarketValue":380000,"ExistingDTI":0.50},
        {"LoanID":"L-009","Borrower":"StartX Labs","Sector":"Technology","Geography":"WA","LoanAmount":150000,"InterestRate":11.5,"TermMonths":36,"CreditScore":590,"MonthlyIncome":10000,"CollateralType":"Inventory","CollateralValue":60000,"CurrentMarketValue":50000,"ExistingDTI":0.55},
        {"LoanID":"L-010","Borrower":"Metro Stores","Sector":"Retail","Geography":"NJ","LoanAmount":260000,"InterestRate":11.0,"TermMonths":60,"CreditScore":610,"MonthlyIncome":13000,"CollateralType":"AR","CollateralValue":120000,"CurrentMarketValue":100000,"ExistingDTI":0.52},
    ])
    df.to_csv(sample_path, index=False)
    return df

# Session state + load logic (unchanged in spirit; safe & simple)
if "loan_df" not in st.session_state:
    st.session_state.loan_df = None

if use_sample:
    st.session_state.loan_df = _load_sample_df()

def _read_csv_safely(file_like) -> pd.DataFrame:
    """
    Tries utf-8, utf-8-sig, latin-1 and auto-detects comma/semicolon tabs.
    Works well with macOS Numbers/Excel exports.
    """
    attempts = [
        {"encoding": "utf-8", "sep": None, "engine": "python"},
        {"encoding": "utf-8-sig", "sep": None, "engine": "python"},
        {"encoding": "latin-1", "sep": None, "engine": "python"},
    ]
    last_err = None
    for kw in attempts:
        try:
            file_like.seek(0)
            return pd.read_csv(file_like, **kw)
        except Exception as e:
            last_err = e
    raise last_err

if uploaded is not None:
    try:
        st.session_state.loan_df = _read_csv_safely(uploaded)
        st.success("File loaded.")
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.info("Tip: Export as UTF-8, comma-delimited. In Numbers: File → Export To → CSV… → Advanced Options → Unicode (UTF-8), comma delimiter.")

df = st.session_state.loan_df
if df is None:
    st.info("Upload a CSV or click Use sample dataset to continue.")
    st.stop()

# --- Normalize Sector/Geography once for consistent filtering ---
df_norm = df.copy()
for col in ("Sector", "Geography"):
    if col not in df_norm.columns:
        df_norm[col] = "Unknown"
    else:
        df_norm[col] = (
            df_norm[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)  # collapse weird whitespace
        )
        # map empties / string-nans to "Unknown"
        df_norm.loc[df_norm[col].str.lower().isin(["", "nan", "none", "null"]), col] = "Unknown"

# -------------------------
# Global filters
# -------------------------
with st.container():
    st.markdown('<div class="cr-card">', unsafe_allow_html=True)
    st.subheader("Global filters")
    c1, c2, c3 = st.columns(3)

    sector_vals = sorted(set(df_norm["Sector"].fillna("Unknown").tolist()))
    geo_vals    = sorted(set(df_norm["Geography"].fillna("Unknown").tolist()))

    with c1:
        sector_pick = st.selectbox("Sector", ["All"] + sector_vals, index=0, key="filter_sector")
    with c2:
        geo_pick    = st.selectbox("Geography", ["All"] + geo_vals, index=0, key="filter_geo")
    with c3:
        tier_pick   = st.selectbox("Risk tier (stressed)", ["All","Low","Medium","High"], index=0, key="filter_tier")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Derived metrics & stress (robust coercions)
# -------------------------
work = df_norm.copy()

# Coerce numerics (safe)
for col in ["LoanAmount","InterestRate","TermMonths","CreditScore","MonthlyIncome","NOI","EBITDA","CollateralValue","CurrentMarketValue"]:
    if col in work.columns:
        work[col] = pd.to_numeric(work[col], errors="coerce")

# Clean rate and collateral, clamp negatives
work["InterestRate_clean"] = work.get("InterestRate", pd.Series(dtype=float)).apply(clean_interest_rate)
work["CollateralBase"] = work.apply(choose_collateral_value, axis=1)
work["CollateralBase"] = pd.to_numeric(work["CollateralBase"], errors="coerce").clip(lower=1e-9)

# Safe months integer for calculations
def _safe_int(x):
    try:
        i = int(float(x))
        return i if i > 0 else np.nan
    except:
        return np.nan

work["TermMonths"] = work.get("TermMonths", pd.Series(dtype=float)).apply(_safe_int)

# Ensure columns exist (df_norm already adds them, but keep extra-safe)
# Ensure columns exist (df_norm already adds them, but keep extra-safe)
if "Sector" not in work.columns: work["Sector"] = "Unknown"
if "Geography" not in work.columns: work["Geography"] = "Unknown"
# NEW: ensure LoanID exists to avoid KeyErrors in views/tables
if "LoanID" not in work.columns:
    work["LoanID"] = [f"L-{i+1:03d}" for i in range(len(work))]
else:
    work["LoanID"] = work["LoanID"].astype(str)


# Use *effective* picks so we don't mutate widget state
sector_eff = sector_pick
geo_eff = geo_pick

# Apply filters
filtered = work
if sector_eff != "All":
    filtered = filtered.loc[filtered["Sector"].fillna("Unknown") == sector_eff]
if geo_eff != "All":
    filtered = filtered.loc[filtered["Geography"].fillna("Unknown") == geo_eff]

# Auto-recover if empty by relaxing the most specific filter
if filtered.empty:
    if geo_eff != "All":
        st.info(f"No loans for Geography='{geo_eff}' with Sector='{sector_eff}'. Showing Geography=All instead.")
        geo_eff = "All"
        filtered = work if sector_eff == "All" else work.loc[work["Sector"].fillna("Unknown") == sector_eff]
    elif sector_eff != "All":
        st.info(f"No loans for Sector='{sector_eff}'. Showing Sector=All instead.")
        sector_eff = "All"
        filtered = work if geo_eff == "All" else work.loc[work["Geography"].fillna("Unknown") == geo_eff]

work = filtered.copy()

@st.cache_data(show_spinner=False)
def _mp_cached(P, r, n): return monthly_payment(P, r, n)

@st.cache_data(show_spinner=False)
def _ob_cached(P, r, n, k): return outstanding_balance(P, r, n, k)

work["MonthlyPayment"] = work.apply(lambda r: _mp_cached(r.get("LoanAmount"), r.get("InterestRate_clean"), r.get("TermMonths")), axis=1)
work["AnnualDebtService"] = work["MonthlyPayment"] * 12.0
work["Balance_12m"] = work.apply(lambda r: _ob_cached(r.get("LoanAmount"), r.get("InterestRate_clean"), r.get("TermMonths"), 12), axis=1)
work["Interest_Y1"] = work.apply(lambda r: first_year_interest(r.get("LoanAmount"), r.get("InterestRate_clean"), r.get("TermMonths")), axis=1)

work["LTV"] = work["LoanAmount"] / work["CollateralBase"].replace({0: np.nan})
work["DTI"] = work["MonthlyPayment"] / work.get("MonthlyIncome", pd.Series(dtype=float)).replace({0: np.nan})
if "NOI" in work.columns:
    work["DSCR"] = work["NOI"] / work["AnnualDebtService"].replace({0: np.nan})
else:
    work["DSCR"] = (work.get("MonthlyIncome", pd.Series(dtype=float)) * 12.0) / work["AnnualDebtService"].replace({0: np.nan})
work["ICR"] = (work.get("EBITDA", pd.Series(dtype=float)) / work["Interest_Y1"].replace({0: np.nan}))

dq = validate_rows(work)

# Base risk
work["PD_12m"]   = work.apply(lambda r: pd_scorecard(r.get("CreditScore"), r.get("DTI"), r.get("LTV"), r.get("Sector")), axis=1)
work["LGD"]      = work.apply(lambda r: lgd_from_collateral(r.get("LoanAmount"), r.get("CollateralBase"), r.get("CollateralType")), axis=1)
work["EAD_12m"]  = work.apply(lambda r: ead_12m_avg(r.get("LoanAmount"), r.get("Balance_12m")), axis=1)
work["EL_12m"]   = work["PD_12m"] * work["LGD"] * work["EAD_12m"]
_cuts_base = _risk_cuts_from_base(work["PD_12m"])
work["RiskTier"] = _assign_tiers_from_pd(work["PD_12m"], _cuts_base)

total_ead_base = float(work["EAD_12m"].sum())
wavg_pd_base   = float((work["PD_12m"] * work["EAD_12m"]).sum() / total_ead_base) if total_ead_base>0 else np.nan
total_el_base  = float(work["EL_12m"].sum())
pct_high_base  = float((work.loc[work["RiskTier"]=="High","EAD_12m"].sum() / total_ead_base)*100.0) if total_ead_base>0 else np.nan

# Stress
work["InterestRate_st"]     = (work["InterestRate_clean"] + (rate_bps/10000.0)).clip(lower=0.0)
work["MonthlyPayment_st"]   = work.apply(lambda r: _mp_cached(r.get("LoanAmount"), r.get("InterestRate_st"), r.get("TermMonths")), axis=1)
work["AnnualDebtService_st"]= work["MonthlyPayment_st"] * 12.0
work["Balance_12m_st"]      = work.apply(lambda r: _ob_cached(r.get("LoanAmount"), r.get("InterestRate_st"), r.get("TermMonths"), 12), axis=1)
work["Interest_Y1_st"]      = work.apply(lambda r: first_year_interest(r.get("LoanAmount"), r.get("InterestRate_st"), r.get("TermMonths")), axis=1)
work["CollateralBase_st"]   = (work["CollateralBase"] * (1.0 - coll_drop/100.0)).clip(lower=1e-9)
work["LTV_st"] = work["LoanAmount"] / work["CollateralBase_st"]
work["DTI_st"] = work["MonthlyPayment_st"] / work.get("MonthlyIncome", pd.Series(dtype=float)).replace({0: np.nan})
if "NOI" in work.columns:
    work["DSCR_st"] = work["NOI"] / work["AnnualDebtService_st"].replace({0: np.nan})
else:
    work["DSCR_st"] = (work.get("MonthlyIncome", pd.Series(dtype=float)) * 12.0) / work["AnnualDebtService_st"].replace({0: np.nan})
work["ICR_st"] = (work.get("EBITDA", pd.Series(dtype=float)) / work["Interest_Y1_st"].replace({0: np.nan}))

work["PD_12m_st_preU"] = work.apply(lambda r: pd_scorecard(r.get("CreditScore"), r.get("DTI_st"), r.get("LTV_st"), r.get("Sector")), axis=1)
unemp_mult = min(1.0 + 0.10 * float(unemp_pp), 2.5)  # demo cap at 2.5x
work["PD_12m_st"] = (work["PD_12m_st_preU"] * unemp_mult).clip(0.002, 0.60)
work["LGD_st"]    = work.apply(lambda r: lgd_from_collateral(r.get("LoanAmount"), r.get("CollateralBase_st"), r.get("CollateralType")), axis=1)
work["EAD_12m_st"]= work.apply(lambda r: ead_12m_avg(r.get("LoanAmount"), r.get("Balance_12m_st")), axis=1)
work["EL_12m_st"] = work["PD_12m_st"] * work["LGD_st"] * work["EAD_12m_st"]
work["RiskTier_st"] = _assign_tiers_from_pd(work["PD_12m_st"], _cuts_base)

# View filter by tier (stressed)
# Guard tier with an effective value (don't mutate widget)
tier_eff = tier_pick
available_tiers = set(work["RiskTier_st"].dropna().unique().tolist())
if tier_eff != "All" and tier_eff not in available_tiers:
    st.info(f"No loans in '{tier_eff}' tier for the current filters. Showing Tier=All instead.")
    tier_eff = "All"

# Build the view safely
work_view = work.copy() if tier_eff == "All" else work.loc[work["RiskTier_st"] == tier_eff].copy()

# Extra guard
if work_view.empty:
    st.warning("No loans remain after applying the current filters.")
    st.stop()

# KPI helper
def _kpis(df_base, df_st):
    total_ead_b = float(df_base["EAD_12m"].sum())
    wavg_pd_b   = float((df_base["PD_12m"] * df_base["EAD_12m"]).sum() / total_ead_b) if total_ead_b > 0 else np.nan
    total_el_b  = float(df_base["EL_12m"].sum())
    pct_high_b  = float((df_base.loc[df_base["RiskTier"]=="High","EAD_12m"].sum() / total_ead_b)*100.0) if total_ead_b>0 else np.nan

    total_ead_s = float(df_st["EAD_12m_st"].sum())
    wavg_pd_s   = float((df_st["PD_12m_st"] * df_st["EAD_12m_st"]).sum() / total_ead_s) if total_ead_s > 0 else np.nan
    total_el_s  = float(df_st["EL_12m_st"].sum())
    pct_high_s  = float((df_st.loc[df_st["RiskTier_st"]=="High","EAD_12m_st"].sum() / total_ead_s)*100.0) if total_ead_s>0 else np.nan
    return (total_ead_b, wavg_pd_b, total_el_b, pct_high_b), (total_ead_s, wavg_pd_s, total_el_s, pct_high_s)

base_kpis_view, stress_kpis_view = _kpis(work_view, work_view)

# Context ribbon
st.markdown('<div class="cr-ribbon">', unsafe_allow_html=True)
st.markdown(
    f"""
    <span class="cr-pill"><b>Scenario:</b> {preset}</span>
    <span class="cr-pill">Rate: {rate_bps} bps</span>
    <span class="cr-pill">Unemployment: +{unemp_pp} pp</span>
    <span class="cr-pill">Collateral: -{coll_drop}%</span>
    <span class="cr-pill"><b>Filters</b> — Sector: {sector_pick} · Geography: {geo_pick} · Tier: {tier_pick}</span>
    <span class="cr-pill">Loans in view: {len(work_view)}</span>
    """,
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# TABS
# -------------------------
tab_overview, tab_data, tab_base, tab_stress, tab_cov, tab_cecl, tab_pricing, tab_reports = st.tabs(
    ["Overview", "Data", "Base Risk", "Stress Test", "Covenants & Watchlist", "CECL / IFRS", "Pricing", "Reports"]
)

# ---------- Overview ----------
with tab_overview:
    st.markdown('<div class="cr-card">', unsafe_allow_html=True)
    st.subheader("Portfolio snapshot")
    st.caption("A single view of exposure, loss, and risk mix—before and after stress.")
    (total_ead_b, wavg_pd_b, total_el_b, pct_high_b) = base_kpis_view
    (total_ead_s, wavg_pd_s, total_el_s, pct_high_s) = stress_kpis_view
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total EAD (12m avg) — Base", f"${total_ead_b:,.0f}")
    k2.metric("W-avg PD (12m) — Base", f"{wavg_pd_b*100:,.2f}%")
    k3.metric("Total EL (12m) — Base", f"${total_el_b:,.0f}")
    k4.metric("High-risk share — Base", f"{pct_high_b:,.1f}%")

    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Total EAD — Stressed", f"${total_ead_s:,.0f}", delta=f"${(total_ead_s-total_ead_b):,.0f}")
    s2.metric("W-avg PD — Stressed", f"{wavg_pd_s*100:,.2f}%", delta=f"{(wavg_pd_s-wavg_pd_b)*100:,.2f} pp")
    s3.metric("Total EL — Stressed", f"${total_el_s:,.0f}", delta=f"${(total_el_s-total_el_b):,.0f}")
    s4.metric("High-risk share — Stressed", f"{pct_high_s:,.1f}%", delta=f"{(pct_high_s-pct_high_b):,.1f} pp")
    st.caption("KPIs reflect current filters and scenario.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="cr-card">', unsafe_allow_html=True)
    st.subheader("Key visuals")

    def _tier_chart_png(df_png, path):
        counts = df_png["RiskTier_st"].value_counts(dropna=False).reindex(["Low","Medium","High"]).fillna(0)
        fig, ax = plt.subplots(figsize=(4.6,3.0), dpi=200)
        counts.plot(kind="bar", ax=ax)
        ax.set_title("Risk Tier (Stressed)")
        ax.set_xlabel("Tier"); ax.set_ylabel("Count")
        mx = max(counts.values) if len(counts.values) else 0
        for i, v in enumerate(counts.values):
            ax.text(i, v + (mx * 0.03 if mx else 0.1), str(int(v)), ha="center", va="bottom", fontsize=8)
        plt.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)

    def _sector_chart_png(df_png, path):
        df2 = df_png.copy(); df2["Sector"] = df2["Sector"].fillna("Unknown")
        agg = df2.groupby("Sector")["EAD_12m_st"].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(5.6,3.0), dpi=200)
        agg.plot(kind="bar", ax=ax)
        ax.set_title("Top Sectors by EAD (Stressed)")
        ax.set_xlabel("Sector"); ax.set_ylabel("EAD (USD)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)

    def _render_chart_to_bytes(fn, df_in, tmpname="tmp_chart.png"):
        """
        Render a chart to bytes without touching the filesystem.
        Works with _tier_chart_png/_sector_chart_png since matplotlib.savefig
        accepts file-like objects (BytesIO).
        """
        buf = BytesIO()
        fn(df_in, buf)  # pass the buffer instead of a path
        data = buf.getvalue()
        buf.close()
        return data

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Risk Tier (Stressed)")
        st.image(_render_chart_to_bytes(_tier_chart_png, work_view, "tmp_tier.png"), use_container_width=True)
    with c2:
        st.caption("Top Sectors by EAD (Stressed)")
        st.image(_render_chart_to_bytes(_sector_chart_png, work_view, "tmp_sector.png"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.info("Demo model: PD/LGD heuristics are for illustration only and are not a production credit policy.")

# ---------- Data ----------
with tab_data:
    st.markdown('<div class="cr-card">', unsafe_allow_html=True)
    st.subheader("Data preview (first 10)")
    disp = work.head(10).copy()
    st.dataframe(disp, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="cr-card">', unsafe_allow_html=True)
    st.subheader("Derived metrics — Base")
    disp = work.copy()

    if "InterestRate_clean" in disp.columns:
        disp["InterestRate_%"] = disp["InterestRate_clean"].astype(float) * 100.0

    money_cols = ["LoanAmount","MonthlyPayment","AnnualDebtService","Balance_12m","Interest_Y1","CollateralBase"]
    ratio_cols = ["LTV","DTI","DSCR","ICR"]
    for c in money_cols:
        if c in disp.columns: disp[c] = disp[c].round(2)
    for c in ratio_cols:
        if c in disp.columns: disp[c] = disp[c].round(4)

    show_cols = [c for c in [
        "LoanID","Borrower","Sector","Geography","LoanAmount",
        "InterestRate_%","TermMonths","MonthlyPayment","AnnualDebtService",
        "Balance_12m","Interest_Y1","LTV","DTI","DSCR","ICR"
    ] if c in disp.columns]

    st.dataframe(
        disp[show_cols], use_container_width=True,
        column_config={
            "LoanAmount": st.column_config.NumberColumn("Loan Amount", format="$%,.2f"),
            "MonthlyPayment": st.column_config.NumberColumn("Monthly Payment", format="$%,.2f"),
            "AnnualDebtService": st.column_config.NumberColumn("Annual Debt Service", format="$%,.2f"),
            "Balance_12m": st.column_config.NumberColumn("Balance @ 12m", format="$%,.2f"),
            "Interest_Y1": st.column_config.NumberColumn("Year-1 Interest", format="$%,.2f"),
            "InterestRate_%": st.column_config.NumberColumn("Interest Rate (%)", format="%.2f"),
            "LTV": st.column_config.NumberColumn("LTV", format="%.2f"),
            "DTI": st.column_config.NumberColumn("DTI", format="%.2f"),
            "DSCR": st.column_config.NumberColumn("DSCR", format="%.2f"),
            "ICR": st.column_config.NumberColumn("ICR", format="%.2f"),
        },
    )

    if not dq.empty:
        st.caption("Rows with data issues (fix in your CSV if needed).")
        st.dataframe(dq, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Base Risk ----------
with tab_base:
    st.markdown('<div class="cr-card">', unsafe_allow_html=True)
    st.subheader("Base risk (12m)")
    st.caption("Ground truth under current conditions with explainable PDs.")
    risk_cols = ["LoanID","PD_12m","LGD","EAD_12m","EL_12m","RiskTier"]
    risk_disp = work[risk_cols].copy()
    risk_disp["PD_12m"] = (risk_disp["PD_12m"] * 100).round(2)
    risk_disp["LGD"]    = (risk_disp["LGD"] * 100).round(2)
    for c in ["EAD_12m","EL_12m"]: risk_disp[c] = risk_disp[c].round(2)
    st.dataframe(
        risk_disp, use_container_width=True,
        column_config={
            "PD_12m": st.column_config.NumberColumn("PD (12m)", format="%.2f %%"),
            "LGD": st.column_config.NumberColumn("LGD", format="%.2f %%"),
            "EAD_12m": st.column_config.NumberColumn("EAD (12m avg)", format="$%,.2f"),
            "EL_12m": st.column_config.NumberColumn("EL (12m)", format="$%,.2f"),
        },
    )
    b1,b2,b3,b4 = st.columns(4)
    b1.metric("Total EAD (12m avg)", f"${total_ead_base:,.0f}")
    b2.metric("W-avg PD (12m)", f"{wavg_pd_base*100:,.2f}%")
    b3.metric("Total EL (12m)", f"${total_el_base:,.0f}")
    b4.metric("High-risk share", f"{pct_high_base:,.1f}%")
    st.caption("Legend - Risk tiers (base): percentile cuts on PD (≈33/66); stable fallback used for small samples.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Stress Test ----------
with tab_stress:
    st.markdown('<div class="cr-card">', unsafe_allow_html=True)
    st.subheader("Stressed risk")
    st.caption("Macro shocks pushed through the cash-flow stack.")
    risk_cols_st = ["LoanID","PD_12m_st","LGD_st","EAD_12m_st","EL_12m_st","RiskTier_st"]
    risk_disp_st = work_view[risk_cols_st].copy()
    risk_disp_st["PD_12m_st"] = (risk_disp_st["PD_12m_st"] * 100).round(2)
    risk_disp_st["LGD_st"]    = (risk_disp_st["LGD_st"] * 100).round(2)
    for c in ["EAD_12m_st","EL_12m_st"]: risk_disp_st[c] = risk_disp_st[c].round(2)
    st.dataframe(
        risk_disp_st, use_container_width=True,
        column_config={
            "PD_12m_st": st.column_config.NumberColumn("PD (12m, st)", format="%.2f %%"),
            "LGD_st": st.column_config.NumberColumn("LGD (st)", format="%.2f %%"),
            "EAD_12m_st": st.column_config.NumberColumn("EAD (12m, st)", format="$%,.2f"),
            "EL_12m_st": st.column_config.NumberColumn("EL (12m, st)", format="$%,.2f"),
        },
    )
    (total_ead_bv, wavg_pd_bv, total_el_bv, pct_high_bv) = base_kpis_view
    (total_ead_sv, wavg_pd_sv, total_el_sv, pct_high_sv) = stress_kpis_view
    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Total EAD — Stressed", f"${total_ead_sv:,.0f}", delta=f"${(total_ead_sv-total_ead_bv):,.0f}")
    s2.metric("W-avg PD — Stressed", f"{wavg_pd_sv*100:,.2f}%", delta=f"{(wavg_pd_sv-wavg_pd_bv)*100:,.2f} pp")
    s3.metric("Total EL — Stressed", f"${total_el_sv:,.0f}", delta=f"${(total_el_sv-total_el_bv):,.0f}")
    s4.metric("High-risk share — Stressed", f"{pct_high_sv:,.1f}%", delta=f"{(pct_high_sv-pct_high_bv):,.1f} pp")
    st.caption("Unemployment shock multiplier capped at 2.5× for demo stability.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Covenants & Watchlist ----------
with tab_cov:
    st.markdown('<div class="cr-card">', unsafe_allow_html=True)
    st.subheader("Covenants & watchlist")
    work_view["HR_DSCR_st"] = work_view["DSCR_st"] - dscr_min
    work_view["HR_ICR_st"]  = work_view["ICR_st"] - icr_min
    work_view["HR_LTV_st"]  = ltv_max - work_view["LTV_st"]
    work_view["HR_DTI_st"]  = dti_max - work_view["DTI_st"]
    work_view["Badge_st"] = work_view.apply(lambda r: badge_from_covenants(
        r.get("DSCR_st"), r.get("ICR_st"), r.get("LTV_st"), r.get("DTI_st"),
        dscr_min, icr_min, ltv_max, dti_max, dscr_amber, icr_amber, ltv_amber, dti_amber
    ), axis=1)

    hr_cols = ["HR_DSCR_st","HR_ICR_st","HR_LTV_st","HR_DTI_st"]
    work_view["WorstHeadroom_st"] = pd.DataFrame({c: work_view[c] for c in hr_cols}).min(axis=1, skipna=True)

    cov_cols = ["LoanID","Badge_st","DSCR_st","ICR_st","LTV_st","DTI_st","HR_DSCR_st","HR_ICR_st","HR_LTV_st","HR_DTI_st","WorstHeadroom_st"]
    cov_disp = work_view[cov_cols].copy()
    for c in ["DSCR_st","ICR_st","LTV_st","DTI_st","HR_DSCR_st","HR_ICR_st","HR_LTV_st","HR_DTI_st","WorstHeadroom_st"]:
        if c in cov_disp.columns: cov_disp[c] = cov_disp[c].astype(float).round(4)
    st.dataframe(cov_disp, use_container_width=True)
    st.caption("Badges: Red = breach; Amber = near; Green = healthy.")

    by_badge = work_view["Badge_st"].value_counts(dropna=False)
    roll_df = pd.DataFrame({"Badge": by_badge.index, "Count": by_badge.values,
                            "Percent": (by_badge.values/ max(len(work_view),1) * 100.0).round(1)})
    st.dataframe(roll_df, use_container_width=True)

    watch = work_view.loc[work_view["Badge_st"].isin(["Red","Amber"])].copy().sort_values(
        ["Badge_st","WorstHeadroom_st"], ascending=[True, True]).head(10)
    wl_cols = [c for c in ["LoanID","Badge_st","WorstHeadroom_st","DSCR_st","LTV_st","DTI_st","PD_12m_st","LGD_st","EL_12m_st"] if c in watch.columns]
    wl_disp = watch[wl_cols].copy()
    if "PD_12m_st" in wl_disp.columns: wl_disp["PD_12m_st"] = (wl_disp["PD_12m_st"]*100).round(2)
    if "LGD_st" in wl_disp.columns:    wl_disp["LGD_st"]    = (wl_disp["LGD_st"]*100).round(2)
    for c in ["WorstHeadroom_st","DSCR_st","LTV_st","DTI_st","EL_12m_st"]:
        if c in wl_disp.columns: wl_disp[c] = wl_disp[c].astype(float).round(4)
    st.dataframe(wl_disp, use_container_width=True)

    # export exceptions
    if not watch.empty:
        exc_cols = [c for c in ["LoanID","Borrower","Sector","Geography","DSCR_st","ICR_st","LTV_st","DTI_st","Badge_st","WorstHeadroom_st"] if c in work_view.columns]
        exc_csv = work_view.loc[work_view["Badge_st"].isin(["Red","Amber"]), exc_cols].to_csv(index=False).encode("utf-8")
        st.download_button("Download Exceptions (CSV)", data=exc_csv, file_name="exceptions_watchlist.csv", mime="text/csv", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- CECL / IFRS ----------
with tab_cecl:
    st.markdown('<div class="cr-card">', unsafe_allow_html=True)
    st.subheader("Lifetime expected loss (CECL/IFRS)")
    st.caption("Lifetime EL via amortization curve and monthly default intensity.")
    with st.expander("Options", expanded=False):
        horizon = st.radio("Loss horizon", ["12-month", "Lifetime"], index=0, horizontal=True)
        disc_rate_pct = st.number_input("Discount rate for lifetime EL (%)", value=st.session_state["disc_rate_pct"], step=0.25, format="%.2f")
        st.session_state["disc_rate_pct"] = float(disc_rate_pct)
    discount_rate = float(st.session_state["disc_rate_pct"]) / 100.0

    @st.cache_data(show_spinner=False)
    def _life_cached(P, r, n, pd12, lgd, dr): return lifetime_el_row(P, r, n, pd12, lgd, dr)

    work_view["EL_life_base"] = work_view.apply(lambda r: _life_cached(r.get("LoanAmount"), r.get("InterestRate_clean"), r.get("TermMonths"), r.get("PD_12m"), r.get("LGD"), discount_rate), axis=1)
    work_view["EL_life_st"]   = work_view.apply(lambda r: _life_cached(r.get("LoanAmount"), r.get("InterestRate_st"),   r.get("TermMonths"), r.get("PD_12m_st"), r.get("LGD_st"), discount_rate), axis=1)

    total_el_life_base = float(np.nansum(work_view["EL_life_base"]))
    total_el_life_st   = float(np.nansum(work_view["EL_life_st"]))
    delta_life         = total_el_life_st - total_el_life_base
    total_ead_start    = float(np.nansum(work_view["LoanAmount"]))
    pct_life_base      = (total_el_life_base/total_ead_start*100.0) if total_ead_start>0 else np.nan
    pct_life_st        = (total_el_life_st/total_ead_start*100.0) if total_ead_start>0 else np.nan
    delta_pct          = (pct_life_st - pct_life_base) if np.isfinite(pct_life_base) and np.isfinite(pct_life_st) else np.nan

    lt_disp = work_view[["LoanID","EL_life_base","EL_life_st"]].copy()
    for c in ["EL_life_base","EL_life_st"]: lt_disp[c] = lt_disp[c].astype(float).round(2)
    st.dataframe(lt_disp, use_container_width=True)

    l1,l2,l3,l4 = st.columns(4)
    l1.metric("Total EL - Lifetime (Base, PV)", f"${total_el_life_base:,.2f}")
    l2.metric("Total EL - Lifetime (Stressed, PV)", f"${total_el_life_st:,.2f}", delta=f"${delta_life:,.2f}")
    l3.metric("Lifetime EL % of Start EAD - Base", f"{pct_life_base:,.2f}%")
    l4.metric("Lifetime EL % of Start EAD - Stressed", f"{pct_life_st:,.2f}%", delta=f"{delta_pct:,.2f} pp")
    st.caption("Method: amortization-based monthly default model with PV discount. Demo parameters only.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Pricing ----------
with tab_pricing:
    st.markdown('<div class="cr-card">', unsafe_allow_html=True)
    st.subheader("Risk based pricing")
    st.caption("Translate risk into economics: spread, all-in, and portfolio targets.")
    with st.expander("Assumptions", expanded=False):
        st.caption("Simple add-ons (per year) applied to EL to produce a suggested spread.")
        cap_buffer_pct    = st.number_input("Capital buffer (% of EAD)", value=1.00, step=0.10, format="%.2f")
        opex_pct          = st.number_input("Operating cost (% of EAD)", value=0.50, step=0.10, format="%.2f")
        target_profit_pct = st.number_input("Target profit (% of EAD)", value=0.50, step=0.10, format="%.2f")
        funding_cost_pct  = st.number_input("Funding cost (% per annum)", value=3.00, step=0.10, format="%.2f")

    work_view["EL_rate_st"] = np.where(work_view["EAD_12m_st"]>0, work_view["EL_12m_st"]/work_view["EAD_12m_st"], np.nan).astype(float)
    add_ons_pct = (cap_buffer_pct + opex_pct + target_profit_pct) / 100.0
    work_view["SuggestedSpread_pct"]  = (work_view["EL_rate_st"] + add_ons_pct)
    work_view["SuggestedSpread_bps"]  = (work_view["SuggestedSpread_pct"] * 10000).round(0)
    work_view["SuggestedAllInRate_pct"] = (funding_cost_pct/100.0 + work_view["SuggestedSpread_pct"]) * 100.0

    total_ead_st = float(work_view["EAD_12m_st"].sum())
    total_el_st  = float(work_view["EL_12m_st"].sum())
    portfolio_el_rate = (total_el_st/total_ead_st) if total_ead_st>0 else np.nan
    portfolio_spread_pct = (portfolio_el_rate*100.0 if np.isfinite(portfolio_el_rate) else np.nan) + cap_buffer_pct + opex_pct + target_profit_pct
    portfolio_spread_bps = portfolio_spread_pct * 100.0 if np.isfinite(portfolio_spread_pct) else np.nan
    portfolio_allin_pct  = funding_cost_pct + portfolio_spread_pct if np.isfinite(portfolio_spread_pct) else np.nan

    p1,p2,p3 = st.columns(3)
    p1.metric("Portfolio EL rate (12m)", f"{(portfolio_el_rate*100.0):,.2f}%")
    p2.metric("Suggested spread (portfolio)", f"{portfolio_spread_bps:,.0f} bps")
    p3.metric("Suggested all-in rate", f"{portfolio_allin_pct:,.2f}%")

    price_cols = [c for c in ["LoanID","PD_12m_st","LGD_st","EAD_12m_st","EL_12m_st","EL_rate_st","SuggestedSpread_bps","SuggestedAllInRate_pct"] if c in work_view.columns]
    price_disp = work_view[price_cols].copy()
    for pct_col in ["PD_12m_st","LGD_st","EL_rate_st","SuggestedAllInRate_pct"]:
        if pct_col in price_disp.columns:
            if pct_col == "SuggestedAllInRate_pct": price_disp[pct_col] = price_disp[pct_col].astype(float).round(2)
            else: price_disp[pct_col] = (price_disp[pct_col].astype(float)*100.0).round(2)
    for money_col in ["EAD_12m_st","EL_12m_st"]:
        if money_col in price_disp.columns: price_disp[money_col] = price_disp[money_col].astype(float).round(2)
    if "SuggestedSpread_bps" in price_disp.columns:
        price_disp["SuggestedSpread_bps"] = price_disp["SuggestedSpread_bps"].astype(float).round(0)

    st.dataframe(
        price_disp, use_container_width=True,
        column_config={
            "PD_12m_st": st.column_config.NumberColumn("PD (12m, st)", format="%.2f %%"),
            "LGD_st": st.column_config.NumberColumn("LGD (st)", format="%.2f %%"),
            "EAD_12m_st": st.column_config.NumberColumn("EAD (12m, st)", format="$%,.2f"),
            "EL_12m_st": st.column_config.NumberColumn("EL (12m, st)", format="$%,.2f"),
            "EL_rate_st": st.column_config.NumberColumn("EL rate (st)", format="%.2f %%"),
            "SuggestedSpread_bps": st.column_config.NumberColumn("Suggested spread (bps)", format="%,.0f"),
            "SuggestedAllInRate_pct": st.column_config.NumberColumn("All-in rate (%)", format="%.2f"),
        },
    )
    pricing_csv = price_disp.to_csv(index=False).encode("utf-8")
    ts_now = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    st.download_button("Download Pricing Table (CSV)", data=pricing_csv, file_name=f"pricing_table_{preset}_{ts_now}.csv", mime="text/csv", use_container_width=True)
    st.caption("EL% + (capital + opex + profit) → spread; all-in = funding + spread. Demo math for illustration.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Reports (Showcase Report Mode — Polished) ----------
with tab_reports:
    # ----------- Print-friendly CSS -----------
    st.markdown(
        """
        <style>
        .report-card { border:1px solid var(--border); border-radius:14px; padding:16px 18px; background:var(--surface); box-shadow:0 1px 2px rgba(0,0,0,.03); }
        .report-h1 { font-size:1.7rem; margin:0 0 .25rem 0; }
        .report-sub { color:var(--muted); margin:.2rem 0 .8rem 0; }
        .report-kpi-grid { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:10px; margin:.6rem 0 .6rem 0; }
        .report-kpi { border:1px solid var(--border); border-radius:12px; padding:10px 12px; background:var(--surface); }
        .report-kpi b { display:block; margin-bottom:4px; }
        .delta { display:block; font-size:.9rem; color:#334155; margin-top:2px; }
        .report-section { margin: 16px 0 8px 0; border-top:1px dashed var(--border); padding-top:12px; }
        .report-note { color:var(--muted); font-size:.92rem; }
        .toc { background:#f6f8fb; border:1px solid var(--border); border-radius:10px; padding:10px 12px; margin:.6rem 0 1rem; }
        .badge { display:inline-block; border:1px solid var(--border); border-radius:999px; padding:2px 10px; margin-left:6px; font-size:.85rem; }
        .badge-base { background:#eef6ff; }
        .badge-adverse { background:#fff7ed; }
        .badge-severe { background:#fef2f2; }
        .badge-custom { background:#f5f5f5; }
        .chip { display:inline-block; border-radius:999px; padding:2px 8px; margin:0 4px; font-size:.8rem; }
        .chip-green { background:#eafff0; border:1px solid #c5f2d6; }
        .chip-amber { background:#fff7d6; border:1px solid #ffe8a3; }
        .chip-red { background:#ffecec; border:1px solid #ffc9c9; }
        .callout { background:#fbfbfd; border:1px solid var(--border); border-radius:10px; padding:8px 10px; margin-top:6px; }
        .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
        @media print {
          [data-testid="stSidebar"], .stButton, .stDownloadButton, .stFileUploader, header, footer { display: none !important; }
          .block-container { max-width: 100% !important; }
          .report-kpi-grid { grid-template-columns: repeat(3, minmax(0,1fr)); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Report Mode (Showcase)")

    st.markdown(
        '<div style="text-align:right; margin-top:-8px; margin-bottom:6px;">'
        '<a class="cr-pill" href="javascript:window.print()">🧾 Print / Save as PDF</a>'
        '</div>',
        unsafe_allow_html=True
    )

    # ----------- Local helpers (display)
    def _mask_pii(df_in: pd.DataFrame, do_mask: bool) -> pd.DataFrame:
        df = df_in.copy()
        if not do_mask:
            return df
        if "LoanID" in df.columns:
            tail = df["LoanID"].astype(str).str[-4:].fillna("")
            df["LoanID"] = "****" + tail
        if "Borrower" in df.columns:
            df["Borrower"] = df["Borrower"].astype(str).str.replace(r"(.)(.*)", r"\1***", regex=True)
        return df

    def _fmt_money(x):
        try: return f"${float(x):,.0f}"
        except: return "-"

    def _fmt_money2(x):
        try: return f"${float(x):,.2f}"
        except: return "-"

    def _fmt_pct(x, dp=2):
        try: return f"{float(x)*100:.{dp}f}%"
        except: return "-"

    def _safe_pp(x):
        try:
            val = float(x)
            # If it's a decimal (e.g., 0.073), convert to pp; if already pp (e.g., 7.3), leave it.
            if abs(val) <= 1.0:
                val *= 100.0
            return f"{val:+.2f} pp"
        except:
            return "—"

    def _safe_dollar(x):
        try:
            return f"{float(x):+,.0f}"
        except:
            return "—"

    def _img_from_fig(fig):
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=240, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    def _tier_chart(df_png, col="RiskTier_st", title="Risk Tier (Stressed)"):
        counts = df_png[col].value_counts(dropna=False).reindex(["Low","Medium","High"]).fillna(0)
        fig, ax = plt.subplots(figsize=(4.8,3.0), dpi=240)
        counts.plot(kind="bar", ax=ax)
        ax.set_title(title); ax.set_xlabel("Tier"); ax.set_ylabel("Count")
        mx = counts.max() if len(counts) else 0
        for i, v in enumerate(counts.values):
            ax.text(i, v + (mx * 0.03 if mx else 0.1), str(int(v)), ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        return _img_from_fig(fig)

    def _sector_chart(df_png):
        df2 = df_png.copy(); df2["Sector"] = df2["Sector"].fillna("Unknown")
        agg = df2.groupby("Sector")["EAD_12m_st"].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(5.8,3.2), dpi=240)
        agg.plot(kind="bar", ax=ax)
        ax.set_title("Top Sectors by EAD (Stressed)"); ax.set_xlabel("Sector"); ax.set_ylabel("EAD (USD)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return _img_from_fig(fig)

    def _geo_chart(df_png):
        if "Geography" not in df_png.columns: return None
        agg = df_png.groupby("Geography")["EAD_12m_st"].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(5.8,3.2), dpi=240)
        agg.plot(kind="bar", ax=ax)
        ax.set_title("Top Geographies by EAD (Stressed)"); ax.set_xlabel("Geography"); ax.set_ylabel("EAD (USD)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return _img_from_fig(fig)

    # Local covenant calc
    def _calc_covenants(df_src: pd.DataFrame):
        df = df_src.copy()
        df["HR_DSCR_st"] = df.get("DSCR_st", np.nan) - dscr_min
        df["HR_ICR_st"]  = df.get("ICR_st", np.nan) - icr_min
        df["HR_LTV_st"]  = ltv_max - df.get("LTV_st", np.nan)
        df["HR_DTI_st"]  = dti_max - df.get("DTI_st", np.nan)

        def _badge(dscr, icr, ltv, dti):
            return badge_from_covenants(dscr, icr, ltv, dti, dscr_min, icr_min, ltv_max, dti_max, dscr_amber, icr_amber, ltv_amber, dti_amber)

        df["Badge_st"] = df.apply(lambda r: _badge(r.get("DSCR_st"), r.get("ICR_st"), r.get("LTV_st"), r.get("DTI_st")), axis=1)

        def _tightest_label(row):
            scores = {"DSCR": row.get("HR_DSCR_st", np.nan),
                      "ICR":  row.get("HR_ICR_st", np.nan),
                      "LTV":  row.get("HR_LTV_st", np.nan),
                      "DTI":  row.get("HR_DTI_st", np.nan)}
            try:
                k = min((k for k,v in scores.items() if pd.notna(v)), key=lambda k: scores[k])
                return k
            except ValueError:
                return np.nan

        df["Tightest"] = df.apply(_tightest_label, axis=1)

        # Compute WorstHeadroom_st (min across covenant headrooms), matching tab_cov logic
        hr_cols_present = [c for c in ["HR_DSCR_st","HR_ICR_st","HR_LTV_st","HR_DTI_st"] if c in df.columns]
        if hr_cols_present:
            df["WorstHeadroom_st"] = pd.DataFrame({c: df[c] for c in hr_cols_present}).min(axis=1, skipna=True)
        else:
            df["WorstHeadroom_st"] = np.nan

        return df

    # Local pricing calc
    def _calc_pricing(df_src: pd.DataFrame, cap_buffer_pct_val, opex_pct_val, target_profit_pct_val, funding_cost_pct_val):
        df = df_src.copy()
        df["EL_rate_st"] = np.where(df["EAD_12m_st"]>0, df["EL_12m_st"]/df["EAD_12m_st"], np.nan).astype(float)
        add_ons_pct = (cap_buffer_pct_val + opex_pct_val + target_profit_pct_val) / 100.0
        df["SuggestedSpread_pct"]  = df["EL_rate_st"] + add_ons_pct
        df["SuggestedSpread_bps"]  = df["SuggestedSpread_pct"] * 10000
        df["SuggestedAllInRate_pct"] = (funding_cost_pct_val/100.0 + df["SuggestedSpread_pct"]) * 100.0

        total_ead_st_local = float(df["EAD_12m_st"].sum())
        total_el_st_local  = float(df["EL_12m_st"].sum())
        portfolio_el_rate_local = (total_el_st_local/total_ead_st_local) if total_ead_st_local>0 else np.nan
        portfolio_spread_pct_local = (portfolio_el_rate_local*100.0 if np.isfinite(portfolio_el_rate_local) else np.nan) + cap_buffer_pct_val + opex_pct_val + target_profit_pct_val
        portfolio_spread_bps_local = portfolio_spread_pct_local * 100.0 if np.isfinite(portfolio_spread_pct_local) else np.nan
        portfolio_allin_pct_local  = funding_cost_pct_val + portfolio_spread_pct_local if np.isfinite(portfolio_spread_pct_local) else np.nan

        return df, portfolio_el_rate_local, portfolio_spread_bps_local, portfolio_allin_pct_local

    # Lifetime EL (CECL) summary shared with CECL tab
    def _cecl_totals(df_src: pd.DataFrame, discount_rate_val: float):
        if "EL_life_base" in df_src.columns and "EL_life_st" in df_src.columns:
            base_total = float(np.nansum(df_src["EL_life_base"]))
            stress_total = float(np.nansum(df_src["EL_life_st"]))
        else:
            base_total = float(np.nansum(df_src.apply(
                lambda r: lifetime_el_row(r.get("LoanAmount"), r.get("InterestRate_clean"), r.get("TermMonths"), r.get("PD_12m"), r.get("LGD"), discount_rate_val), axis=1
            )))
            stress_total = float(np.nansum(df_src.apply(
                lambda r: lifetime_el_row(r.get("LoanAmount"), r.get("InterestRate_st"), r.get("TermMonths"), r.get("PD_12m_st"), r.get("LGD_st"), discount_rate_val), axis=1
            )))
        start_ead = float(np.nansum(df_src["LoanAmount"])) if "LoanAmount" in df_src.columns else np.nan
        pct_b = (base_total / start_ead * 100.0) if start_ead > 0 else np.nan
        pct_s = (stress_total / start_ead * 100.0) if start_ead > 0 else np.nan
        return base_total, stress_total, pct_b, pct_s

    # Build snapshot
    def _build_snapshot(include_appendix_val: bool, mask_pii_val: bool, price_params: dict, recruiter_mode_val: bool):
        df_view = _mask_pii(work_view, mask_pii_val)

        (total_ead_b, wavg_pd_b, total_el_b, pct_high_b) = base_kpis_view
        (total_ead_s, wavg_pd_s, total_el_s, pct_high_s) = stress_kpis_view

        delta_ead = (total_ead_s - total_ead_b) if np.isfinite(total_ead_b) and np.isfinite(total_ead_s) else np.nan
        delta_pd  = (wavg_pd_s - wavg_pd_b) if np.isfinite(wavg_pd_b) and np.isfinite(wavg_pd_s) else np.nan
        delta_el  = (total_el_s - total_el_b) if np.isfinite(total_el_b) and np.isfinite(total_el_s) else np.nan
        delta_hr  = (pct_high_s - pct_high_b) if np.isfinite(pct_high_b) and np.isfinite(pct_high_s) else np.nan

        discount_rate_val = float(st.session_state.get("disc_rate_pct", 6.0)) / 100.0
        el_life_b, el_life_s, pct_life_b, pct_life_s = _cecl_totals(df_view, discount_rate_val)

        img_tier_st = img_tier_base = img_sector = img_geo = None
        try: img_tier_st = _tier_chart(df_view, col="RiskTier_st", title="Risk Tier (Stressed)")
        except Exception: pass
        try:
            src_for_base = df_view if "RiskTier" in df_view.columns else work
            img_tier_base = _tier_chart(src_for_base, col="RiskTier", title="Risk Tier (Base)")
        except Exception: pass
        try: img_sector = _sector_chart(df_view)
        except Exception: pass
        try: img_geo = _geo_chart(df_view)
        except Exception: pass

        top3_share = np.nan
        try:
            if "Sector" in df_view.columns and "EAD_12m_st" in df_view.columns and len(df_view):
                sec = df_view.copy(); sec["Sector"] = sec["Sector"].fillna("Unknown")
                total = float(sec["EAD_12m_st"].sum())
                top3 = float(sec.groupby("Sector")["EAD_12m_st"].sum().sort_values(ascending=False).head(3).sum())
                top3_share = (top3/total*100.0) if total>0 else np.nan
        except Exception:
            top3_share = np.nan

        core_cols = [c for c in ["LoanID","Borrower","Sector","Geography","EAD_12m_st","EL_12m_st","PD_12m_st","LGD_st","RiskTier_st"] if c in df_view.columns]
        top_ead = pd.DataFrame(); top_el = pd.DataFrame()
        if "EAD_12m_st" in df_view.columns and core_cols:
            top_ead = df_view[core_cols].copy().sort_values("EAD_12m_st", ascending=False).head(10)
        if "EL_12m_st" in df_view.columns and core_cols:
            top_el = df_view[core_cols].copy().sort_values("EL_12m_st", ascending=False).head(10)
        for _table in [top_ead, top_el]:
            if "PD_12m_st" in _table.columns: _table["PD_12m_st"] = (pd.to_numeric(_table["PD_12m_st"], errors="coerce")*100).round(2)
            if "LGD_st" in _table.columns:    _table["LGD_st"]    = (pd.to_numeric(_table["LGD_st"], errors="coerce")*100).round(2)
            for c in ["EAD_12m_st","EL_12m_st"]:
                if c in _table.columns: _table[c] = pd.to_numeric(_table[c], errors="coerce").round(2)

        cov_df = _calc_covenants(df_view)
        badge_counts = cov_df["Badge_st"].value_counts(dropna=False) if "Badge_st" in cov_df.columns else pd.Series(dtype=int)
        watch_cols = [c for c in ["LoanID","Badge_st","Tightest","WorstHeadroom_st","DSCR_st","LTV_st","DTI_st","PD_12m_st","LGD_st"] if c in cov_df.columns]
        watch = pd.DataFrame()
        if "Badge_st" in cov_df.columns and watch_cols:
            watch = cov_df.loc[cov_df["Badge_st"].isin(["Red","Amber"]), watch_cols].copy()
            if not watch.empty:
                for c in ["WorstHeadroom_st","DSCR_st","LTV_st","DTI_st","PD_12m_st","LGD_st"]:
                    if c in watch.columns: watch[c] = pd.to_numeric(watch[c], errors="coerce").round(4)
                watch = watch.sort_values(["Badge_st","WorstHeadroom_st"], ascending=[True, True]).head(10)

        price_df, portfolio_el_rate, portfolio_spread_bps, portfolio_allin_pct = _calc_pricing(
            df_view,
            cap_buffer_pct_val=price_params["cap_buffer_pct"],
            opex_pct_val=price_params["opex_pct"],
            target_profit_pct_val=price_params["target_profit_pct"],
            funding_cost_pct_val=price_params["funding_cost_pct"],
        )
        price_top = price_df.copy()
        if "SuggestedSpread_bps" in price_top.columns:
            price_top = price_top.sort_values("SuggestedSpread_bps", ascending=False).head(10)
        price_cols = [c for c in ["LoanID","PD_12m_st","LGD_st","EAD_12m_st","EL_12m_st","EL_rate_st","SuggestedSpread_bps","SuggestedAllInRate_pct"] if c in price_top.columns]
        price_top = price_top[price_cols].copy()
        if "PD_12m_st" in price_top.columns: price_top["PD_12m_st"] = (pd.to_numeric(price_top["PD_12m_st"], errors="coerce")*100).round(2)
        if "LGD_st" in price_top.columns:    price_top["LGD_st"]    = (pd.to_numeric(price_top["LGD_st"], errors="coerce")*100).round(2)
        if "EL_rate_st" in price_top.columns: price_top["EL_rate_st"] = (pd.to_numeric(price_top["EL_rate_st"], errors="coerce")*100).round(2)
        if "SuggestedAllInRate_pct" in price_top.columns: price_top["SuggestedAllInRate_pct"] = pd.to_numeric(price_top["SuggestedAllInRate_pct"], errors="coerce").round(2)
        if "SuggestedSpread_bps" in price_top.columns: price_top["SuggestedSpread_bps"] = pd.to_numeric(price_top["SuggestedSpread_bps"], errors="coerce").round(0)
        for c in ["EAD_12m_st","EL_12m_st"]:
            if c in price_top.columns: price_top[c] = pd.to_numeric(price_top[c], errors="coerce").round(2)

        dq_summary = dq.copy().head(10) if isinstance(dq, pd.DataFrame) and not dq.empty else None

        scen = st.session_state.get("preset", preset)
        scen_cls = {"Base":"badge-base","Adverse":"badge-adverse","Severe":"badge-severe"}.get(scen, "badge-custom")

        meta = {
            "Title": "Credit Risk Intelligence Platform — Portfolio Risk Report",
            "Author": "Aurokrishnaa R L — MS Finance (Quant) · MBA Finance",
            "Generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Scenario": scen,
            "ScenarioClass": scen_cls,
            "Shocks": f"Rate {rate_bps} bps, Unemp +{unemp_pp} pp, Collateral −{coll_drop}%",
            "Filters": f"Sector={sector_pick}, Geography={geo_pick}, Tier={tier_pick}",
            "Discount rate (CECL)": f"{st.session_state.get('disc_rate_pct', 6.0):.2f}%",
            "Pricing assumptions": (
                f"Cap buffer {price_params['cap_buffer_pct']:.2f}%, "
                f"Opex {price_params['opex_pct']:.2f}%, "
                f"Profit {price_params['target_profit_pct']:.2f}%, "
                f"Funding {price_params['funding_cost_pct']:.2f}%"
            ),
            "cap_buffer_pct": price_params['cap_buffer_pct'],
            "opex_pct": price_params['opex_pct'],
            "target_profit_pct": price_params['target_profit_pct'],
            "funding_cost_pct": price_params['funding_cost_pct'],
            "RecruiterMode": recruiter_mode_val,
            "SnapshotID": pd.Timestamp.now().strftime("RPT-%Y%m%d-%H%M"),
            "Version": APP_VER,
        }

        return {
            "meta": meta,
            "kpis": {
                "base":  {"EAD": total_ead_b, "PD": wavg_pd_b, "EL": total_el_b, "HR": pct_high_b},
                "stress":{"EAD": total_ead_s, "PD": wavg_pd_s, "EL": total_el_s, "HR": pct_high_s},
                "delta": {"EAD": delta_ead, "PD": delta_pd, "EL": delta_el, "HR": delta_hr},
                "cecl":  {"EL_base": el_life_b, "EL_st": el_life_s, "pct_b": pct_life_b, "pct_s": pct_life_s},
            },
            "imgs": {"tier_base": img_tier_base, "tier_stress": img_tier_st, "sector": img_sector, "geo": img_geo},
            "top3_share": top3_share,
            "top_ead": top_ead,
            "top_el": top_el,
            "cov": {"badge_counts": badge_counts, "watch": watch},
            "pricing": {"top": price_top, "el_rate": portfolio_el_rate, "spread_bps": portfolio_spread_bps, "allin_pct": portfolio_allin_pct},
            "dq": dq_summary,
            "appendix": {},
            "mask_pii": mask_pii_val,
            "include_appendix": False,  # keep simple by default in Showcase
        }

    # ----------- Controls -----------
    colA, colB, colC, colD = st.columns([1,1,1,2])
    with colA:
        lock_now = st.button("🔒 Lock snapshot", use_container_width=True)
    with colB:
        clear_snap = st.button("Reset snapshot", use_container_width=True)
    with colC:
        mask_pii = st.checkbox("Mask sensitive IDs", value=True)
    with colD:
        recruiter_mode = st.checkbox("Detailed View (explain methods & stack)", value=True)

    with st.expander("Report assumptions used (read-only)", expanded=False):
        cap_buffer_pct    = st.number_input("Capital buffer (% of EAD)", value=1.00, step=0.10, format="%.2f", key="rep_cap")
        opex_pct          = st.number_input("Operating cost (% of EAD)", value=0.50, step=0.10, format="%.2f", key="rep_opex")
        target_profit_pct = st.number_input("Target profit (% of EAD)", value=0.50, step=0.10, format="%.2f", key="rep_profit")
        funding_cost_pct  = st.number_input("Funding cost (% per annum)", value=3.00, step=0.10, format="%.2f", key="rep_funding")

    if clear_snap:
        st.session_state.pop("report_snapshot", None)

    price_params = {
        "cap_buffer_pct": cap_buffer_pct,
        "opex_pct": opex_pct,
        "target_profit_pct": target_profit_pct,
        "funding_cost_pct": funding_cost_pct,
    }

    if lock_now or ("report_snapshot" not in st.session_state):
        st.session_state["report_snapshot"] = _build_snapshot(False, mask_pii, price_params, recruiter_mode)
    else:
        snap_old = st.session_state["report_snapshot"]
        params_changed = any(
            abs(float(price_params[k]) - float(snap_old["meta"].get(k, price_params[k]))) > 1e-12
            for k in ["cap_buffer_pct","opex_pct","target_profit_pct","funding_cost_pct"]
        )
        if (
            snap_old.get("mask_pii", True) != mask_pii
            or snap_old["meta"].get("RecruiterMode", True) != recruiter_mode
            or params_changed
        ):
            st.session_state["report_snapshot"] = _build_snapshot(False, mask_pii, price_params, recruiter_mode)

    snap = st.session_state["report_snapshot"]

    # ----------- Render the Report -----------
    st.markdown('<div class="report-card">', unsafe_allow_html=True)

    st.markdown(f"<div class='report-h1'>{snap['meta']['Title']} <span class='badge {snap['meta']['ScenarioClass']}'>{snap['meta']['Scenario']}</span></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='report-sub'>Built by <b>{snap['meta']['Author']}</b> &nbsp;·&nbsp; Generated: {snap['meta']['Generated']} &nbsp;·&nbsp; Snapshot: <span class='mono'>{snap['meta']['SnapshotID']}</span> &nbsp;·&nbsp; {snap['meta']['Version']}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='report-sub'>Shocks: {snap['meta']['Shocks']} &nbsp;·&nbsp; Filters: {snap['meta']['Filters']} &nbsp;·&nbsp; CECL discount: {snap['meta']['Discount rate (CECL)']} &nbsp;·&nbsp; {snap['meta']['Pricing assumptions']}</div>",
        unsafe_allow_html=True
    )

    st.markdown("<div class='toc'><b>Contents</b><br/>1. Inputs & Assumptions · 2. Executive KPIs · 3. Composition & Concentrations · 4. Risk Analysis · 5. Covenants & Watchlist · 6. Pricing Insights · 7. Data Quality</div>", unsafe_allow_html=True)

    if snap["meta"].get("RecruiterMode", True):
        st.markdown(
            "<div class='callout'><b>What this app demonstrates:</b> data ingestion & validation → base risk modeling (PD/LGD/EAD/EL) → macro stress impacts → CECL lifetime EL → covenant analytics & watchlist → risk-based pricing → Report Generation.<br><br>"
            "<b>Methods:</b> PD heuristic (credit score + leverage + sector), LGD via collateral haircuts, EAD as 12-month average, CECL via amortization & monthly default hazard, pricing = EL% + add-ons.<br>"
            "<b>Tech stack:</b> Streamlit · pandas · NumPy · numpy-financial · Matplotlib.</div>",
            unsafe_allow_html=True
        )

    # 1. Inputs & Assumptions
    st.markdown("### 1. Inputs & Assumptions")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"- **Scenario:** {snap['meta']['Scenario']}")
        st.markdown(f"- **Shocks:** {snap['meta']['Shocks']}")
        st.markdown(f"- **Filters:** {snap['meta']['Filters']}")
    with c2:
        st.markdown(f"- **CECL discount:** {snap['meta']['Discount rate (CECL)']}")
        st.markdown(f"- **Pricing:** {snap['meta']['Pricing assumptions']}")
        st.markdown(f"- **Loans in view:** {len(work_view)}")

    # 2. Executive KPIs
    st.markdown("### 2. Executive KPIs")
    kb = snap["kpis"]["base"]; ks = snap["kpis"]["stress"]; kd = snap["kpis"]["delta"]; kc = snap["kpis"]["cecl"]

    st.markdown("<div class='report-kpi-grid'>", unsafe_allow_html=True)
    st.markdown(f"<div class='report-kpi'><b>Total EAD — Base</b>{_fmt_money(kb['EAD'])}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='report-kpi'><b>W-avg PD — Base</b>{_fmt_pct(kb['PD'])}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='report-kpi'><b>Total EL — Base</b>{_fmt_money(kb['EL'])}</div>", unsafe_allow_html=True)
    if pd.notna(kb["HR"]): st.markdown(f"<div class='report-kpi'><b>High-risk share — Base</b>{kb['HR']:.1f}%</div>", unsafe_allow_html=True)
    else: st.markdown(f"<div class='report-kpi'><b>High-risk share — Base</b>-</div>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='report-kpi'><b>Total EAD — Stressed</b>{_fmt_money(ks['EAD'])}"
        f"<span class='delta'>{_safe_dollar(kd['EAD'])}</span></div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='report-kpi'><b>W-avg PD — Stressed</b>{_fmt_pct(ks['PD'])}"
        f"<span class='delta'>{_safe_pp(kd['PD'])}</span></div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='report-kpi'><b>Total EL — Stressed</b>{_fmt_money(ks['EL'])}"
        f"<span class='delta'>{_safe_dollar(kd['EL'])}</span></div>",
        unsafe_allow_html=True
    )
    if pd.notna(ks["HR"]):
        st.markdown(
            f"<div class='report-kpi'><b>High-risk share — Stressed</b>{ks['HR']:.1f}%"
            f"<span class='delta'>{_safe_pp(kd['HR']) if pd.notna(kb['HR']) and pd.notna(ks['HR']) else '—'}</span></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown("<div class='report-kpi'><b>High-risk share — Stressed</b>-</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f"- **Lifetime EL (PV)** — Base: **{_fmt_money2(kc['EL_base'])}** ({kc['pct_b']:.2f}% of start EAD) | "
        f"Stressed: **{_fmt_money2(kc['EL_st'])}** ({kc['pct_s']:.2f}% of start EAD)"
    )
    st.markdown("<div class='report-note'>Drivers: PD ↑ with rate/unemployment shocks; LGD ↑ with collateral shock; together these raise EL and the high-risk share.</div>", unsafe_allow_html=True)

    # 3. Composition & Concentrations
    st.markdown("### 3. Portfolio Composition & Concentrations")
    g1, g2 = st.columns(2)
    with g1:
        st.caption("Risk Tier (Base)")
        if snap["imgs"]["tier_base"] is not None:
            st.image(snap["imgs"]["tier_base"], use_container_width=True)
        else:
            st.info("Chart unavailable.")
    with g2:
        st.caption("Risk Tier (Stressed)")
        if snap["imgs"]["tier_stress"] is not None:
            st.image(snap["imgs"]["tier_stress"], use_container_width=True)
        else:
            st.info("Chart unavailable.")
    g3, g4 = st.columns(2)
    with g3:
        st.caption("Top Sectors by EAD (Stressed)")
        if snap["imgs"]["sector"] is not None:
            st.image(snap["imgs"]["sector"], use_container_width=True)
        else:
            st.info("Chart unavailable.")
    with g4:
        st.caption("Top Geographies by EAD (Stressed)")
        if snap["imgs"]["geo"] is not None:
            st.image(snap["imgs"]["geo"], use_container_width=True)
        else:
            st.info("Chart unavailable.")
    if np.isfinite(snap.get("top3_share", np.nan)):
        st.markdown(f"<div class='callout'>Concentration: Top 3 sectors account for <b>{snap['top3_share']:.1f}%</b> of stressed EAD.</div>", unsafe_allow_html=True)

    st.markdown("**Top 10 Exposures — by EAD (Stressed)**")
    if isinstance(snap["top_ead"], pd.DataFrame) and not snap["top_ead"].empty:
        st.table(snap["top_ead"])
    else:
        st.info("No exposures to display.")

    st.markdown("**Top 10 Exposures — by EL (Stressed)**")
    if isinstance(snap["top_el"], pd.DataFrame) and not snap["top_el"].empty:
        st.table(snap["top_el"])
    else:
        st.info("No exposures to display.")

    # 4. Risk Analysis
    st.markdown("### 4. Risk Analysis")
    base_cols = [c for c in ["LoanID","PD_12m","LGD","EAD_12m","EL_12m","RiskTier"] if c in work_view.columns]
    stress_cols = [c for c in ["LoanID","PD_12m_st","LGD_st","EAD_12m_st","EL_12m_st","RiskTier_st"] if c in work_view.columns]
    st.markdown("**Base Risk (12m)**")
    if base_cols: st.table(work_view[base_cols].head(15))
    else: st.info("Base risk columns not available.")
    st.markdown("**Stressed Risk (12m)**")
    if stress_cols: st.table(work_view[stress_cols].head(15))
    else: st.info("Stressed risk columns not available.")
    st.markdown("<div class='report-note'>Compare PD/LGD/EAD shifts between Base and Stressed to understand EL drivers.</div>", unsafe_allow_html=True)

    # 5. Covenants & Watchlist
    st.markdown("### 5. Covenant Compliance & Watchlist")
    bc = snap["cov"]["badge_counts"]
    if isinstance(bc, pd.Series) and not bc.empty:
        red = int(bc.get("Red", 0)); amber = int(bc.get("Amber", 0)); green = int(bc.get("Green", 0))
        st.markdown(
            f"- **Badges:** "
            f"<span class='chip chip-red'>Red {red}</span> "
            f"<span class='chip chip-amber'>Amber {amber}</span> "
            f"<span class='chip chip-green'>Green {green}</span>",
            unsafe_allow_html=True
        )
    else:
        st.markdown("- **Badges:** Red 0 · Amber 0 · Green 0")

    st.markdown("**Watchlist — Top 10 (by Worst Headroom)**")
    if isinstance(snap["cov"]["watch"], pd.DataFrame) and not snap["cov"]["watch"].empty:
        st.table(snap["cov"]["watch"])
        st.markdown("<div class='report-note'>“Tightest” shows which covenant is closest to breach for each loan.</div>", unsafe_allow_html=True)
    else:
        st.info("No Red/Amber items for current view.")

    # 6. Pricing Insights
    st.markdown("### 6. Pricing Insights (Risk-based)")
    p = snap["pricing"]
    line1 = "- **Portfolio EL rate (12m):** " + (_fmt_pct(p["el_rate"]) if np.isfinite(p["el_rate"]) else "-")
    line2 = "- **Suggested spread (portfolio):** " + (f"{p['spread_bps']:.0f} bps" if np.isfinite(p["spread_bps"]) else "-")
    line3 = "- **All-in rate (portfolio):** " + (f"{p['allin_pct']:.2f}%" if np.isfinite(p["allin_pct"]) else "-")
    st.markdown(line1); st.markdown(line2); st.markdown(line3)
    st.markdown("**Top 10 Loans by Suggested Spread**")
    if isinstance(p["top"], pd.DataFrame) and not p["top"].empty:
        st.table(p["top"])
    else:
        st.info("No pricing table available.")
    st.markdown("<div class='report-note'>Method: Spread = EL% + (cap + opex + profit). All-in = funding + spread.</div>", unsafe_allow_html=True)

    # 7. Data Quality
    st.markdown("### 7. Data Quality")
    if snap["dq"] is not None and not snap["dq"].empty:
        st.markdown(f"- **Rows with issues (sample):** {len(snap['dq'])} shown")
        st.table(snap["dq"])
    else:
        st.markdown("- No data quality issues detected in sample (top 10).")

    # Print tips
    st.markdown("**Print tips (Save as PDF):**")
    st.markdown(
        "- In the browser print dialog: set **Paper** to Letter or A4, **Margins** to Default/Narrow.\n"
        "- Turn **Headers/Footers off** for a clean look.\n"
        "- Keep sections concise for short PDFs."
    )

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Footer
# -------------------------
st.divider()

# Safe (re)definitions — tweak as needed
AUTHOR_NAME = "Aurokrishnaa R L"
AUTHOR_TAGLINE = "MS Finance (Quant) · MBA Finance"
LINKEDIN_URL = "https://www.linkedin.com/in/aurokrishnaa"
WEBSITE_URL  = "https://www.aurokrishnaa.me"

st.markdown(
    f"""
    <style>
      .cr-footerbar {{
        display:flex; align-items:center; justify-content:space-between; gap:14px; flex-wrap:wrap;
        background:var(--surface); border:1px solid var(--border); border-radius:14px;
        padding:12px 14px; box-shadow:0 1px 2px rgba(0,0,0,.03);
      }}
      .cr-foot-left {{
        display:flex; align-items:center; gap:10px; flex-wrap:wrap; color:var(--text);
      }}
      .cr-foot-name {{ font-weight:600; }}
      .cr-foot-tag  {{ color:var(--muted); }}
      .cr-links {{ display:flex; gap:8px; flex-wrap:wrap; }}
      .cr-btn {{
        display:inline-flex; align-items:center; gap:8px; text-decoration:none;
        padding:8px 12px; border-radius:10px; border:1px solid var(--border);
        box-shadow:0 1px 2px rgba(0,0,0,.04); transition:transform .02s ease-in-out;
      }}
      .cr-btn:hover {{ transform:translateY(-1px); }}
      .cr-btn.primary {{ background:var(--primary); color:white; border-color:rgba(0,0,0,.06); }}
      .cr-btn.alt {{ background:#fff; color:var(--text); }}
      .cr-foot-meta {{ margin-top:8px; text-align:center; color:var(--muted); font-size:.92rem; }}
    </style>

    <div class="cr-footerbar">
      <div class="cr-foot-left">
        <div class="cr-foot-name">Credit Risk Intelligence</div>
        <span>•</span>
        <div>Built by <b>{AUTHOR_NAME}</b></div>
        <span>—</span>
        <div class="cr-foot-tag">{AUTHOR_TAGLINE}</div>
      </div>
      <div class="cr-links">
        <a class="cr-btn alt" href="{WEBSITE_URL}" target="_blank" rel="noopener noreferrer"> Website</a>
        <a class="cr-btn primary" href="{LINKEDIN_URL}" target="_blank" rel="noopener noreferrer"> LinkedIn</a>
      </div>
    </div>
   <div class="cr-foot-meta">© 2025 • Aurokrishnaa R L — MS Finance (Quant) · MBA Finance • Credit Risk Intelligence • v1.3.0</div>
    """,
    unsafe_allow_html=True,
)


