# Credit Risk Intelligence (Streamlit)

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://makeapullrequest.com)

> Stress, assess, and price credit risk with explainable metrics & a print-friendly report — built in Python/Streamlit by **Aurokrishnaa R L** (MS Finance — Quant).

<!-- After deploying on Streamlit Cloud, replace the # link below with your live app URL -->
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](#)

---

## ✨ What this app does

- **Data ingestion** from CSV (robust to encoding issues) + **template** included
- **Base risk**: PD / LGD / EAD / EL at loan level with explainable heuristics
- **Stress testing**: rate, unemployment, collateral shocks → risk & loss impact
- **Covenants & Watchlist**: DSCR, ICR, LTV, DTI headrooms with Red/Amber/Green badges
- **CECL / IFRS**: lifetime expected loss via amortization & monthly default intensity
- **Pricing**: EL% + (capital + opex + profit) → suggested spread & all-in rate
- **Report Mode**: clean, print-ready executive view (save as PDF from browser)

Built with: **Streamlit · pandas · NumPy · numpy-financial · Matplotlib**

---

## 🖼️ Screenshots

> You’ll add these later to `/assets`. Keep the names and they’ll auto-render here.

- **Overview**  
  ![Overview](assets/overview.png)

- **KPIs**  
  ![KPIs](assets/kpis.png)

- **Risk Tiers & Concentrations**  
  ![Risk Tiers](assets/risk_tiers.png)

- **Pricing Table**  
  ![Pricing](assets/pricing.png)

---

## 🚀 Quickstart (local)

```bash
# clone
git clone https://github.com/<your-username>/credit-risk-intel.git
cd credit-risk-intel

# (optional) create venv
python -m venv .venv
source .venv/bin/activate     # mac/linux
# .venv\Scripts\activate      # windows

# install
pip install -U pip
pip install -r requirements.txt

# run
streamlit run app.py
