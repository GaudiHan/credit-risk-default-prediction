# Credit Risk Default Prediction
### End-to-end machine learning pipeline for loan default classification · Python · XGBoost · Power BI

---

## Overview

This project builds a full credit risk analytics pipeline on the [Home Credit Default Risk dataset](https://www.kaggle.com/c/home-credit-default-risk/data) (307,511 applicants, 122 features). The goal is to predict the probability that a borrower will default on a loan — framed not just as a modelling exercise, but as a business decision-support tool for credit officers.

**Final model:** XGBoost classifier · ROC-AUC: `[your score]` · Precision@10%: `[your score]`

> 📊 [View the interactive Power BI dashboard](#) ← replace with your link

---

## Business Context

Traditional credit scoring often excludes applicants with limited credit history, creating both financial risk and missed opportunity for lenders. This analysis uses alternative applicant data — income, employment, family status, prior credit bureau behaviour — to build a more complete default risk picture.

**Key question answered:** *Which borrower segments carry disproportionate default risk, and what variables drive those outcomes?*

---

## Project Structure

```
credit-risk-default/
│
├── data/                        # Raw data (not committed — see Data section)
│   ├── application_train.csv
│   ├── application_test.csv
│   └── bureau.csv
│
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis & visualisation
│   ├── 02_feature_engineering.ipynb   # Feature creation & preprocessing
│   └── 03_modelling.ipynb       # Model training, evaluation & SHAP analysis
│
├── outputs/
│   ├── figures/                 # All saved charts
│   ├── model/                   # Saved XGBoost model (.pkl)
│   └── consulting_summary.pdf   # 1-page business summary
│
├── dashboard/
│   └── credit_risk_dashboard.pbix   # Power BI file
│
├── requirements.txt
└── README.md
```

---

## Data

**Source:** [Kaggle — Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data) (free, requires Kaggle account)

**Tables used:**

| File | Rows | Description |
|---|---|---|
| `application_train.csv` | 307,511 | Core application data with target variable |
| `application_test.csv` | 48,744 | Holdout set for final predictions |
| `bureau.csv` | 1,716,428 | Prior credit bureau records per applicant |

**Target variable:** `TARGET` — 1 = payment difficulties (default), 0 = repaid on time

**Class imbalance:** ~8% positive class. Handled via `scale_pos_weight` in XGBoost.

To download via Kaggle CLI:
```bash
pip install kaggle
kaggle competitions download -c home-credit-default-risk
unzip home-credit-default-risk.zip -d data/
```

---

## Methodology

### 1. Exploratory Data Analysis
- Profiled 122 features across 307K applicants
- Identified class imbalance (8% default rate) and missing value patterns
- Key finding: `EXT_SOURCE_1/2/3` (external credit scores) are the strongest predictors

### 2. Feature Engineering
Created domain-driven features grounded in credit analysis logic:

| Feature | Formula | Rationale |
|---|---|---|
| `CREDIT_INCOME_RATIO` | `AMT_CREDIT / AMT_INCOME_TOTAL` | Debt burden relative to income |
| `ANNUITY_INCOME_RATIO` | `AMT_ANNUITY / AMT_INCOME_TOTAL` | Monthly repayment affordability |
| `EMPLOYMENT_TO_AGE_RATIO` | `DAYS_EMPLOYED / DAYS_BIRTH` | Job stability relative to career stage |
| `BUREAU_ACTIVE_LOANS` | Aggregated from `bureau.csv` | Current external credit exposure |

### 3. Modelling

| Model | ROC-AUC | Notes |
|---|---|---|
| Logistic Regression (baseline) | `[score]` | After scaling & imputation |
| XGBoost | `[score]` | Tuned via RandomizedSearchCV |

### 4. Explainability (SHAP)
Used SHAP TreeExplainer to surface the top drivers of default probability — making the model interpretable for non-technical credit officers.

**Top predictors of default:**
1. `EXT_SOURCE_2` — external credit score (lower = higher risk)
2. `EXT_SOURCE_3` — secondary credit score
3. `CREDIT_INCOME_RATIO` — high debt-to-income flags elevated risk
4. `DAYS_EMPLOYED` — shorter employment tenure correlates with default
5. `AMT_GOODS_PRICE` — loan purpose and size matter

---

## Model Performance

| Model | ROC-AUC | Precision@80%Recall | Notes |
|-------|---------|---------------------|-------|
| Random Forest (baseline) | 0.701 | — | Simple, no tuning |
| Random Forest (optimized) | 0.737 | — | class_weight='balanced' |
| **XGBoost (tuned)** | **0.753** | **13.6%** | RandomizedSearchCV, SHAP |

---

## Key Business Findings

- Applicants with `CREDIT_INCOME_RATIO > 3.5x` are **`[X]`x more likely to default** — flagging this threshold could reduce portfolio risk without significantly restricting approvals
- Borrowers with no prior credit bureau history (`bureau.csv` nulls) show a **`[X]`% higher default rate** — suggesting value in alternative data scoring for this segment
- `EXT_SOURCE_2` below `0.3` correlates with a **`[X]`% default rate** vs `[Y]`% for scores above `0.6`

> See the full consulting summary in `outputs/consulting_summary.pdf`

---

## Dashboard

Built in Power BI with three views:

1. **Portfolio Overview** — default rate by loan type, income bracket, and family status
2. **Risk Segmentation** — borrower risk score distribution across key variables
3. **Model Insights** — SHAP feature importance with business interpretation

> 📊 [View on Tableau Public / Power BI Web](#) ← replace with your published link

---

## Setup & Reproduction

```bash
# Clone the repo
git clone https://github.com/GaudiHan/credit-risk-default.git
cd credit-risk-default

# Install dependencies
pip install -r requirements.txt

# Download data (see Data section above)

# Run notebooks in order
jupyter notebook notebooks/01_eda.ipynb
```

**Requirements:**
```
pandas==2.1.0
numpy==1.26.0
scikit-learn==1.3.0
xgboost==2.0.0
shap==0.43.0
matplotlib==3.8.0
seaborn==0.13.0
jupyter==1.0.0
```

---

## Skills Demonstrated

`Python` `Pandas` `XGBoost` `Scikit-learn` `SHAP` `Feature Engineering` `Power BI` `Imbalanced Classification` `Business Communication`

---

## Author

**Alexander Gaudi Suhandjaja**
[linkedin.com/in/gaudihan](https://linkedin.com/in/gaudihan) · [github.com/GaudiHan](https://github.com/GaudiHan)

*Final-year Data Science & Engineering student at HKU | Targeting Data Analyst roles in Hong Kong*
