# Credit Risk Default Prediction
### End-to-end machine learning pipeline for loan default classification · Python · XGBoost · Power BI

---

## Quick Look at Key Metrics 

| Metric | Value |
|--------|-------|
| Dataset size | 307,511 applications |
| Default rate | 8.07% |
| XGBoost ROC-AUC | **0.753** |
| Low credit score (<0.3) default rate | 15.88% (3.5x risk) |
| Short employment (<1 year) default rate | 10.98% (1.7x risk) |
| Peak DTI risk band | 2x-3.5x (8.81% default) |
| Precision at 80% recall | 13.6% (2x better than random) |

---

## Overview

This project builds a full credit risk analytics pipeline on the [Home Credit Default Risk dataset](https://www.kaggle.com/c/home-credit-default-risk/data) (307,511 applicants, 122 features). The goal is to predict the probability that a borrower will default on a loan. This project is intended to be a modelling exercise and a business decision-support tool for credit officers.

**Final model:** XGBoost classifier · ROC-AUC: **0.753** · Precision@80%Recall: **13.6%**

> 📊 [View the interactive Power BI dashboard](https://gaudihan-credit-risk-default-prediction.streamlit.app/) ← coming soon

---

## Business Context

Traditional credit scoring often excludes applicants with limited credit history, creating both financial risk and missed opportunity for lenders. This analysis uses alternative applicant data such as income, employment, family status, and prior credit bureau behaviour to build a more complete default risk picture.

**Key question answered:** *Which borrower segments carry disproportionate default risk, and what variables drive those outcomes?*

---

## Project Structure

```
credit-risk-default/
│ 
├── data/                               # Raw data (not committed, exceed file limit, see the following Data section)
│   ├── application_train.csv
│   ├── application_test.csv
│   └── bureau.csv 
│   
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory data analysis & visualisation
│   ├── 02_feature_engineering.ipynb    # Feature creation & preprocessing 
│   ├── 03_modelling.ipynb              # RandomForest Model training and evaluation
│   ├── 04_xgboost_modelling.ipynb      # XGBoost Model training, evaluation & SHAP analysis 
│   └── 05_lightgbm_modelling.ipynb     # LightGBM Model training, evaluation & SHAP analysis 
│ 
├── outputs/
│   ├── data/                           # Saved training data
│   ├── figures/                        # All saved charts 
│   ├── models/                         # Saved models (.pkl)
│   └── results/                        # Various process results 
│   
├── app.py                              # Streamlit app 
├── consulting_summary.html             # Consulting summary
├── consulting_summary.pdf              # Consulting summary, but pdf
├── requirements.txt
└── README.md
```
 
---

## Data

**Source:** [Kaggle - Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data)  

**Tables used:**

| File | Rows | Description | 
|---|---|---|
| `application_train.csv` | 307,511 | Core application data with target variable | 
| `application_test.csv` | 48,744 | Holdout set for final predictions |
| `bureau.csv` | 1,716,428 | Prior credit bureau records per applicant | 

**Target variable:** `TARGET`: 1 = payment difficulties (default), 0 = repaid on time   

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
| `FAMILY_SIZE` | `CNT_FAM_MEMBERS + 1` | Household size impact on repayment capacity |  

### 3. Modelling

| Model | ROC-AUC | Notes |
|-------|---------|-------------|-------|
| Random Forest (baseline) | 0.701 | Simple, no tuning |
| Random Forest (optimized) | 0.737 | `class_weight='balanced'` |
| XGBoost (tuned) | **0.753** | RandomizedSearchCV, SHAP |
| LightGBM (tuned) | 0.752 | Slightly faster training |
| **Best Model** | **XGBoost 0.753** | Selected for final deployment |

### 4. Explainability (SHAP)
SHAP TreeExplainer was used to show the top drivers of default probability, so that the model can be interpreted for non-technical credit officers.

**Top predictors of default:**
1. `EXT_SOURCE_2`, external credit score (lower = higher risk)
2. `EXT_SOURCE_3`, secondary credit score
3. `AMT_GOODS_PRICE`, loan purpose and size matter
4. `EXT_SOURCE_1`, tertiary credit score
5. `AMT_CREDIT`, total credit amount

---

## Model Performance

| Model | ROC-AUC | Improvement | Notes | 
|-------|---------|-------------|-------|
| Random Forest (baseline) | 0.701 | - | Simple, no tuning |
| Random Forest (optimized) | 0.737 | +3.6% | `class_weight='balanced'` |  
| XGBoost (tuned) | **0.753** | **+5.2%** | RandomizedSearchCV, SHAP | 
| LightGBM (tuned) | 0.752 | +5.1% | Slightly faster training |
| **Best Model** | **XGBoost 0.753** | - | Selected for final deployment |

---

## Key Business Findings

### 1. External Credit Scores Dominate Prediction 

SHAP analysis confirms `EXT_SOURCE_2`, `EXT_SOURCE_3`, and `EXT_SOURCE_1` are the top three predictors of default.
 
| Credit Score Range | Default Rate | Risk vs. High Score | 
|-------------------|--------------|---------------------|
| `EXT_SOURCE_2` < 0.3 (Low) | **15.88%** | **3.5x higher** | 
| `EXT_SOURCE_2` > 0.6 (High) | 4.56% | Baseline | 

**Business implication:** Applicants with scores below 0.3 should trigger enhanced review. This single flag identifies a segment with nearly 4x the risk of high-score applicants.

### 2. Debt Burden Shows Non-Linear Risk 

Unlike the common assumption that "higher debt = higher risk," our analysis reveals a peak in the middle:

| Debt-to-Income Band | Default Rate | Insight |  
|--------------------|--------------|---------| 
| Low (≤2x) | 7.43% | Safest group |
| **Medium (2x-3.5x)** | **8.81%** | **Peak risk** |
| High (>3.5x) | 7.95% | Slightly lower |

**Why this happens:** Applicants with very high debt burdens may be higher income or have compensating factors (good credit scores). The medium band contains more marginal borrowers. 
 
**Business implication:** Use multi-band thresholds, not a single cap. 

### 3. Employment Tenure is a Material Risk Factor

| Employment Length | Default Rate | Risk Multiplier |
|------------------|--------------|-----------------|
| < 1 year | **10.98%** | **1.7x** |
| > 5 years | 6.41% | Baseline | 

**Business implication:** Short-tenure employees, particularly young professionals, represent a distinct risk segment. Consider alternative verification (e.g., employment contract, industry stability) for this group.

### 4. Thin-File Applicants (no bureau history) 

Only 0.1% of applicants in this dataset lack bureau history, and they show no elevated risk (8.14% vs 8.07%). This finding is dataset-specific; in Hong Kong's context with high young professional immigration, thin-file risk may be more significant and warrants separate analysis.
 
### 5. SHAP Feature Importance

The SHAP summary plot below shows which features push default probability up (red/high values) vs down (blue/low values): 

![SHAP Summary Plot](outputs/figures/shap_summary_plot.png)  

*Interpretation: Low values of EXT_SOURCE_2/3 (blue) push risk higher; high values (red) reduce risk.* 

![SHAP Bar Plot](outputs/figures/shap_bar_plot.png)

*Average impact magnitude: External credit scores have the largest influence, followed by loan amount (`AMT_GOODS_PRICE`) and credit amount (`AMT_CREDIT`).*

---

**Key takeaway for credit officers:** Focus on external credit scores first as it is the strongest signal. Use DTI as a secondary flag, though keep in mind that the 2x-3.5x band contains hidden risk. Short employment tenure is a material risk factor worth investigating further.

---

## Dashboard

Streamlit dashboard is deployed for real-time credit risk exploration:

> 📊 [View Live Dashboard](https://gaudihan-credit-risk-default-prediction.streamlit.app/)

### Dashboard Features 

| Feature | Description |
|------------------|--------------| 
| KPI Card | Total applications (307K), default rate (8.07%), model ROC-AUC (0.753), and flagged rate at 80% recall |
| Risk Bucket Distribution | Bar chart showing applications segmented by predicted risk (Very Low → Very High) |
| Default Rate by Segment | Historical default rates across borrower segments (credit score, employment tenure, DTI bands) | 
| Default Probability Distribution | Histogram of model predictions with 0.4 risk threshold line (captures 80% of defaults) |  
| Top Predictors | XGBoost feature importance chart; `EXT_SOURCE_2/3/1` dominate |
| Risk Segment Details | Sortable table with default rates and sample sizes for all segments | 

### Key Dashboard Insights

1. **Short-term employees (<1 year)** show the highest default rate at 10.98% (1.7× risk multiplier)
2. **Low credit score applicants (<0.3)** default at 15.88% (3.5× higher than high-score group)
3. The model flags **13.6% of applicants** to capture **80% of actual defaults**

---

## Setup & Reproduction

```bash
# Clone the repo
git clone https://github.com/GaudiHan/Credit-Risk-Default-Prediction.git
cd Credit-Risk-Default-Prediction

# Install dependencies
pip install -r requirements.txt

# Download data (see Data section above)

# Run notebooks in the notebooks/ directory in order
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
