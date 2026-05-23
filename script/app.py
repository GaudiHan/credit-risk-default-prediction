import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 15px; padding: 6px 0; }
.metric-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center;
}
.metric-card .label { font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 4px; }
.metric-card .value { font-size: 28px; font-weight: 600; color: #0f172a; }
.metric-card .delta { font-size: 12px; color: #64748b; margin-top: 4px; }
.insight-box {
    background: #eff6ff; border-left: 4px solid #3b82f6;
    border-radius: 0 8px 8px 0; padding: 1rem 1.25rem; margin: 1rem 0;
}
.insight-box p { margin: 0; font-size: 14px; color: #1e3a5f; }
.warn-box {
    background: #fff7ed; border-left: 4px solid #f97316;
    border-radius: 0 8px 8px 0; padding: 1rem 1.25rem; margin: 1rem 0;
}
.warn-box p { margin: 0; font-size: 14px; color: #7c2d12; }
</style>
""", unsafe_allow_html=True)

# ── color palette ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
BLUE   = "#3b82f6"
PURPLE = "#8b5cf6"
AMBER  = "#f59e0b"
RED    = "#ef4444"
GREEN  = "#10b981"
GRAY   = "#94a3b8"
PALETTE = [BLUE, PURPLE, AMBER, GREEN, RED, GRAY]

# ── data loading ────────────────────────────────────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_main(path="../data/application_train.csv"):
    cols = [
        "TARGET", "NAME_CONTRACT_TYPE", "AMT_INCOME_TOTAL", "AMT_CREDIT",
        "AMT_ANNUITY", "NAME_FAMILY_STATUS", "NAME_INCOME_TYPE",
        "DAYS_EMPLOYED", "DAYS_BIRTH", "EXT_SOURCE_1", "EXT_SOURCE_2",
        "EXT_SOURCE_3", "CNT_FAM_MEMBERS", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
        "CODE_GENDER",
    ]
    existing = None
    for candidate in [path, f"data/{path}", f"outputs/{path}"]:
        if os.path.exists(candidate):
            existing = candidate
            break
    if existing is None:
        return None
    df = pd.read_csv(existing, usecols=lambda c: c in cols)
    # derived features
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365).round(0)
    df["INCOME_BRACKET"] = pd.qcut(
        df["AMT_INCOME_TOTAL"].clip(0, 500_000), q=4,
        labels=["Low (<112K)", "Lower-mid (112–157K)", "Upper-mid (157–225K)", "High (>225K)"]
    )
    df["DTI_BAND"] = pd.cut(
        df["CREDIT_INCOME_RATIO"],
        bins=[0, 1, 2, 3.5, 100],
        labels=["<1×", "1×–2×", "2×–3.5×", ">3.5×"],
    )
    return df

@st.cache_data(show_spinner=False)
def load_feature_importance():
    for p in ["outputs/results/feature_importance.csv",
              "outputs/feature_importance.csv",
              "feature_importance.csv",
              "xgboost_feature_importance.csv"]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            # normalise column names
            df.columns = [c.lower().strip() for c in df.columns]
            if "feature" not in df.columns:
                df = df.rename(columns={df.columns[0]: "feature"})
            if "importance" not in df.columns:
                for c in df.columns:
                    if c != "feature":
                        df = df.rename(columns={c: "importance"})
                        break
            return df.nlargest(20, "importance")
    return None

@st.cache_data(show_spinner=False)
def load_predictions():
    for p in ["outputs/results/xgboost_predictions.csv",
              "outputs/xgboost_predictions.csv",
              "xgboost_predictions.csv"]:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None

# ── business interpretation map ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
FEATURE_LABELS = {
    "EXT_SOURCE_2": "External credit score 2",
    "EXT_SOURCE_3": "External credit score 3",
    "EXT_SOURCE_1": "External credit score 1",
    "DAYS_BIRTH": "Applicant age",
    "DAYS_EMPLOYED": "Employment duration",
    "AMT_CREDIT": "Loan amount",
    "AMT_ANNUITY": "Annual repayment",
    "AMT_INCOME_TOTAL": "Annual income",
    "CREDIT_INCOME_RATIO": "Credit-to-income ratio",
    "ANNUITY_INCOME_RATIO": "Repayment affordability",
    "EMPLOYMENT_TO_AGE_RATIO": "Job stability index",
    "THIN_FILE_FLAG": "No bureau history flag",
    "REGION_RATING_CLIENT": "Regional risk rating",
    "AMT_GOODS_PRICE": "Purchase price of goods",
    "DAYS_REGISTRATION": "Registration recency",
    "CNT_FAM_MEMBERS": "Family size",
}

FEATURE_INSIGHTS = {
    "EXT_SOURCE_2": "Strongest single predictor. Scores below 0.3 correlate with 3.5× higher default rates — use as primary screen.",
    "EXT_SOURCE_3": "Second-ranked credit signal. Corroborates EXT_SOURCE_2; when both are low, risk compounds significantly.",
    "EXT_SOURCE_1": "Third external bureau score. Useful for thin-file applicants where sources 2 & 3 may be missing.",
    "DAYS_BIRTH": "Older applicants default less. Under-25 cohort shows elevated risk, likely due to shorter credit histories.",
    "DAYS_EMPLOYED": "Employment under 1 year raises default probability by 1.7×. Flag short-tenure applicants for manual review.",
    "CREDIT_INCOME_RATIO": "Non-linear risk. Peak default at 2×–3.5× DTI, not at the highest debt levels — use multi-band thresholds.",
    "ANNUITY_INCOME_RATIO": "Monthly payment burden. Applicants spending >30% of income on repayments show elevated stress.",
}

# ── sidebar ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("##Credit Risk")
    st.markdown("---")
    view = st.radio(
        "View",
        ["Portfolio Overview", "Risk Segmentation", "Model Insights"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#94a3b8'>XGBoost · ROC-AUC 0.753<br>"
        "LightGBM · ROC-AUC 0.752<br>"
        "Home Credit · 307K applications</small>",
        unsafe_allow_html=True,
    )

# ── load data ─────────────────────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading data…"):
    df = load_main()
    fi = load_feature_importance()
    preds = load_predictions()

if df is None:
    st.error(
        "**application_train.csv not found.** "
        "Place it in the same directory as app.py (or in a `data/` subfolder) and restart."
    )
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# VIEW 1 — PORTFOLIO OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
if view == "Portfolio Overview":
    st.title("Portfolio Overview")
    st.caption(f"{len(df):,} loan applications · {df['TARGET'].mean()*100:.2f}% portfolio default rate")

    # KPI row
    total = len(df)
    defaults = df["TARGET"].sum()
    rate = df["TARGET"].mean() * 100
    cash_pct = (df["NAME_CONTRACT_TYPE"] == "Cash loans").mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, delta in [
        (c1, "Total Applications", f"{total:,}", ""),
        (c2, "Defaults", f"{int(defaults):,}", ""),
        (c3, "Default Rate", f"{rate:.2f}%", "8.07% portfolio avg"),
        (c4, "Cash Loans Share", f"{cash_pct:.1f}%", "vs revolving"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{val}</div>
            <div class="delta">{delta}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Row 1: loan type + family status
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Default rate by loan type")
        lt = df.groupby("NAME_CONTRACT_TYPE")["TARGET"].agg(["mean", "count"]).reset_index()
        lt.columns = ["Loan type", "Default rate", "Count"]
        lt["Default rate %"] = lt["Default rate"] * 100
        fig = px.bar(
            lt, x="Loan type", y="Default rate %", text="Default rate %",
            color="Loan type", color_discrete_sequence=[BLUE, PURPLE],
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(showlegend=False, yaxis_title="Default rate (%)",
                          plot_bgcolor="white", margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Default rate by family status")
        fs = df.groupby("NAME_FAMILY_STATUS")["TARGET"].mean().reset_index()
        fs.columns = ["Family status", "Default rate"]
        fs["Default rate %"] = fs["Default rate"] * 100
        fs = fs.sort_values("Default rate %", ascending=True)
        fig = px.bar(
            fs, x="Default rate %", y="Family status", orientation="h",
            text="Default rate %", color="Default rate %",
            color_continuous_scale=["#dbeafe", "#3b82f6", "#1d4ed8"],
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, xaxis_title="Default rate (%)",
                          plot_bgcolor="white", margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: income bracket
    st.subheader("Default rate by income bracket")
    ib = df.groupby("INCOME_BRACKET", observed=True)["TARGET"].agg(["mean", "count"]).reset_index()
    ib.columns = ["Income bracket", "Default rate", "Count"]
    ib["Default rate %"] = ib["Default rate"] * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=ib["Income bracket"], y=ib["Default rate %"],
        name="Default rate", marker_color=BLUE, text=ib["Default rate %"],
        texttemplate="%{text:.2f}%", textposition="outside",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=ib["Income bracket"], y=ib["Count"],
        name="# applicants", mode="lines+markers",
        line=dict(color=AMBER, width=2), marker=dict(size=8),
    ), secondary_y=True)
    fig.update_layout(plot_bgcolor="white", legend=dict(orientation="h", y=1.1),
                      margin=dict(t=20, b=20))
    fig.update_yaxes(title_text="Default rate (%)", secondary_y=False)
    fig.update_yaxes(title_text="# applicants", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class="insight-box"><p>
    💡 <strong>Key finding:</strong> Higher income brackets show lower default rates, but the relationship
    is not perfectly linear — lower-mid income applicants sometimes outperform upper-mid cohorts due to
    more conservative borrowing habits. Volume is concentrated in the lower-mid bracket.
    </p></div>""", unsafe_allow_html=True)

    # Row 3: income type heatmap
    st.subheader("Default rate by income type × loan type")
    hm = df.groupby(["NAME_INCOME_TYPE", "NAME_CONTRACT_TYPE"])["TARGET"].mean().unstack()
    hm = hm * 100
    fig = px.imshow(
        hm, text_auto=".1f", color_continuous_scale=["#dbeafe", "#3b82f6", "#1e3a8a"],
        aspect="auto", labels=dict(color="Default rate %"),
    )
    fig.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# VIEW 2 — RISK SEGMENTATION
# ═════════════════════════════════════════════════════════════════════════════
elif view == "Risk Segmentation":
    st.title("Risk Segmentation")
    st.caption("Borrower risk score distribution and segmentation across key variables")

    # filters
    col_f1, col_f2 = st.columns(2)
    loan_types = ["All"] + sorted(df["NAME_CONTRACT_TYPE"].dropna().unique().tolist())
    sel_loan = col_f1.selectbox("Loan type", loan_types)
    family_types = ["All"] + sorted(df["NAME_FAMILY_STATUS"].dropna().unique().tolist())
    sel_family = col_f2.selectbox("Family status", family_types)

    dff = df.copy()
    if sel_loan != "All":
        dff = dff[dff["NAME_CONTRACT_TYPE"] == sel_loan]
    if sel_family != "All":
        dff = dff[dff["NAME_FAMILY_STATUS"] == sel_family]

    st.caption(f"Showing {len(dff):,} applicants · {dff['TARGET'].mean()*100:.2f}% default rate")
    st.markdown("---")

    # EXT_SOURCE_2 distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Credit score distribution (EXT_SOURCE_2)")
        fig = go.Figure()
        for target, label, color in [(0, "Non-default", BLUE), (1, "Default", RED)]:
            vals = dff[dff["TARGET"] == target]["EXT_SOURCE_2"].dropna()
            fig.add_trace(go.Histogram(
                x=vals, name=label, opacity=0.7, nbinsx=40,
                marker_color=color, histnorm="probability density",
            ))
        fig.update_layout(
            barmode="overlay", plot_bgcolor="white",
            xaxis_title="EXT_SOURCE_2 score", yaxis_title="Density",
            legend=dict(orientation="h", y=1.1), margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Default rate by credit score band")
        dff["EXT_BAND"] = pd.cut(
            dff["EXT_SOURCE_2"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"],
        )
        eb = dff.groupby("EXT_BAND", observed=True)["TARGET"].agg(["mean", "count"]).reset_index()
        eb.columns = ["Score band", "Default rate", "Count"]
        eb["Default rate %"] = eb["Default rate"] * 100
        colors = [RED if r > 10 else AMBER if r > 6 else GREEN for r in eb["Default rate %"]]
        fig = px.bar(eb, x="Score band", y="Default rate %", text="Default rate %",
                     color="Score band", color_discrete_sequence=colors)
        fig.add_hline(y=dff["TARGET"].mean()*100, line_dash="dash",
                      line_color=GRAY, annotation_text="Portfolio avg")
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(showlegend=False, plot_bgcolor="white",
                          yaxis_title="Default rate (%)", margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # DTI bands
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Default rate by DTI band")
        dti = dff.groupby("DTI_BAND", observed=True)["TARGET"].agg(["mean", "count"]).reset_index()
        dti.columns = ["DTI band", "Default rate", "Count"]
        dti["Default rate %"] = dti["Default rate"] * 100
        fig = px.bar(dti, x="DTI band", y="Default rate %", text="Default rate %",
                     color="Default rate %",
                     color_continuous_scale=["#dcfce7", "#fef3c7", "#fee2e2"])
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, plot_bgcolor="white",
                          yaxis_title="Default rate (%)", margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""<div class="warn-box"><p>
        <strong>Non-linear DTI risk:</strong> The 2×–3.5× band peaks higher than >3.5×.
        Very high-DTI applicants may have compensating factors (higher income, better credit scores).
        Apply multi-band thresholds rather than a single cut-off.
        </p></div>""", unsafe_allow_html=True)

    with col4:
        st.subheader("Employment tenure vs default rate")
        dff["EMP_BAND"] = pd.cut(
            (-dff["DAYS_EMPLOYED"].clip(upper=0)) / 365,
            bins=[0, 1, 3, 5, 10, 50],
            labels=["<1 yr", "1–3 yrs", "3–5 yrs", "5–10 yrs", ">10 yrs"],
        )
        emp = dff.groupby("EMP_BAND", observed=True)["TARGET"].agg(["mean", "count"]).reset_index()
        emp.columns = ["Employment", "Default rate", "Count"]
        emp["Default rate %"] = emp["Default rate"] * 100
        fig = px.bar(emp, x="Employment", y="Default rate %", text="Default rate %",
                     color="Employment",
                     color_discrete_sequence=[RED, AMBER, AMBER, GREEN, GREEN])
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(showlegend=False, plot_bgcolor="white",
                          yaxis_title="Default rate (%)", margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Age distribution
    st.subheader("Age distribution by default status")
    fig = go.Figure()
    for target, label, color in [(0, "Non-default", BLUE), (1, "Default", RED)]:
        vals = dff[dff["TARGET"] == target]["AGE_YEARS"].dropna()
        fig.add_trace(go.Violin(
            x=vals, name=label, box_visible=True, meanline_visible=True,
            line_color=color, opacity=0.7,
        ))
    fig.update_layout(
        violinmode="overlay", plot_bgcolor="white",
        xaxis_title="Age (years)", legend=dict(orientation="h", y=1.1),
        margin=dict(t=20, b=20), height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# VIEW 3 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════
elif view == "Model Insights":
    st.title("Model Insights")
    st.caption("XGBoost (ROC-AUC 0.753) · SHAP-based feature importance with business interpretation")

    # Model comparison banner
    col_m1, col_m2, col_m3 = st.columns(3)
    for col, name, auc, note in [
        (col_m1, "Random Forest", "0.701", "Baseline"),
        (col_m2, "Random Forest balanced", "0.737", "class_weight='balanced'"),
        (col_m3, "XGBoost tuned *", "0.753", "Winner · scale_pos_weight"),
    ]:
        col.markdown(f"""<div class="metric-card">
            <div class="label">{name}</div>
            <div class="value">{auc}</div>
            <div class="delta">{note}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    if fi is not None:
        fi["label"] = fi["feature"].map(FEATURE_LABELS).fillna(fi["feature"])
        fi_sorted = fi.sort_values("importance", ascending=True)

        col_fi, col_insight = st.columns([3, 2])

        with col_fi:
            st.subheader("Feature importance (top 20)")
            colors = [RED if f.startswith("EXT_SOURCE") else
                      PURPLE if f in ("CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "EMPLOYMENT_TO_AGE_RATIO", "THIN_FILE_FLAG") else
                      BLUE for f in fi_sorted["feature"]]
            fig = go.Figure(go.Bar(
                x=fi_sorted["importance"], y=fi_sorted["label"],
                orientation="h", marker_color=colors,
                text=fi_sorted["importance"].round(4),
                textposition="outside",
            ))
            fig.update_layout(
                plot_bgcolor="white", height=600,
                xaxis_title="Importance score",
                margin=dict(t=10, b=10, r=120),
            )
            st.plotly_chart(fig, use_container_width=True)

            # legend
            st.markdown("""
            <div style='display:flex;gap:1.5rem;font-size:13px;color:#64748b;margin-top:-1rem'>
            <span><span style='background:#ef4444;border-radius:3px;display:inline-block;width:12px;height:12px;margin-right:4px'></span>External credit bureau</span>
            <span><span style='background:#8b5cf6;border-radius:3px;display:inline-block;width:12px;height:12px;margin-right:4px'></span>Engineered features</span>
            <span><span style='background:#3b82f6;border-radius:3px;display:inline-block;width:12px;height:12px;margin-right:4px'></span>Raw application data</span>
            </div>""", unsafe_allow_html=True)

        with col_insight:
            st.subheader("Business interpretation")
            top_features = fi.sort_values("importance", ascending=False).head(7)["feature"].tolist()
            for feat in top_features:
                if feat in FEATURE_INSIGHTS:
                    label = FEATURE_LABELS.get(feat, feat)
                    st.markdown(f"""<div style='margin-bottom:10px;padding:10px 14px;
                        background:#f8fafc;border-radius:8px;border:1px solid #e2e8f0'>
                        <div style='font-size:13px;font-weight:600;color:#0f172a;margin-bottom:4px'>{label}</div>
                        <div style='font-size:12px;color:#475569'>{FEATURE_INSIGHTS[feat]}</div>
                    </div>""", unsafe_allow_html=True)

    else:
        st.warning(
            "Feature importance file not found. "
            "Place `xgboost_feature_importance.csv` (columns: feature, importance) "
            "in the project directory or `outputs/results/`."
        )

    st.markdown("---")

    # Score distribution from predictions
    st.subheader("Predicted risk score distribution")
    if preds is not None:
        score_col = next((c for c in preds.columns if "prob" in c.lower() or "score" in c.lower() or "predict" in c.lower()), None)
        target_col = next((c for c in preds.columns if c.upper() == "TARGET"), None)

        if score_col:
            fig = go.Figure()
            if target_col:
                for target, label, color in [(0, "Non-default", BLUE), (1, "Default", RED)]:
                    vals = preds[preds[target_col] == target][score_col]
                    fig.add_trace(go.Histogram(
                        x=vals, name=label, opacity=0.7, nbinsx=50,
                        marker_color=color, histnorm="probability density",
                    ))
                fig.update_layout(barmode="overlay")
            else:
                fig.add_trace(go.Histogram(x=preds[score_col], nbinsx=50, marker_color=BLUE))

            fig.add_vline(x=0.5, line_dash="dash", line_color=GRAY,
                          annotation_text="Default threshold (0.5)")
            fig.update_layout(
                plot_bgcolor="white", xaxis_title="Predicted default probability",
                yaxis_title="Density", legend=dict(orientation="h", y=1.1),
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Score column not detected in predictions file. Expected a column named 'probability', 'score', or similar.")
    else:
        st.info("Predictions file not found. Place `xgboost_predictions.csv` in the project directory or `outputs/results/`.")

    # Recall curve callout
    st.markdown("""<div class="insight-box"><p>
    <strong>Operational efficiency:</strong> Reviewing only the top 13.6% of applicants by predicted risk
    captures 80% of all defaults — 2× better than random screening. This means a credit team can focus
    manual review effort on 1 in 7 applicants while catching 4 in 5 defaults.
    </p></div>""", unsafe_allow_html=True)