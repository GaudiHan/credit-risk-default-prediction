"""
app.py - Credit Risk Default Prediction Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="📊",
    layout="wide"
)

# ============================================================================
# LOAD DATA
# ============================================================================

@st.cache_data
def load_data():
    """Load all CSV files"""
    try:
        predictions = pd.read_csv('outputs/results/xgboost_predictions.csv')
        segment_analysis = pd.read_csv('outputs/results/segment_analysis.csv')
        feature_importance = pd.read_csv('outputs/results/xgboost_feature_importance.csv')
        return predictions, segment_analysis, feature_importance
    except FileNotFoundError as e:
        st.error(f"Data files not found: {e}")
        st.info("Make sure CSV files are in 'outputs/results/' directory")
        return None, None, None

predictions, segment_analysis, feature_importance = load_data()

if predictions is None:
    st.stop()

# ============================================================================
# HEADER
# ============================================================================

st.title("📊 Credit Risk Default Prediction Dashboard")
st.markdown("XGBoost Model · ROC-AUC: **0.753** · 307,511 Applications")
st.markdown("---")

# ============================================================================
# KPI ROW
# ============================================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Applications", f"{len(predictions):,}")

with col2:
    default_rate = predictions['actual_default'].mean()
    st.metric("Default Rate", f"{default_rate:.2%}")

with col3:
    st.metric("Model ROC-AUC", "0.753")

with col4:
    if 'risk_flag' in predictions.columns:
        flagged_rate = predictions['risk_flag'].mean()
        st.metric("Flagged as Risky", f"{flagged_rate:.1%}")
    else:
        st.metric("Flagged as Risky", "N/A")

st.markdown("---")

# ============================================================================
# TWO COLUMN LAYOUT
# ============================================================================

col_left, col_right = st.columns(2)

# ============================================================================
# LEFT: RISK BUCKET DISTRIBUTION
# ============================================================================

with col_left:
    st.subheader("🎯 Risk Bucket Distribution")
    
    # Create risk buckets if not already present
    if 'risk_bucket' not in predictions.columns:
        predictions['risk_bucket'] = pd.cut(
            predictions['default_probability'],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
    
    risk_counts = predictions['risk_bucket'].value_counts().reset_index()
    risk_counts.columns = ['Risk Bucket', 'Count']
    
    # Order buckets correctly
    bucket_order = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    risk_counts['Risk Bucket'] = pd.Categorical(risk_counts['Risk Bucket'], categories=bucket_order, ordered=True)
    risk_counts = risk_counts.sort_values('Risk Bucket')
    
    fig = px.bar(
        risk_counts, 
        x='Risk Bucket', 
        y='Count',
        color='Risk Bucket',
        color_discrete_sequence=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad'],
        title="Number of Applications by Risk Bucket"
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# RIGHT: DEFAULT RATE BY SEGMENT
# ============================================================================

with col_right:
    st.subheader("📈 Default Rate by Segment")
    
    if segment_analysis is not None:
        segment_fig = px.bar(
            segment_analysis,
            x='Segment',
            y='Default_Rate',
            color='Default_Rate',
            color_continuous_scale='Reds',
            title="Actual Default Rates from Historical Data"
        )
        segment_fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(segment_fig, use_container_width=True)
    else:
        st.info("Segment analysis data not available")

st.markdown("---")

# ============================================================================
# SECOND ROW: PROBABILITY DISTRIBUTION + FEATURE IMPORTANCE
# ============================================================================

col_left2, col_right2 = st.columns(2)

# ============================================================================
# LEFT: PROBABILITY DISTRIBUTION
# ============================================================================

with col_left2:
    st.subheader("📊 Default Probability Distribution")
    
    fig_hist = px.histogram(
        predictions,
        x='default_probability',
        nbins=50,
        title="Distribution of Predicted Default Probabilities",
        labels={'default_probability': 'Default Probability', 'count': 'Number of Applications'},
        color_discrete_sequence=['#3498db']
    )
    fig_hist.add_vline(x=0.4, line_dash="dash", line_color="red", 
                       annotation_text="Risk Threshold (80% Recall)")
    st.plotly_chart(fig_hist, use_container_width=True)

# ============================================================================
# RIGHT: FEATURE IMPORTANCE
# ============================================================================

with col_right2:
    st.subheader("🔍 Top Predictors of Default")
    
    if feature_importance is not None and len(feature_importance) > 0:
        top_features = feature_importance.head(10).copy()
        
        fig_importance = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title="Feature Importance (XGBoost)",
            labels={'importance': 'Importance Score', 'feature': ''},
            color='importance',
            color_continuous_scale='Blues'
        )
        fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_importance, use_container_width=True)
    else:
        st.info("Feature importance data not available")

st.markdown("---")

# ============================================================================
# THIRD ROW: SEGMENT DETAILS TABLE
# ============================================================================

st.subheader("📋 Risk Segment Details")

if segment_analysis is not None:
    display_df = segment_analysis.copy()
    if 'Default_Rate' in display_df.columns:
        display_df['Default_Rate'] = display_df['Default_Rate'].apply(lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else x)
    if 'Sample_Size' in display_df.columns:
        display_df['Sample_Size'] = display_df['Sample_Size'].apply(lambda x: f"{x:,}" if isinstance(x, (int, float)) else x)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("Segment analysis data not available")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <b>Model:</b> XGBoost (tuned) | <b>ROC-AUC:</b> 0.753 | <b>Data:</b> Home Credit Default Risk (307,511 applications)
    </div>
    """,
    unsafe_allow_html=True
)