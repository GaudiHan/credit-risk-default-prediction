# create_dashboard_data.py

# 0. Initialisation 

from pathlib import Path
import pandas as pd
import numpy as np

script_dir = Path(__file__).parent 
script_dir / "../outputs/results/feature_importance.csv"
xgboost_pred = pd.read_csv((script_dir / "../outputs/results/xgboost_predictions.csv").resolve())
segment_analysis = pd.read_csv((script_dir / "../outputs/results/segment_analysis.csv").resolve())
feature_importance = pd.read_csv((script_dir / "../outputs/results/feature_importance.csv").resolve())



# 1. Prediction Data (risk scoring visualizations)

# Add risk buckets to predictions
xgboost_pred['risk_bucket'] = pd.cut(
    xgboost_pred['default_probability'],
    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
)

# Create binary risk flag at 0.4 threshold (80% recall)
xgboost_pred['risk_flag_80recall'] = (xgboost_pred['default_probability'] > 0.4).astype(int)

# Save for Power BI
xgboost_pred.to_csv((script_dir / "../dashboard/predictions.csv").resolve(), index=False)
print(f">>> Predictions saved: {xgboost_pred.shape}")



# 2. Segment Analysis Data (KPI cards and segment comparisons)

# Clean up segment names for display
segment_analysis['Segment_Display'] = segment_analysis['Segment'].replace({
    'Low Credit Score (<0.3)': 'Low Credit Score (<0.3)',
    'High Credit Score (>0.6)': 'High Credit Score (>0.6)',
    'Short Employment (<1 year)': 'Short Employment (<1 year)',
    'Long Employment (>5 years)': 'Long Employment (>5 years)',
    'Medium DTI (2-3.5x)': 'Medium DTI (2-3.5x)',
    'Low DTI (≤2x)': 'Low DTI (≤2x)',
    'High DTI (>3.5x)': 'High DTI (>3.5x)'
})

segment_analysis.to_csv((script_dir / "../dashboard/segment_analysis.csv").resolve(), index=False)
print(">>> Segment analysis saved")



# 3. Feature Importance Data (model insights page)

# Keep top 15 features for readability
feature_importance_top = feature_importance.head(15).copy()
feature_importance_top.to_csv((script_dir / "../dashboard/feature_importance.csv").resolve(), index=False)
print(">>> Feature importance saved")



# 4. Summary Table (KPI cards)

summary_data = pd.DataFrame({
    'Metric': [
        'Total Applications',
        'Default Rate',
        'Model ROC-AUC',
        'Low Credit Score Default Rate',
        'High Credit Score Default Rate',
        'Short Employment Default Rate',
        'Long Employment Default Rate',
        'Peak DTI Default Rate',
        'Precision at 80% Recall',
        'Risk Threshold (80% Recall)'
    ],
    'Value': [
        '307,511',
        '8.07%',
        '75.3%',
        '15.88%',
        '4.56%',
        '10.98%',
        '6.41%',
        '8.81%',
        '13.6%',
        '0.40'
    ]
})
summary_data.to_csv((script_dir / "../dashboard/summary.csv").resolve(), index=False)
print(">>> Summary data saved")

print("\n>>> All data files ready for Power BI!")
print("    Location: ../dashboard/")