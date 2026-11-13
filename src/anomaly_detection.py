import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_and_clean_anomalies(df, contamination=0.025, random_state=42):
    """
    Fits an unsupervised Isolation Forest model to flag outliers in power_raw,
    then replaces detected anomalies using linear temporal interpolation.
    Produces the 'power_cleaned' column.
    """
    df = df.copy()
    
    # Feature matrix for Isolation Forest
    X_if = df[['power_raw', 'hour_sin', 'hour_cos']].values
    
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=100)
    preds = iso_forest.fit_predict(X_if) # -1 for outlier, 1 for inlier
    
    df['detected_outlier'] = (preds == -1).astype(int)
    
    # Clean via linear interpolation
    df['power_cleaned'] = df['power_raw'].copy()
    df.loc[df['detected_outlier'] == 1, 'power_cleaned'] = np.nan
    df['power_cleaned'] = df['power_cleaned'].interpolate(method='linear').bfill().ffill()
    
    # Physical clip to [0, capacity]
    capacity = df['capacity_mw'].iloc[0]
    df['power_cleaned'] = np.clip(df['power_cleaned'], 0, capacity)
    
    # Detection stats vs ground truth injected
    true_anomalies = (df['is_anomaly'] > 0).astype(int)
    detected = df['detected_outlier']
    tp = ((true_anomalies == 1) & (detected == 1)).sum()
    fp = ((true_anomalies == 0) & (detected == 1)).sum()
    fn = ((true_anomalies == 1) & (detected == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"[AnomalyDetection] Isolation Forest Evaluation:")
    print(f"  - Flagged Outliers: {detected.sum()} rows")
    print(f"  - Precision:        {precision:.4f}")
    print(f"  - Recall:           {recall:.4f}")
    # Oracle Cleaned signal: interpolate strictly using ground-truth injected anomaly indices
    if 'is_anomaly' in df.columns:
        df['power_oracle_cleaned'] = df['power_raw'].copy()
        df.loc[df['is_anomaly'] > 0, 'power_oracle_cleaned'] = np.nan
        df['power_oracle_cleaned'] = df['power_oracle_cleaned'].interpolate(method='linear').bfill().ffill()
        df['power_oracle_cleaned'] = np.clip(df['power_oracle_cleaned'], 0, capacity)
        print("  - Reconstructed 'power_oracle_cleaned' signal via ground-truth oracle interpolation.")

    return df


def clean_with_oracle(df):
    """
    Replaces corrupted readings strictly using the ground truth 'is_anomaly' column
    and applies linear temporal interpolation. Serves as Run B (Oracle Imputation upper bound).
    """
    df = df.copy()
    capacity = df['capacity_mw'].iloc[0]
    df['power_oracle_cleaned'] = df['power_raw'].copy()
    df.loc[df['is_anomaly'] > 0, 'power_oracle_cleaned'] = np.nan
    df['power_oracle_cleaned'] = df['power_oracle_cleaned'].interpolate(method='linear').bfill().ffill()
    df['power_oracle_cleaned'] = np.clip(df['power_oracle_cleaned'], 0, capacity)
    return df
