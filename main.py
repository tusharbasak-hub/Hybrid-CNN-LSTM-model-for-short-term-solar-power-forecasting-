"""
Main Pipeline Entry Point - Solar Power Forecasting (ICSPER 2026)
Hybrid CNN-LSTM with Anomaly-Aware Preprocessing
"""

import os
from src.utils import set_seed, get_device
from src.data_loader import load_site_data, split_and_save_data
from src.feature_engineering import engineer_time_features
from src.anomaly_injection import inject_synthetic_anomalies
from src.anomaly_detection import detect_and_clean_anomalies
from src.windowing import prepare_windowed_data
from src.training import run_all_models
from src.visualization import generate_publication_figures


def run_pipeline(
    db_path=r"C:\Users\TUSHAR BASAK\Downloads\al-pv-2006.db",
    lat=30.45,
    lon=-88.25,
    site_type='UPV',
    resolution='5_Min',
    epochs=18,
    horizon=6,
    data_dir='data/processed',
    output_dir='output'
):
    print("==========================================================================")
    print("SOLAR POWER FORECASTING PIPELINE (ICSPER 2026)")
    print(f"Architecture: 1D CNN-LSTM | Anomaly Mitigation: Isolation Forest | Horizon={horizon} ({horizon*5} min)")
    print("==========================================================================")
    
    # 0. Setup reproducibility and device
    set_seed(42)
    device = get_device()
    print(f"[System] Active compute device: {device}")

    # 1. Load Data & Select Site
    df = load_site_data(db_path, lat=lat, lon=lon, site_type=site_type, resolution=resolution)

    # 2. Feature Engineering (Cyclical Encodings)
    df = engineer_time_features(df)

    # 3. Synthetic Anomaly Injection (~2%)
    df = inject_synthetic_anomalies(df, anomaly_frac=0.02)

    # 4. Anomaly Detection & Signal Cleaning (Isolation Forest)
    df = detect_and_clean_anomalies(df, contamination=0.025)

    # 5. Export Train / Val / Test Datasets to Folder
    print("\n--- Exporting Train, Val, and Test Splits ---")
    train_df, val_df, test_df = split_and_save_data(df, output_dir=data_dir)

    # 6. Windowing (Past 24 steps -> Next H steps ahead)
    print(f"\n--- Generating Sequence Windows (Window=24, Horizon={horizon}) ---")
    print("Preparing sequences for Reference Ceiling (Run A: Original Clean Data)...")
    data_pristine = prepare_windowed_data(df, target_col='power_mw', window_size=24, horizon=horizon)

    print("Preparing sequences for Imputation Upper Bound (Run B: Oracle Ground-Truth Interpolation)...")
    data_oracle = prepare_windowed_data(df, target_col='power_oracle_cleaned', window_size=24, horizon=horizon)

    print("Preparing sequences for Proposed Pipeline (Cleaned Data via Isolation Forest)...")
    data_cleaned = prepare_windowed_data(df, target_col='power_cleaned', window_size=24, horizon=horizon)
    
    print("Preparing sequences for Ablation Baseline (Raw Corrupted Data)...")
    data_raw = prepare_windowed_data(df, target_col='power_raw', window_size=24, horizon=horizon)

    # 7. Model Training and Benchmarking
    print("\n--- Training and Evaluating 7 Benchmark & Ablation Conditions across 5 Seeds ---")
    results, predictions, y_test_actual, summary_stats = run_all_models(
        data_cleaned, data_raw, 
        data_pristine=data_pristine, data_oracle=data_oracle, 
        epochs=epochs
    )

    # 8. Generate Publication Plots and Summary Table
    print("\n--- Generating Publication Figures (with Multi-Seed Error Bars) ---")
    generate_publication_figures(results, predictions, y_test_actual, df, summary_stats=summary_stats, output_dir=output_dir)

    print("\n[SUCCESS] Pipeline executed successfully. All models, tables, and figures generated.")
    return results, summary_stats


if __name__ == '__main__':
    run_pipeline()
