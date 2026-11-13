import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_publication_figures(results, predictions, y_test_actual, df_raw, summary_stats=None, output_dir='output'):
    """
    Generates 4 high-resolution publication-quality PNG figures for paper submission.
    Supports multi-seed summary statistics (mean ± std) and error bars on Figure 2.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    desired_order = [
        'CNN-LSTM (Pristine Reference)',
        'CNN-LSTM (Oracle Cleaned)',
        'CNN-LSTM (Cleaned - Proposed)',
        'Plain LSTM',
        'ARIMA / AutoReg',
        'Persistence',
        'CNN-LSTM (Raw Data)'
    ]

    # 1. Comparison Table CSV Export
    if summary_stats is not None:
        stats_df = pd.DataFrame(summary_stats).T
        raw_rmse = stats_df.loc['CNN-LSTM (Raw Data)', 'RMSE_mean'] if 'CNN-LSTM (Raw Data)' in stats_df.index else None
        pris_rmse = stats_df.loc['CNN-LSTM (Pristine Reference)', 'RMSE_mean'] if 'CNN-LSTM (Pristine Reference)' in stats_df.index else None
        denom = raw_rmse - pris_rmse if (raw_rmse is not None and pris_rmse is not None and raw_rmse != pris_rmse) else 1.0

        stats_df['% RMSE Impr. vs Raw'] = [
            f"{((raw_rmse - r) / raw_rmse) * 100:+.2f}%" if raw_rmse is not None else "0.00%"
            for r in stats_df['RMSE_mean']
        ]
        stats_df['% Signal Recovery'] = [
            f"{min(100.0, max(0.0, ((raw_rmse - r) / denom) * 100)):.2f}%" if raw_rmse is not None else "0.00%"
            for r in stats_df['RMSE_mean']
        ]

        display_rows = {}
        for model in stats_df.index:
            r_m, r_s = stats_df.loc[model, 'RMSE_mean'], stats_df.loc[model, 'RMSE_std']
            m_m, m_s = stats_df.loc[model, 'MAE_mean'], stats_df.loc[model, 'MAE_std']
            r2_m, r2_s = stats_df.loc[model, 'R2_mean'], stats_df.loc[model, 'R2_std']

            if r_s == 0.0:  # Deterministic baseline
                rmse_str = f"{r_m:.4f} (fixed)"
                mae_str = f"{m_m:.4f} (fixed)"
                r2_str = f"{r2_m:.4f} (fixed)"
            else:
                rmse_str = f"{r_m:.4f} ± {r_s:.4f}"
                mae_str = f"{m_m:.4f} ± {m_s:.4f}"
                r2_str = f"{r2_m:.4f} ± {r2_s:.4f}"

            display_rows[model] = {
                'RMSE (MW)': rmse_str,
                'MAE (MW)': mae_str,
                'R²': r2_str,
                '% RMSE Impr. vs Raw': stats_df.loc[model, '% RMSE Impr. vs Raw'],
                '% Signal Recovery': stats_df.loc[model, '% Signal Recovery']
            }

        present_order = [m for m in desired_order if m in display_rows] + [m for m in display_rows if m not in desired_order]
        res_df = pd.DataFrame(display_rows).T.loc[present_order]
    else:
        res_df = pd.DataFrame(results).T.round(4)
        raw_rmse = res_df.loc['CNN-LSTM (Raw Data)', 'RMSE'] if 'CNN-LSTM (Raw Data)' in res_df.index else None
        
        if raw_rmse is not None:
            res_df['% RMSE Impr. vs Raw'] = [f"{((raw_rmse - r)/raw_rmse)*100:+.2f}%" for r in res_df['RMSE']]
        
        if 'CNN-LSTM (Pristine Reference)' in res_df.index and raw_rmse is not None:
            pris_rmse = res_df.loc['CNN-LSTM (Pristine Reference)', 'RMSE']
            denom = raw_rmse - pris_rmse if (raw_rmse - pris_rmse) != 0 else 1.0
            res_df['% Signal Recovery'] = [f"{min(100.0, max(0.0, ((raw_rmse - r)/denom)*100)):.2f}%" for r in res_df['RMSE']]

        present_order = [m for m in desired_order if m in res_df.index] + [m for m in res_df.index if m not in desired_order]
        res_df = res_df.loc[present_order]

    table_path = os.path.join(output_dir, 'model_comparison_results.csv')
    res_df.to_csv(table_path)
    print("\n=======================================================")
    print("FINAL MODEL COMPARISON RESULTS TABLE:")
    print(res_df.to_string())
    print("=======================================================")
    print(f"[Visualization] Saved comparison table to: {table_path}")

    # Figure 1: Predicted vs Actual Forecast (Test Window Segment - Canonical Seed 42 Trajectory)
    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=300)
    sample_len = 288  # 24 hours of 5-min data
    start_idx = 100
    end_idx = start_idx + sample_len

    ax.plot(y_test_actual[start_idx:end_idx], label='Actual Power (Ground Truth)', color='black', linewidth=2.2, alpha=0.9)
    if 'CNN-LSTM (Pristine Reference)' in predictions:
        ax.plot(predictions['CNN-LSTM (Pristine Reference)'][start_idx:end_idx], label='CNN-LSTM Pristine Ceiling (Run A)', color='#6a3d9a', linestyle='-.', linewidth=1.8, alpha=0.85)
    if 'CNN-LSTM (Oracle Cleaned)' in predictions:
        ax.plot(predictions['CNN-LSTM (Oracle Cleaned)'][start_idx:end_idx], label='CNN-LSTM Oracle Cleaned (Run B)', color='#17becf', linestyle='--', linewidth=1.6, alpha=0.85)
    if 'CNN-LSTM (Cleaned - Proposed)' in predictions:
        ax.plot(predictions['CNN-LSTM (Cleaned - Proposed)'][start_idx:end_idx], label='CNN-LSTM Proposed (Cleaned)', color='#1f77b4', linewidth=2.0)
    if 'CNN-LSTM (Raw Data)' in predictions:
        ax.plot(predictions['CNN-LSTM (Raw Data)'][start_idx:end_idx], label='CNN-LSTM Ablation (Raw Data)', color='#d62728', linestyle=':', linewidth=1.5)
    if 'Plain LSTM' in predictions:
        ax.plot(predictions['Plain LSTM'][start_idx:end_idx], label='Plain LSTM', color='#2ca02c', linestyle=':', linewidth=1.3)

    ax.set_title('Short-Term Solar Power Forecast vs. Actual Output (24-Hour Test Window, H=6 / 30-Min Ahead, Seed 42)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Time Step (5-Minute Intervals)', fontsize=11)
    ax.set_ylabel('Power Output (MW)', fontsize=11)
    ax.set_ylim(-2, 58)
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.92, fontsize=9.5)
    fig.tight_layout()
    fig1_path = os.path.join(output_dir, 'figure1_predicted_vs_actual.png')
    fig.savefig(fig1_path)
    plt.close()
    print(f"[Visualization] Saved Figure 1 to: {fig1_path}")

    # Figure 2: RMSE & MAE Comparison Bar Chart (with Multi-Seed Error Bars)
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)
    models = list(res_df.index)

    if summary_stats is not None:
        rmses = [summary_stats[m]['RMSE_mean'] for m in models]
        rmses_std = [summary_stats[m]['RMSE_std'] for m in models]
        maes = [summary_stats[m]['MAE_mean'] for m in models]
        maes_std = [summary_stats[m]['MAE_std'] for m in models]
    else:
        rmses = [res_df.loc[m, 'RMSE'] for m in models]
        rmses_std = [0.0] * len(models)
        maes = [res_df.loc[m, 'MAE'] for m in models]
        maes_std = [0.0] * len(models)

    x = np.arange(len(models))
    width = 0.35

    rects1 = ax.bar(
        x - width/2, rmses, width, yerr=rmses_std, capsize=4,
        label='RMSE (MW)', color='#1f77b4', edgecolor='black', linewidth=0.5,
        error_kw={'elinewidth': 1.2, 'ecolor': '#222222', 'capthick': 1.2}
    )
    rects2 = ax.bar(
        x + width/2, maes, width, yerr=maes_std, capsize=4,
        label='MAE (MW)', color='#ff7f0e', edgecolor='black', linewidth=0.5,
        error_kw={'elinewidth': 1.2, 'ecolor': '#222222', 'capthick': 1.2}
    )

    chart_title = 'Model Performance Comparison (H=6 / 30-Min Ahead, 5 Seeds: Mean ± Std, Lower is Better)' if summary_stats else 'Model Performance Comparison across Metrics (H=6 / 30-Min Ahead, Lower is Better)'
    ax.set_title(chart_title, fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Error Metric (MW)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha='right', fontsize=9.5)
    ax.legend(loc='upper left', frameon=True, fontsize=10)

    for i, bar in enumerate(rects1):
        height = bar.get_height()
        err = rmses_std[i] if summary_stats else 0.0
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height + err),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5)
    for i, bar in enumerate(rects2):
        height = bar.get_height()
        err = maes_std[i] if summary_stats else 0.0
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height + err),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5)

    fig.tight_layout()
    fig2_path = os.path.join(output_dir, 'figure2_rmse_bar_chart.png')
    fig.savefig(fig2_path)
    plt.close()
    print(f"[Visualization] Saved Figure 2 to: {fig2_path}")

    # Figure 3: Residual Error Distribution Histogram (Canonical Seed 42 Trajectory)
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)
    proposed_errors = predictions['CNN-LSTM (Cleaned - Proposed)'] - y_test_actual
    raw_errors = predictions['CNN-LSTM (Raw Data)'] - y_test_actual

    sns.histplot(raw_errors, color='#d62728', label='CNN-LSTM (Raw Data)', kde=True, ax=ax, bins=50, stat="density", alpha=0.35)
    sns.histplot(proposed_errors, color='#1f77b4', label='CNN-LSTM Proposed (Cleaned)', kde=True, ax=ax, bins=50, stat="density", alpha=0.55)
    if 'CNN-LSTM (Pristine Reference)' in predictions:
        pris_errors = predictions['CNN-LSTM (Pristine Reference)'] - y_test_actual
        sns.histplot(pris_errors, color='#6a3d9a', label='CNN-LSTM Pristine Ceiling (Run A)', kde=True, ax=ax, bins=50, stat="density", alpha=0.4)

    ax.set_title('Residual Forecast Error Distribution (Residual = Predicted - Actual, Seed 42)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Forecast Residual Error (MW)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(loc='upper right', frameon=True, fontsize=9.5)
    fig.tight_layout()
    fig3_path = os.path.join(output_dir, 'figure3_error_distribution.png')
    fig.savefig(fig3_path)
    plt.close()
    print(f"[Visualization] Saved Figure 3 to: {fig3_path}")

    # Figure 4: Anomaly Detection Demonstration (Raw vs Cleaned telemetry zoom)
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    anomaly_window = df_raw.iloc[864:1152].copy()  # 288 points (24-hour cycle)

    ax.plot(anomaly_window['dt'], anomaly_window['power_mw'], label='Ground Truth Power', color='gray', linestyle=':', linewidth=1.5)
    ax.plot(anomaly_window['dt'], anomaly_window['power_raw'], label='Synthetically Corrupted Telemetry', color='#d62728', alpha=0.8, linewidth=1.5)
    ax.plot(anomaly_window['dt'], anomaly_window['power_cleaned'], label='Isolation Forest Cleaned Signal', color='#1f77b4', linewidth=2.0)

    flagged = anomaly_window[anomaly_window['detected_outlier'] == 1]
    if not flagged.empty:
        ax.scatter(flagged['dt'], flagged['power_raw'], color='red', marker='x', s=60, label='Isolation Forest Flagged Anomalies', zorder=5)

    ax.set_title('Demonstration of Isolation Forest Anomaly Detection and Signal Reconstruction', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Local Time', fontsize=11)
    ax.set_ylabel('Power Output (MW)', fontsize=11)
    ax.legend(loc='upper right', frameon=True)
    fig.tight_layout()
    fig4_path = os.path.join(output_dir, 'figure4_anomaly_detection_demo.png')
    fig.savefig(fig4_path)
    plt.close()
    print(f"[Visualization] Saved Figure 4 to: {fig4_path}")
