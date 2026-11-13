import numpy as np
import pandas as pd


def inject_synthetic_anomalies(df, anomaly_frac=0.02, random_state=42):
    """
    Synthetically corrupts ~2% of rows using three realistic PV fault types:
    1. Dropout: force power reading to zero during daytime (dead sensor / disconnect).
    2. Spike: multiply reading by 3x-5x (glitch / electrical transient).
    3. Drift: add a fixed positive offset to a block of consecutive readings (calibration drift).
    """
    np.random.seed(random_state)
    df = df.copy()
    df['power_raw'] = df['power_mw'].astype(float)
    df['is_anomaly'] = 0

    n_rows = len(df)
    n_anomalies = int(n_rows * anomaly_frac)
    n_per_type = n_anomalies // 3

    capacity = df['capacity_mw'].iloc[0]

    # Type 1: Dropouts during daylight hours (power_mw > 5 MW)
    daylight_idx = df[df['power_mw'] > 5.0].index.values
    if len(daylight_idx) > 0:
        dropout_idx = np.random.choice(daylight_idx, size=min(n_per_type, len(daylight_idx)), replace=False)
        df.loc[dropout_idx, 'power_raw'] = 0.0
        df.loc[dropout_idx, 'is_anomaly'] = 1

    # Type 2: Spikes (glitches during non-zero output)
    spike_candidates = df[(df['power_mw'] > 2.0) & (df['is_anomaly'] == 0)].index.values
    if len(spike_candidates) > 0:
        spike_idx = np.random.choice(spike_candidates, size=min(n_per_type, len(spike_candidates)), replace=False)
        multiplier = np.random.uniform(3.0, 5.0, size=len(spike_idx))
        df.loc[spike_idx, 'power_raw'] = np.clip(df.loc[spike_idx, 'power_raw'] * multiplier, 0, capacity * 1.5)
        df.loc[spike_idx, 'is_anomaly'] = 2

    # Type 3: Calibration Drift (consecutive offset blocks)
    clean_candidates = df[df['is_anomaly'] == 0].index.values
    drift_count = 0
    while drift_count < n_per_type and len(clean_candidates) > 10:
        start_i = np.random.choice(clean_candidates)
        block_len = np.random.randint(4, 12)
        block_idx = np.arange(start_i, min(start_i + block_len, n_rows))
        offset = np.random.uniform(0.15 * capacity, 0.35 * capacity)
        df.loc[block_idx, 'power_raw'] += offset
        df.loc[block_idx, 'is_anomaly'] = 3
        drift_count += len(block_idx)
        clean_candidates = np.setdiff1d(clean_candidates, block_idx)

    total_injected = (df['is_anomaly'] > 0).sum()
    print(f"[AnomalyInjection] Injected faults into {total_injected} rows ({total_injected/n_rows*100:.2f}% of series):")
    print(f"  - Daytime Dropouts:   {(df['is_anomaly'] == 1).sum()} rows")
    print(f"  - Glitch Spikes:      {(df['is_anomaly'] == 2).sum()} rows")
    print(f"  - Calibration Drift:  {(df['is_anomaly'] == 3).sum()} rows")

    return df
