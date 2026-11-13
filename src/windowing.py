import numpy as np
from sklearn.preprocessing import MinMaxScaler


def prepare_windowed_data(df, target_col='power_cleaned', window_size=24, horizon=6, train_ratio=0.7, val_ratio=0.15):
    """
    Converts multivariate time series into overlapping sliding sequence windows.
    Past W steps -> Predict next H steps ahead.
    Includes feature scaling fitted ONLY on training split to prevent leakage.
    """
    feature_cols = [target_col, 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    data = df[feature_cols].values

    # Train / Val / Test Split
    n_samples = len(data)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(data[:train_end])
    val_scaled = scaler.transform(data[train_end:val_end])
    test_scaled = scaler.transform(data[val_end:])

    scaled_data = np.vstack([train_scaled, val_scaled, test_scaled])

    def create_sequences(arr):
        X, y = [], []
        for i in range(len(arr) - window_size - horizon + 1):
            X.append(arr[i : i + window_size])
            y.append(arr[i + window_size + horizon - 1, 0]) # target is power (index 0)
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data)

    # Adjust indices for sequence boundaries
    train_w_end = train_end - window_size - horizon + 1
    val_w_end = val_end - window_size - horizon + 1

    X_train, y_train = X[:train_w_end], y[:train_w_end]
    X_val, y_val = X[train_w_end:val_w_end], y[train_w_end:val_w_end]
    X_test, y_test = X[val_w_end:], y[val_w_end:]

    # Target denormalization stats (min & max for power column)
    power_min = scaler.data_min_[0]
    power_max = scaler.data_max_[0]

    print(f"[Windowing] Constructed sequences (Window={window_size}, Horizon={horizon}, Target={target_col}):")
    print(f"  - Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  - Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"  - Test:  X={X_test.shape}, y={y_test.shape}")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'scaler': scaler,
        'power_min': power_min,
        'power_max': power_max,
        'raw_df': df,
        'val_w_end': val_w_end,
        'feature_cols': feature_cols
    }
