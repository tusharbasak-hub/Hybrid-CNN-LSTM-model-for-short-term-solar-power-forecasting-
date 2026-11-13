import torch
import torch.nn as nn
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


# 1. Plain LSTM Deep Learning Baseline
class PlainLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.1):
        super(PlainLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Last time step output
        return self.fc(out)


# 2. Proposed Hybrid CNN-LSTM Architecture
class HybridCNNLSTM(nn.Module):
    def __init__(self, input_dim, conv_filters=64, kernel_size=3, hidden_dim=64, num_layers=2, dropout=0.1):
        super(HybridCNNLSTM, self).__init__()
        # 1D CNN expects shape (Batch, Channels, Length)
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=conv_filters, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(conv_filters),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(
            conv_filters, hidden_dim, num_layers=num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Features) -> transpose to (Batch, Features, Seq_Len) for Conv1d
        x = x.transpose(1, 2)
        conv_out = self.conv1d(x)
        # transpose back to (Batch, Seq_Len, Filters) for LSTM
        conv_out = conv_out.transpose(1, 2)
        lstm_out, _ = self.lstm(conv_out)
        out = lstm_out[:, -1, :]
        return self.fc(out)


# 3. Naive Persistence Forecaster
def predict_persistence(X_test, denorm_func):
    """Persistence baseline: predicts target equals the last known observation in window."""
    y_pred_norm = X_test[:, -1, 0]
    return denorm_func(y_pred_norm)


# 4. Statistical Autoregressive / ARIMA Baseline
def predict_arima(train_series, X_test, denorm_func, X_train=None, y_train=None):
    """
    Fits an Autoregressive / ARIMA model predicting target at horizon H from recent lags (t, t-1).
    """
    import numpy as np
    from sklearn.linear_model import Ridge
    try:
        if X_train is not None and y_train is not None:
            reg = Ridge(alpha=1.0).fit(X_train[:, [-1, -2], 0], y_train)
            y_pred_norm = reg.predict(X_test[:, [-1, -2], 0])
            y_pred = denorm_func(y_pred_norm)
            return np.clip(y_pred, 0, np.max(train_series))
        else:
            from statsmodels.tsa.ar_model import AutoReg
            ar_model = AutoReg(train_series, lags=[1, 2, 288]).fit()
            c = ar_model.params[0]
            p1 = ar_model.params[1]
            p2 = ar_model.params[2]
            y_pred = c + p1 * denorm_func(X_test[:, -1, 0]) + p2 * denorm_func(X_test[:, -2, 0])
            return np.clip(y_pred, 0, np.max(train_series))
    except Exception as e:
        print(f"[ARIMA Baseline] Exception ({e}), applying persistence fallback.")
        return denorm_func(X_test[:, -1, 0])

