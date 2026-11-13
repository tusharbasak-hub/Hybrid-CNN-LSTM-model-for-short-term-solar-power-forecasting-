import random
import numpy as np
import torch
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def set_seed(seed=42):
    """Sets random seeds across random, numpy, and torch for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Returns PyTorch device (CUDA or CPU)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def evaluate_metrics(y_true, y_pred):
    """Computes RMSE, MAE, and R^2 score."""
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}
