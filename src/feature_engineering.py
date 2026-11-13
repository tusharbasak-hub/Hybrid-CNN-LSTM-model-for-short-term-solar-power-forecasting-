import numpy as np
import pandas as pd


def engineer_time_features(df):
    """
    Derives sine/cosine encodings of hour-of-day and day-of-year from local_time.
    These continuous cyclical proxies provide solar geometric orientation.
    """
    df = df.copy()
    hour = df['dt'].dt.hour + df['dt'].dt.minute / 60.0
    day = df['dt'].dt.dayofyear

    # Cyclical sine / cosine encodings
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    df['day_sin'] = np.sin(2 * np.pi * day / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * day / 365.25)

    print("[FeatureEngineering] Computed cyclical encodings: hour_sin, hour_cos, day_sin, day_cos")
    return df
