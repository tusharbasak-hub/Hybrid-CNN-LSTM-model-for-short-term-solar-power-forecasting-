import os
import sqlite3
import pandas as pd


def load_site_data(db_path, lat=30.45, lon=-88.25, site_type='UPV', resolution='5_Min'):
    """
    Connects to SQLite DB and queries data for ONE specific site and resolution.
    Avoids mixing physically different locations or sampling frequencies.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"SQLite database file not found at: {db_path}")

    conn = sqlite3.connect(db_path)
    
    query = f"""
        SELECT id, latitude, longitude, year, site_type, capacity_mw, resolution, row_index, local_time, power_mw
        FROM solar_data
        WHERE latitude = {lat} AND longitude = {lon} AND site_type = '{site_type}' AND resolution = '{resolution}'
        ORDER BY row_index ASC
    """
    print(f"[DataLoader] Querying site: Lat={lat}, Lon={lon}, Type={site_type}, Res={resolution}...")
    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        raise ValueError(f"No records found for site Lat={lat}, Lon={lon}, Type={site_type}, Res={resolution}")

    df['dt'] = pd.to_datetime(df['local_time'], format='%m/%d/%y %H:%M')
    df = df.sort_values('dt').reset_index(drop=True)
    
    print(f"[DataLoader] Loaded {len(df)} rows ({df['dt'].min()} to {df['dt'].max()})")
    print(f"[DataLoader] Plant Capacity: {df['capacity_mw'].iloc[0]} MW | Site Type: {df['site_type'].iloc[0]}")
    return df


def split_and_save_data(df, output_dir='data/processed', train_ratio=0.7, val_ratio=0.15):
    """
    Splits the continuous time-series into train, val, and test sets,
    and saves them as CSV files in the designated folder.
    """
    os.makedirs(output_dir, exist_ok=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[DataLoader] Saved data splits to '{output_dir}':")
    print(f"  - Train: {len(train_df)} rows -> {train_path}")
    print(f"  - Val:   {len(val_df)} rows -> {val_path}")
    print(f"  - Test:  {len(test_df)} rows -> {test_path}")

    return train_df, val_df, test_df
