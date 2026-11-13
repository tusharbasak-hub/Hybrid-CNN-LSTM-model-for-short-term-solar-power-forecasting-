import argparse
from main import run_pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Solar Power Forecasting Pipeline Runner (ICSPER 2026)")
    parser.add_argument('--db-path', type=str, default=r"C:\Users\TUSHAR BASAK\Downloads\al-pv-2006.db", help="Path to SQLite database file")
    parser.add_argument('--lat', type=float, default=30.45, help="Site latitude")
    parser.add_argument('--lon', type=float, default=-88.25, help="Site longitude")
    parser.add_argument('--site-type', type=str, default='UPV', choices=['UPV', 'DPV'], help="Site type (Utility PV or Distributed PV)")
    parser.add_argument('--resolution', type=str, default='5_Min', choices=['5_Min', '60_Min'], help="Data resolution")
    parser.add_argument('--epochs', type=int, default=18, help="Number of deep learning training epochs")
    parser.add_argument('--horizon', type=int, default=6, help="Forecast horizon in steps (default: 6 steps = 30 min ahead)")
    parser.add_argument('--data-dir', type=str, default='data/processed', help="Directory to save train/val/test splits")
    parser.add_argument('--output-dir', type=str, default='output', help="Directory to save figures and results")
    
    args = parser.parse_args()
    
    run_pipeline(
        db_path=args.db_path,
        lat=args.lat,
        lon=args.lon,
        site_type=args.site_type,
        resolution=args.resolution,
        epochs=args.epochs,
        horizon=args.horizon,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
