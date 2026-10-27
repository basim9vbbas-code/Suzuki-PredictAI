import pandas as pd
import numpy as np
import os

def load_and_process_fuel_data(file_path="pakistan_fuel_prices.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fuel dataset not found at {file_path}")
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    df["petrol_mom_change_pct"] = df["petrol_pkr_per_liter"].pct_change() * 100
    df["diesel_mom_change_pct"] = df["diesel_pkr_per_liter"].pct_change() * 100
    df = df.fillna(0)
    print(f"Successfully loaded {len(df)} records of Karachi fuel pricing data.")
    return df

if __name__ == "__main__":
    df_fuel = load_and_process_fuel_data()
    df_fuel.to_csv("processed_fuel_features.csv", index=False)
    print("Exported processed fuel features to 'processed_fuel_features.csv'")
