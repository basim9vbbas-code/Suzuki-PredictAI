import pandas as pd
import numpy as np
import os

def load_and_process_kibor_data(file_path="sbp_kibor_financing_rates.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"KIBOR dataset not found at {file_path}")
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    df["kibor_mom_change"] = df["kibor_6m_pct"].diff()
    df["auto_finance_mom_change"] = df["auto_finance_avg_rate_pct"].diff()
    df = df.fillna(0)
    print(f"Successfully loaded {len(df)} records of SBP KIBOR financing rates.")
    return df

if __name__ == "__main__":
    df_kibor = load_and_process_kibor_data()
    df_kibor.to_csv("processed_kibor_features.csv", index=False)
    print("Exported processed KIBOR features to 'processed_kibor_features.csv'")
