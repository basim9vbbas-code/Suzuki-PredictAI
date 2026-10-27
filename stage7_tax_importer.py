import pandas as pd
import numpy as np
import os

def load_and_process_tax_data(file_path="tax_sro_events_pakistan.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Tax events dataset not found at {file_path}")
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate total effective duty index for hatchback segment
    df['total_duty_rate_pct'] = df['sales_tax_pct'] + df['fed_pct']
    df['duty_change_mom'] = df['total_duty_rate_pct'].diff().fillna(0)
    
    print(f"Successfully loaded {len(df)} records of Tax & SRO duty events (Jan 2024 - Oct 2026).")
    print(df[df['sro_event_flag'] == 1])
    return df

if __name__ == "__main__":
    df_tax = load_and_process_tax_data()
    df_tax.to_csv("processed_tax_features.csv", index=False)
    print("Exported processed tax features to 'processed_tax_features.csv'")
