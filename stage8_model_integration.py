import pandas as pd
import numpy as np
import os

def build_market_intelligence_model():
    print("Loading all processed feature streams...")
    
    # 1. Load processed features
    # Assuming standard monthly date index across datasets
    fuel_df = pd.read_csv("processed_fuel_features.csv")
    kibor_df = pd.read_csv("processed_kibor_features.csv")
    tax_df = pd.read_csv("processed_tax_features.csv")
    
    print(f"Loaded Fuel features: {len(fuel_df)} rows")
    print(f"Loaded KIBOR features: {len(kibor_df)} rows")
    print(f"Loaded Tax features: {len(tax_df)} rows")
    
    # Merge datasets on date
    merged = fuel_df.merge(kibor_df, on="date", how="outer")
    merged = merged.merge(tax_df, on="date", how="outer")
    
    merged['date'] = pd.to_datetime(merged['date'])
    merged = merged.sort_values('date').reset_index(drop=True)
    
    # Save master feature store for Model 2
    merged.to_csv("master_market_intelligence_features.csv", index=False)
    print("Successfully compiled master feature store: 'master_market_intelligence_features.csv'")
    print(merged.tail(3))
    
    return merged

if __name__ == "__main__":
    build_market_intelligence_model()
