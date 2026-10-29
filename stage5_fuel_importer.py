import pandas as pd
import numpy as np

def generate_dynamic_fuel_features():
    print("Initializing Stage 5: Multi-Year Fuel Price Feature Generator (2018-2026)...")
    
    # Extended timeline matching Tax and KIBOR datasets (106 monthly periods)
    dates = pd.date_range(start="2018-01-01", end="2026-10-01", freq="MS")
    
    np.random.seed(42)
    data = []
    
    for d in dates:
        year = d.year
        month = d.month
        
        # Simulate realistic historical Pakistani petrol & diesel price evolution (PKR/Liter)
        if year == 2018:
            base_petrol = 88.0 + (month * 1.5)
            base_diesel = 95.0 + (month * 1.6)
        elif year == 2019:
            base_petrol = 106.0 + (month * 1.1)
            base_diesel = 118.0 + (month * 1.2)
        elif year == 2020:
            base_petrol = 116.0 - (np.sin(month) * 8.0) # COVID Global Oil Crash
            base_diesel = 122.0 - (np.sin(month) * 9.0)
        elif year == 2021:
            base_petrol = 110.0 + (month * 2.8)
            base_diesel = 112.0 + (month * 2.6)
        elif year == 2022:
            base_petrol = 145.0 + (month * 12.0) # 2022 Fuel Deregulation & Subsidy Removal
            base_diesel = 148.0 + (month * 12.5)
        elif year == 2023:
            base_petrol = 270.0 + (np.sin(month) * 25.0) # Peak volatility (>300 PKR)
            base_diesel = 280.0 + (np.sin(month) * 26.0)
        elif year == 2024:
            base_petrol = 267.0 + (np.cos(month) * 12.0)
            base_diesel = 275.0 + (np.cos(month) * 11.0)
        elif year == 2025:
            base_petrol = 255.0 + (month * 1.8)
            base_diesel = 263.0 + (month * 1.9)
        else: # 2026
            base_petrol = 279.1 + (month * 1.5)
            base_diesel = 287.6 + (month * 1.5)
            
        petrol_price = round(base_petrol + np.random.normal(0, 0.8), 1)
        diesel_price = round(base_diesel + np.random.normal(0, 0.8), 1)
        
        data.append({
            'date': d.strftime('%Y-%m-%d'),
            'petrol_pkr_per_liter': petrol_price,
            'diesel_pkr_per_liter': diesel_price,
            'region': 'Karachi, Sindh'
        })
        
    df_fuel = pd.DataFrame(data)
    
    # Calculate MoM percentage changes for momentum features
    df_fuel['petrol_mom_change_pct'] = round(df_fuel['petrol_pkr_per_liter'].pct_change().fillna(0.0) * 100, 2)
    df_fuel['diesel_mom_change_pct'] = round(df_fuel['diesel_pkr_per_liter'].pct_change().fillna(0.0) * 100, 2)
    
    # Calculate rolling 3-month petrol price volatility
    df_fuel['petrol_volatility_3m'] = df_fuel['petrol_pkr_per_liter'].rolling(window=3, min_periods=1).std().fillna(0.0)
    
    # Save raw and processed CSV files
    file_path_1 = "pakistan_fuel_prices.csv"
    file_path_2 = "processed_fuel_features.csv"
    
    df_fuel.to_csv(file_path_1, index=False)
    df_fuel.to_csv(file_path_2, index=False)
    
    print(f"Successfully expanded '{file_path_1}' and '{file_path_2}' to {len(df_fuel)} monthly records!")
    print(df_fuel.head(3))
    print(df_fuel.tail(3))
    
    return df_fuel

if __name__ == "__main__":
    generate_dynamic_fuel_features()