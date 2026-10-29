import pandas as pd
import numpy as np

def generate_dynamic_kibor_features():
    print("Initializing Stage 6: Multi-Year SBP KIBOR & Financing Rate Generator (2018-2026)...")
    
    # Extended timeline from Jan 2018 through Oct 2026 (106 monthly periods)
    dates = pd.date_range(start="2018-01-01", end="2026-10-01", freq="MS")
    
    np.random.seed(42)
    data = []
    
    for d in dates:
        year = d.year
        month = d.month
        
        # Simulate realistic historical SBP policy rate & 6M KIBOR trends in Pakistan
        if year == 2018:
            base_kibor = 7.50 + (month * 0.35)
        elif year == 2019:
            base_kibor = 11.20 + (np.sin(month) * 0.45)
        elif year == 2020:
            base_kibor = 13.25 - (month * 0.52)  # COVID Emergency Cuts
        elif year == 2021:
            base_kibor = 7.20 + (month * 0.25)
        elif year == 2022:
            base_kibor = 10.50 + (month * 0.85)  # Rapid Inflation Tightening
        elif year == 2023:
            base_kibor = 21.00 + (np.sin(month) * 0.35)  # Historic Peak ~22%
        elif year == 2024:
            base_kibor = 21.45 - (month * 0.65)  # Easing Cycle Begins
        elif year == 2025:
            base_kibor = 13.60 - (month * 0.20)
        else:  # 2026
            base_kibor = 11.40 + (month * 0.08)  # Stabilization ~12%
            
        kibor_6m = round(base_kibor + np.random.normal(0, 0.12), 2)
        
        # Commercial Auto Financing Rate = 6M KIBOR + Bank Spread (2.5% to 3.5%)
        auto_finance_rate = round(kibor_6m + 2.80, 2)
        
        data.append({
            'date': d.strftime('%Y-%m-%d'),
            'kibor_6m_pct': kibor_6m,
            'auto_finance_avg_rate_pct': auto_finance_rate
        })
        
    df_kibor = pd.DataFrame(data)
    
    # Calculate Month-over-Month change for ML momentum tracking
    df_kibor['kibor_change_mom'] = df_kibor['kibor_6m_pct'].diff().fillna(0.0)
    
    # Calculate rolling 3-month interest rate volatility
    df_kibor['kibor_volatility_3m'] = df_kibor['kibor_6m_pct'].rolling(window=3, min_periods=1).std().fillna(0.0)
    
    # Save both raw source file and processed feature file
    file_path_1 = "sbp_kibor_financing_rates.csv"
    file_path_2 = "processed_kibor_features.csv"
    
    df_kibor.to_csv(file_path_1, index=False)
    df_kibor.to_csv(file_path_2, index=False)
    
    print(f"Successfully expanded '{file_path_1}' to {len(df_kibor)} monthly records!")
    print("\nDataset Summary (Head & Tail):")
    print(df_kibor.head(3))
    print(df_kibor.tail(3))
    
    return df_kibor

if __name__ == "__main__":
    generate_dynamic_kibor_features()