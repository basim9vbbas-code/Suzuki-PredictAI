import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def engineer_features(df):
    """
    Adds time-series feature engineering:
    - Lag features (prior month demand momentum)
    - Moving averages (3-month rolling averages)
    - Macroeconomic interaction terms & NLP Sentiment Integration
    """
    df = df.copy()
    
    # 1. Historical Lag Features
    if 'suzuki_alto_sales_units' in df.columns:
        df['sales_lag_1m'] = df['suzuki_alto_sales_units'].shift(1)
    else:
        df['sales_lag_1m'] = 1200
        
    # 2. Rolling Macroeconomic Averages
    df['petrol_3m_roll_avg'] = df['petrol_pkr_per_liter'].rolling(window=3, min_periods=1).mean()
    df['kibor_3m_roll_avg'] = df['kibor_6m_pct'].rolling(window=3, min_periods=1).mean()
    
    # 3. Policy, Economic & Sentiment Interactions
    df['fuel_kibor_interaction'] = df['petrol_pkr_per_liter'] * df['kibor_6m_pct']
    df['effective_tax_load'] = df['total_duty_rate_pct'] + (df['sro_event_flag'] * 2.5)
    
    # Fill shifting NaNs gracefully
    return df.bfill().ffill()

def train_suzuki_model(filepath="master_market_intelligence_features.csv"):
    if not os.path.exists(filepath):
        print(f"Error: Master feature store '{filepath}' not found. Please run stage8_model_integration.py first.")
        return None

    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Apply Feature Engineering
    df = engineer_features(df)
    
    # Complete Enterprise Feature Set (Including Transformer NLP Scores)
    feature_cols = [
        'petrol_pkr_per_liter', 
        'kibor_6m_pct', 
        'auto_finance_avg_rate_pct', 
        'total_duty_rate_pct', 
        'sro_event_flag',
        'sentiment_polarity_score',
        'sales_lag_1m',
        'petrol_3m_roll_avg',
        'kibor_3m_roll_avg',
        'fuel_kibor_interaction',
        'effective_tax_load'
    ]
    
    # Filter features that exist in the dataframe
    feature_cols = [c for c in feature_cols if c in df.columns]
    target_col = 'suzuki_alto_sales_units'
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Strict Chronological Split (80% Train / 20% Test) to prevent lookahead bias
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Dataset Size: {len(df)} records | Training: {len(X_train)} | Testing: {len(X_test)}")
    
    # Time-Series Cross Validation with Regularized Grid Bounds
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [3, 4, 5, 6],          # Bounded depth prevents leaf overfitting
        'min_samples_split': [3, 5, 8],
        'min_samples_leaf': [2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    base_rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=base_rf, 
        param_grid=param_grid, 
        cv=tscv, 
        scoring='neg_mean_absolute_error', 
        n_jobs=-1
    )
    
    print("Executing TimeSeriesSplit Grid Search optimization...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # Model Out-of-Sample Evaluation
    preds = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print("\n=== Optimized Suzuki PredictAI Model Results ===")
    print(f"Optimal Hyperparameters: {grid_search.best_params_}")
    print(f"Test MAE:  {mae:.2f} units")
    print(f"Test RMSE: {rmse:.2f} units")
    print(f"Test R2:   {r2:.4f}")
    
    print("\n=== Feature Importance Rankings ===")
    importances = best_model.feature_importances_
    for col, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
        print(f"- {col}: {imp:.4f}")
        
    os.makedirs("src", exist_ok=True)
    model_path = "src/suzuki_alto_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"\nModel artifact saved successfully to '{model_path}'")
    
    return best_model

if __name__ == "__main__":
    train_suzuki_model()