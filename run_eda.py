import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(filepath="master_market_intelligence_features.csv"):
    df = pd.read_csv(filepath)
    print("=== Master Feature Store Summary ===")
    print(df.describe())
    
    # Compute correlation matrix
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    
    os.makedirs("docs", exist_ok=True)
    
    # Save Correlation Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Suzuki Alto Karachi Market Intelligence - Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig("docs/correlation_matrix.png", dpi=300)
    plt.close()
    
    print("Correlation heatmap saved to 'docs/correlation_matrix.png'")
    return df

if __name__ == "__main__":
    run_eda()
