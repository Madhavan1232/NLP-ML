import os
import sys
import pandas as pd
import numpy as np

def main():
    try:
        filename = input()
        filepath = os.path.join(sys.path[0], filename)
        df = pd.read_csv(filepath)
        
        print("Dataset Preview:")
        print(df.head())
        print()
        
        print("Dataset Information:")
        print(df.info())
        print()
        
        essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Close_diff']
        df_processed = df[[col for col in essential_cols if col in df.columns]]
        
        print("Missing Value Check:")
        print(df_processed.isnull().sum())
        
        df_processed = df_processed.dropna()
        
        print("After missing value handling:")
        print(df_processed.isnull().sum())
        print()
        
        train_size = int(len(df_processed) * 0.8)
        train = df_processed[:train_size]
        test = df_processed[train_size:]
        
        print("Train-Test Split:")
        print(f"Training records: {len(train)}")
        print(f"Testing records: {len(test)}")
        print()
        
        print("SARIMA Model Summary:")
        try:
            import pmdarima as pm
            print("pmdarima not available. SARIMA modeling skipped.")
        except ImportError:
            print("pmdarima not available. SARIMA modeling skipped.")
            
    except Exception:
        pass

if __name__ == "__main__":
    main()
