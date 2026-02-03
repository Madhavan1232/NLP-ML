import pandas as pd
import os
import sys
try:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import matplotlib.pyplot as plt
    ts_available = True
except ImportError:
    ts_available = False
df = pd.read_csv(os.path.join(sys.path[0], input()))
print("Dataset Preview:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nMissing Value Check:")
print(df.drop(columns=['Datetime']).isnull().sum())
df = df.dropna()
print("After missing value handling:")
print(df.drop(columns=['Datetime']).isnull().sum())
print("\nACF and PACF Analysis:")
if ts_available:
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(df['Consumption'], ax=ax1)
    plot_pacf(df['Consumption'], ax=ax2)
    plt.tight_layout()
    plt.show()
    print("ACF and PACF plots generated.")
else:
    print("Time series module not available. Skipping ACF/PACF plots.")
train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
print("\nTrain-Test Split:")
print(f"Training records: {train_size}")
print(f"Testing records: {test_size}")