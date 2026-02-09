import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import os , sys

df = pd.read_csv(os.path.join(sys.path[0], input()))
df['DATE'] = pd.to_datetime(df['DATE'] , dayfirst=True)
df.set_index('DATE', inplace=True)

print("First 5 records of dataset:")
print(df.head())

df['Consumption'].fillna(df['Consumption'].mean(), inplace=True)

q1 = df['Consumption'].quantile(0.25)
q3 = df['Consumption'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df['Consumption'] = df["Consumption"].clip(lower_bound, upper_bound)    

print("\nData preprocessing completed.")

decomposition = seasonal_decompose(df['Consumption'], model='additive', period=12)

print("\nAdditive Model Components (First 5 Values)")
print("Trend:\n", decomposition.trend.dropna().head())
print("\nSeasonality:\n", decomposition.seasonal.head())
print("\nResiduals:\n", decomposition.resid.dropna().head())

decomposition_multiplicative = seasonal_decompose(df['Consumption'], model='multiplicative', period=12)

print("\nMultiplicative Model Components (First 5 Values)")
print("Trend:\n", decomposition_multiplicative.trend.dropna().head())
print("\nSeasonality:\n", decomposition_multiplicative.seasonal.head())
print("\nResiduals:\n", decomposition_multiplicative.resid.dropna().head())

print("Model Comparison Conclusion:")
print("If seasonal values are constant → Additive model fits better.")
print("If seasonal values change proportionally with trend → Multiplicative model fits better.")