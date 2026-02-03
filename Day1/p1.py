import pandas as pd
import os , sys

df = pd.read_csv(os.path.join(sys.path[0] , input()))

print("First 5 rows of the dataset:")
print(df.head())
print()

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

print("Missing values in dataset:")
print(df.isnull().sum())

print("\nNumber of duplicate rows:", df.duplicated().sum())

print("Close price summary statistics:")
print(df['Close'].describe())