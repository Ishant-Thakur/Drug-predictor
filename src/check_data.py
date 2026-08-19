import pandas as pd
df = pd.read_csv("data/delaney-processed.csv")

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Dataset shape ---")
print(df.shape)

print("\n--- Column names ---")
print(df.columns.tolist())

print("\n--- Dataset information ---")
print(df.info())

print("\n--- Missing values ---")
print(df.isnull().sum())

print("\n--- Statistical summary ---")
print(df.describe())