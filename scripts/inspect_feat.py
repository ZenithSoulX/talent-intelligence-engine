import pandas as pd

df = pd.read_pickle("artifacts/features.pkl")

print("=" * 60)
print("FEATURE MATRIX OVERVIEW")
print("=" * 60)
print("\nShape:")
print(df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nNull Counts:")
print(df.isna().sum())
print("\nSample Rows:")
print(df.sample(5, random_state=42))
print("\nStatistics:")
print(df.describe().T)
print("\nDone.")