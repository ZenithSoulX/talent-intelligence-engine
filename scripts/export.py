import pandas as pd

df = pd.read_pickle("artifacts/features.pkl")

sample = df.sample(
    n=20,
    random_state=42
)

sample.to_csv(
    "artifacts/feature_matrix_20.csv"
)

print("Saved:")
print("artifacts/feature_matrix_20.csv")