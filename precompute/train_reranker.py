import json
import faiss
import lightgbm as lgb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

print("Loading artifacts...")

features_df = pd.read_pickle(
    "artifacts/risk_scores.pkl"
)
if "candidate_id" in features_df.columns:
    features_df = features_df.set_index("candidate_id")

index = faiss.read_index(
    "artifacts/faiss.index"
)

with open(
    "artifacts/embedding_ids.json",
    "r",
    encoding="utf-8"
) as f:
    embedding_ids = json.load(f)

with open(
    "artifacts/honeypot_flags.json",
    "r",
    encoding="utf-8"
) as f:
    honeypots = set(json.load(f))

print("Artifacts loaded successfully!")
print("Feature matrix shape:", features_df.shape)


model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

with open(
    "data/job_description.md",
    "r",
    encoding="utf-8"
) as f:
    jd_text = f.read()

query = (
    "Represent this sentence for searching relevant passages: "
    + jd_text
)

print("Job description loaded!")

#job description embeddings
print("Embedding job description...")

jd_vector = model.encode(
    [query],
    normalize_embeddings=True
).astype(np.float32)

print("Job description embedded!")

#FAISS Search
print("Running FAISS search...")

scores, indices = index.search(
    jd_vector,
    1000
)

top_ids = [
    embedding_ids[i]
    for i in indices[0]
]

top_scores = scores[0]

print("Retrieved:", len(top_ids), "candidates")

id_to_faiss_score = {
    embedding_ids[i]: float(top_scores[j])
    for j, i in enumerate(indices[0])
}


print("Building training dataframe...")
valid_ids = [
    candidate_id
    for candidate_id in top_ids
    if candidate_id in features_df.index
    and candidate_id not in honeypots
]
train_df = features_df.loc[valid_ids].copy()

# create labels
train_df["faiss_score"] = train_df.index.map(
    id_to_faiss_score
)
print("Training candidates:", len(train_df))

train_df["label"] = pd.cut(
    train_df["faiss_score"],
    bins=5,
    labels=[0, 1, 2, 3, 4]
).astype(int)

# select features
feature_cols = train_df.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

feature_cols = [
    col
    for col in feature_cols
    if col not in ["label", "faiss_score"]
]

X = train_df[feature_cols]
y = train_df["label"]

group = [len(X)]

print(f"Training on {len(X)} candidates")
print(f"Using {len(feature_cols)} features")

# Create a LightGBM Ranker
print("Training LightGBM reranker...")
ranker = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    ndcg_eval_at=[10, 20],
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=5,
    verbose=-1
)

# training the model
ranker.fit(
    X,
    y,
    group=group
)
print("Training completed")

# Feature importance
importance = pd.Series(
    ranker.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)
print("\nTop 10 most important features:\n")
print(importance.head(10))

# saving the model
ranker.booster_.save_model(
    "artifacts/reranker_model.txt"
)
print("Saved in artifacts/reranker_model.txt")