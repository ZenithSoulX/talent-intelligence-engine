import json
import pickle
import numpy as np
import faiss

# Load embeddings
embeddings = np.load(
    "artifacts/embeddings.npy"
)

# Load candidate ids
with open(
    "artifacts/embedding_ids.json",
    "r",
    encoding="utf-8"
) as f:
    candidate_ids = json.load(f)

# Load FAISS
index = faiss.read_index(
    "artifacts/faiss.index"
)

# Load BM25
with open(
    "artifacts/bm25_index.pkl",
    "rb"
) as f:
    bm25 = pickle.load(f)

print("Embeddings shape:", embeddings.shape)
print("Candidate IDs:", len(candidate_ids))
print("FAISS vectors:", index.ntotal)

assert embeddings.shape[0] == len(candidate_ids)
assert embeddings.shape[0] == index.ntotal

print("All verification checks passed!")