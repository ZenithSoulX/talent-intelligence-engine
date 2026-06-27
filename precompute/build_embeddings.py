import json
import numpy as np
from sentence_transformers import SentenceTransformer
from build_composite_text import build_composite_text

candidates = []

with open("data/candidates.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            candidates.append(json.loads(line))

print("Candidates loaded:", len(candidates))

texts = [
    build_composite_text(candidate)
    for candidate in candidates
]

candidate_ids = [
    candidate["candidate_id"]
    for candidate in candidates
]

print("Candidates:", len(candidates))

# Model
model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

embeddings = model.encode(
    texts,
    batch_size=128,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("Embeddings shape:", embeddings.shape)
embeddings = embeddings.astype(np.float32)

# saving
np.save(
    "artifacts/embeddings.npy",
    embeddings
)

print("Saved embeddings.npy")

# save embeddings ids
with open(
    "artifacts/embedding_ids.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(candidate_ids, f)

print("Saved embedding_ids.json")