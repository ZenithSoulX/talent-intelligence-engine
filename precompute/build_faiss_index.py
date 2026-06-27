import numpy as np
import faiss

embeddings = np.load(
    "artifacts/embeddings.npy"
)

print("Loaded embeddings:", embeddings.shape)

# normalize and build index
faiss.normalize_L2(embeddings)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(
    dimension
)
index.add(embeddings)
print("Vectors in index:", index.ntotal)

# Saving
faiss.write_index(
    index,
    "artifacts/faiss.index"
)
print("FAISS index saved!")
