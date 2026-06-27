import json
import pickle
from rank_bm25 import BM25Okapi
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

tokenized_docs = [
    text.lower().split()
    for text in texts
]

bm25 = BM25Okapi(tokenized_docs)

with open(
    "artifacts/bm25_index.pkl",
    "wb"
) as f:
    pickle.dump(bm25, f)

print("BM25 index saved")
print("Documents indexed:", len(texts))