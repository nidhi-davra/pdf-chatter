from typing import List, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


def retrieve_top_k(
    query: str,
    embed_fn,
    nn: NearestNeighbors,
    chunks: List[Tuple[str, int]],
    k: int = 5,
):
    query_emb = embed_fn([query])
    distances, indices = nn.kneighbors(query_emb, n_neighbors=min(k, len(chunks)))
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        text, page_idx = chunks[idx]
        results.append({
            "text": text,
            "page": page_idx + 1,
            "score": float(1.0 - dist),  # cosine similarity proxy
        })
    return results


