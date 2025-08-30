import io
from typing import List, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors
from pypdf import PdfReader


def extract_text_by_page(pdf_bytes: bytes) -> List[str]:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as e:
        raise ValueError("Invalid or corrupted PDF. Please upload a valid PDF file.") from e
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return pages


def chunk_texts(
    texts: List[str],
    max_tokens: int = 500,
    overlap: int = 100,
) -> List[Tuple[str, int]]:
    """
    Simple chunker by character count as a proxy for tokens.
    Returns a list of (chunk_text, source_page_idx)
    """
    chunks: List[Tuple[str, int]] = []
    for page_idx, text in enumerate(texts):
        start = 0
        while start < len(text):
            end = min(start + max_tokens, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append((chunk, page_idx))
            if end == len(text):
                break
            start = end - overlap
            if start < 0:
                start = 0
    return chunks


class SentenceEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.astype(np.float32)



def build_nn_index(embeddings: np.ndarray, n_neighbors: int = 5) -> NearestNeighbors:
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(embeddings)
    return nn


