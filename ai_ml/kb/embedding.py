"""Local question-only E5 embeddings (384 dimensions)."""
import numpy as np
from pathlib import Path

MODEL = "intfloat/multilingual-e5-small"
DIMENSION = 384
STRATEGY = "question_only_e5_v1"


def validate_vectors(vectors, count):
    a = np.asarray(vectors, dtype=np.float32)
    if a.shape != (count, DIMENSION) or not np.isfinite(a).all():
        raise ValueError(f"Expected {count} finite vectors of dimension {DIMENSION}.")
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("A zero embedding is not valid.")
    return a / norms


class E5:
    def __init__(self, device="cpu", batch_size=16):
        from sentence_transformers import SentenceTransformer
        if batch_size < 1:
            raise ValueError("Batch size must be positive.")
        self.model = SentenceTransformer(MODEL, device=device,
            cache_folder=str(Path(__file__).resolve().parents[1]/'.cache/huggingface'))
        self.batch_size = batch_size

    def encode(self, texts, query=False):
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Embedding input must contain nonempty text.")
        prefix = "query: " if query else "passage: "
        inputs = [prefix + text.strip() for text in texts]
        lengths = self.model.tokenizer(inputs, truncation=False, padding=False)["input_ids"]
        if any(len(tokens) > self.model.max_seq_length for tokens in lengths):
            raise ValueError("A question exceeds the model token limit; shorten or split it before indexing.")
        return validate_vectors(self.model.encode(
            inputs, batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=not query,
        ), len(texts))
