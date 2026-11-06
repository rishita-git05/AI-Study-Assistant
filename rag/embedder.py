"""Dense embedding generation module using Sentence Transformers.

Uses 'sentence-transformers/all-MiniLM-L6-v2', a lightweight and widely used
baseline that provides a good trade-off between semantic quality, embedding
dimensionality (384), CPU inference speed, and resource consumption.

Enforces L2 normalization on all generated vectors so that FAISS IndexFlatIP
(Inner Product) computes exact Cosine Similarity.
"""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from rag.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION
from rag.chunker import DocumentChunk


class EmbeddingGenerator:
    """Singleton-style wrapper for SentenceTransformer embedding generation."""

    _instance: Optional["EmbeddingGenerator"] = None
    _model: Optional[SentenceTransformer] = None

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._load_model()

    @classmethod
    def get_instance(cls, model_name: str = EMBEDDING_MODEL_NAME) -> "EmbeddingGenerator":
        """Returns the shared instance of the embedding generator to avoid reloading weights."""
        if cls._instance is None or cls._instance.model_name != model_name:
            cls._instance = cls(model_name)
        return cls._instance

    def _load_model(self):
        """Loads the SentenceTransformer model into memory."""
        print(f"[embedder] Loading embedding model: {self.model_name}...")
        # device will default to CUDA if available, else CPU
        self.model = SentenceTransformer(self.model_name)
        print(f"[embedder] Model {self.model_name} loaded successfully.")

    @staticmethod
    def _normalize_l2(vectors: np.ndarray) -> np.ndarray:
        """Applies L2 normalization so that ||v||_2 = 1.0.
        
        Mathematical rationale:
        For unit-length vectors u and v:
            dot_product(u, v) = ||u|| * ||v|| * cos(theta) = 1.0 * 1.0 * cos(theta) = cos(theta)
        Thus, inner product in FAISS IndexFlatIP is exactly equal to Cosine Similarity.
        """
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        # Avoid division by zero for zero-vectors
        norms = np.maximum(norms, 1e-12)
        return (vectors / norms).astype(np.float32)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embeds a list of strings into L2-normalized float32 numpy vectors.
        
        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for inference.

        Returns:
            np.ndarray of shape (len(texts), 384) with dtype float32.
        """
        if not texts:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

        # Generate embeddings (SentenceTransformers handles batching)
        raw_embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        return self._normalize_l2(raw_embeddings)

    def embed_chunks(self, chunks: List[DocumentChunk], batch_size: int = 32) -> np.ndarray:
        """Embeds a list of DocumentChunk objects."""
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts, batch_size=batch_size)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string into an L2-normalized 2D float32 vector (1, 384).
        
        Args:
            query: User's question or search query.

        Returns:
            np.ndarray of shape (1, 384) with dtype float32.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return np.zeros((1, EMBEDDING_DIMENSION), dtype=np.float32)

        raw_vec = self.model.encode([cleaned_query], show_progress_bar=False, convert_to_numpy=True)
        return self._normalize_l2(raw_vec)
