"""FAISS Vector Store wrapper with in-memory metadata tracking.

Uses faiss.IndexFlatIP (Inner Product) over L2-normalized 384-dimensional embeddings.
Because the embeddings have unit norm, the inner product is mathematically identical
to Cosine Similarity.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import faiss
from rag.config import EMBEDDING_DIMENSION
from rag.chunker import DocumentChunk


class FAISSVectorStore:
    """In-memory FAISS vector index with parallel chunk metadata storage."""

    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        self.dimension = dimension
        # IndexFlatIP performs an exhaustive, exact brute-force search using Inner Product
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        # Registry mapping FAISS integer index (0 .. N-1) to DocumentChunk
        self.metadata_store: Dict[int, DocumentChunk] = {}

    @property
    def total_chunks(self) -> int:
        """Returns total number of chunks currently indexed in FAISS."""
        return self.index.ntotal

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """Adds document chunks and their corresponding dense vectors to the index.
        
        Args:
            chunks: List of DocumentChunk instances.
            embeddings: Float32 numpy array of shape (len(chunks), dimension).
        """
        if not chunks or len(chunks) == 0:
            return

        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks provided but embeddings have shape {embeddings.shape}"
            )

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        start_id = self.index.ntotal
        # Add vectors to FAISS index
        self.index.add(embeddings)

        # Store metadata mapping
        for i, chunk in enumerate(chunks):
            self.metadata_store[start_id + i] = chunk

        print(f"[vector_store] Added {len(chunks)} chunks to FAISS index. Total now: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[Tuple[DocumentChunk, float]]:
        """Searches the index for the top-k most similar chunks to the query vector.
        
        Args:
            query_embedding: L2-normalized float32 vector of shape (1, dimension).
            top_k: Number of candidate neighbors to return.

        Returns:
            List of (DocumentChunk, similarity_score) tuples, ordered by descending score.
            Similarity score is the cosine similarity in [-1.0, 1.0].
        """
        if self.index.ntotal == 0:
            return []

        # Clamp k to total available vectors in index
        actual_k = min(top_k, self.index.ntotal)
        if actual_k <= 0:
            return []

        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        # FAISS search: D contains inner products (scores), I contains integer indices
        scores, indices = self.index.search(query_embedding, actual_k)

        results: List[Tuple[DocumentChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.metadata_store.get(int(idx))
            if chunk is not None:
                results.append((chunk, float(score)))

        return results

    def clear(self):
        """Resets the vector index and metadata registry."""
        self.index.reset()
        self.metadata_store.clear()
        print("[vector_store] FAISS index and metadata store cleared.")
