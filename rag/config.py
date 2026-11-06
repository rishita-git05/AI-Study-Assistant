"""Configuration parameters for the RAG pipeline.

All key parameters (chunk size, overlap, similarity threshold, top-k) are
defined here as configurable constants so they can be easily tuned or experimented with.
"""

# ============================================================================
# Chunking Configuration
# ============================================================================
# Chunk size in characters (recommended: 800 - 1000 characters).
# A chunk of ~900 characters roughly corresponds to 150-180 words or ~130-160 tokens.
# This is large enough to retain complete conceptual propositions and context,
# yet compact enough to avoid diluting semantic meaning in vector space.
CHUNK_SIZE: int = 900

# Overlap between consecutive chunks in characters (recommended: 150 - 200 characters).
# ~20% overlap ensures boundary concepts, cross-sentence references, or entity definitions
# spanning chunk boundaries are not lost or fragmented.
CHUNK_OVERLAP: int = 180

# ============================================================================
# Embedding Configuration
# ============================================================================
# Sentence Transformers model used for both document chunks and user queries.
# 'all-MiniLM-L6-v2' maps sentences to a 384-dimensional dense vector space.
# It serves as a lightweight, widely used baseline providing an optimal balance
# between semantic quality, vector dimension (384), CPU inference speed, and low memory (~80MB).
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384

# ============================================================================
# Retrieval Configuration
# ============================================================================
# Default number of top candidate chunks retrieved from FAISS before threshold filtering.
DEFAULT_TOP_K: int = 4

# Minimum cosine similarity score (inner product of L2-normalized vectors) required
# for a chunk to be considered relevant.
# Range: [-1.0, 1.0]. Typical relevant values for all-MiniLM-L6-v2 are >= 0.30 - 0.45.
# If no retrieved chunks meet or exceed this threshold, retrieval short-circuits
# without calling the LLM, preventing hallucinated answers.
# NOTE: This threshold is not a universal constant; it should be tuned against validation data.
SIMILARITY_THRESHOLD: float = 0.35

# Maximum allowed text/token overlap ratio between retrieved chunks before deduplication filters them.
# Reduces redundancy in context passed to the LLM.
MAX_CHUNK_OVERLAP_RATIO: float = 0.60

# ============================================================================
# Global Synthesis Configuration (Summarization / Quiz)
# ============================================================================
# Threshold above which full-document summarization triggers hierarchical map-reduce
# instead of single-pass prompt injection, preventing context-window exhaustion.
MAX_SUMMARY_CHUNK_CHARS: int = 25000
