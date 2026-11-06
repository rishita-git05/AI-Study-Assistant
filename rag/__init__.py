"""RAG (Retrieval-Augmented Generation) package for AI Study Assistant.

Provides modular components for:
- PDF page-level extraction and artifact cleaning
- Natural-boundary recursive chunking
- Dense semantic embeddings (all-MiniLM-L6-v2)
- FAISS vector indexing (IndexFlatIP with L2 normalized vectors)
- Semantic retrieval with threshold filtering and deduplication
"""

from rag.chunker import DocumentChunk, chunk_document
from rag.text_processor import extract_pages_from_pdf, clean_page_text, get_full_cleaned_text
from rag.embedder import EmbeddingGenerator
from rag.vector_store import FAISSVectorStore
from rag.pipeline import RAGPipeline, RetrievalResult, DebugChunkInfo
from rag.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL_NAME
)

__all__ = [
    "RAGPipeline",
    "RetrievalResult",
    "DebugChunkInfo",
    "DocumentChunk",
    "chunk_document",
    "extract_pages_from_pdf",
    "clean_page_text",
    "get_full_cleaned_text",
    "EmbeddingGenerator",
    "FAISSVectorStore",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "DEFAULT_TOP_K",
    "SIMILARITY_THRESHOLD",
    "EMBEDDING_MODEL_NAME"
]
