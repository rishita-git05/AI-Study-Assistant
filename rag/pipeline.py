"""End-to-end RAG Pipeline coordinator.

Strictly separates the two fundamental phases:
1. OFFLINE INGESTION / INDEXING:
   PDF -> Text extraction (pdfplumber) -> Text cleaning -> Recursive chunking ->
   Dense embeddings (all-MiniLM-L6-v2) -> FAISS IndexFlatIP.

2. ONLINE QUERY-TIME RETRIEVAL:
   User question -> Query embedding -> FAISS similarity search ->
   Deduplication -> Relevance threshold filtering -> Context assembly ->
   Deterministic source provenance.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from rag.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    SIMILARITY_THRESHOLD,
    MAX_CHUNK_OVERLAP_RATIO
)
from rag.text_processor import extract_pages_from_pdf, get_full_cleaned_text
from rag.chunker import chunk_document, DocumentChunk
from rag.embedder import EmbeddingGenerator
from rag.vector_store import FAISSVectorStore
from rag.deduplicator import deduplicate_chunks


@dataclass
class DebugChunkInfo:
    """Individual chunk diagnostics exposed for retrieval inspection and evaluation."""
    rank: int
    chunk_id: str
    page_number: int
    score: float
    passed_threshold: bool
    text: str


@dataclass
class RetrievalResult:
    """Encapsulates the complete result of the semantic retrieval phase."""
    has_sufficient_context: bool
    query: str
    relevant_chunks: List[Tuple[DocumentChunk, float]] = field(default_factory=list)
    debug_chunks: List[DebugChunkInfo] = field(default_factory=list)
    sources: List[int] = field(default_factory=list)
    formatted_context: str = ""
    fallback_message: str = ""


class RAGPipeline:
    """Coordinates document ingestion, vector indexing, and grounded retrieval."""

    def __init__(self):
        self.embedder = EmbeddingGenerator.get_instance()
        self.vector_store = FAISSVectorStore()
        self.pages_data: List[Tuple[int, str]] = []
        self.all_chunks: List[DocumentChunk] = []
        self.current_pdf_name: Optional[str] = None

    @property
    def is_indexed(self) -> bool:
        """Returns True if at least one document chunk is indexed in FAISS."""
        return self.vector_store.total_chunks > 0

    # ========================================================================
    # STAGE 1: OFFLINE INGESTION & INDEXING
    # ========================================================================

    def index_document(
        self,
        file_input,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ) -> Dict[str, Any]:
        """Ingests, cleans, chunks, embeds, and indexes a PDF document.
        
        Args:
            file_input: File path or Gradio file object.
            chunk_size: Target characters per chunk.
            chunk_overlap: Overlap characters between chunks.

        Returns:
            Dictionary containing indexing statistics.
        """
        self.clear()

        # 1. Text Extraction & Cleaning
        self.pages_data = extract_pages_from_pdf(file_input)
        if not self.pages_data:
            return {
                "success": False,
                "error": "No readable text found. The document may be empty, password-protected, or a scanned image lacking an OCR text layer.",
                "total_pages": 0,
                "total_chunks": 0
            }

        # 2. Boundary-aware Chunking
        self.all_chunks = chunk_document(
            self.pages_data,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        if not self.all_chunks:
            return {
                "success": False,
                "error": "Failed to generate any valid text chunks from document.",
                "total_pages": len(self.pages_data),
                "total_chunks": 0
            }

        # 3. Dense Embedding Generation (L2-normalized)
        embeddings = self.embedder.embed_chunks(self.all_chunks)

        # 4. Vector Storage in FAISS IndexFlatIP
        self.vector_store.add_chunks(self.all_chunks, embeddings)

        file_name = getattr(file_input, "name", str(file_input))
        self.current_pdf_name = file_name

        return {
            "success": True,
            "total_pages": len(self.pages_data),
            "total_chunks": len(self.all_chunks),
            "indexed_vectors": self.vector_store.total_chunks
        }

    # ========================================================================
    # STAGE 2: ONLINE QUERY-TIME RETRIEVAL
    # ========================================================================

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD
    ) -> RetrievalResult:
        """Executes semantic retrieval for a user question against the indexed document.
        
        Pipeline:
        Query -> Dense Embedding -> FAISS Inner Product Search ->
        Deduplication -> Relevance Threshold Filtering -> Context Assembly.
        
        Args:
            query: User's question.
            top_k: Maximum number of relevant chunks to retrieve.
            similarity_threshold: Minimum cosine similarity required.

        Returns:
            RetrievalResult containing relevant chunks, debug metadata, and assembled context.
        """
        query_text = query.strip()
        if not query_text:
            return RetrievalResult(
                has_sufficient_context=False,
                query=query,
                fallback_message="Please enter a valid question."
            )

        if not self.is_indexed:
            return RetrievalResult(
                has_sufficient_context=False,
                query=query,
                fallback_message="Please upload a PDF document before asking questions."
            )

        # 1. Embed query (L2-normalized)
        query_vec = self.embedder.embed_query(query_text)

        # 2. FAISS similarity search (retrieve candidate pool of 2 * top_k for deduplication)
        candidate_pool_k = min(top_k * 2, self.vector_store.total_chunks)
        raw_candidates = self.vector_store.search(query_vec, top_k=candidate_pool_k)

        # 3. Lightweight Deduplication based on text/token overlap
        deduped_candidates = deduplicate_chunks(
            raw_candidates,
            max_overlap_ratio=MAX_CHUNK_OVERLAP_RATIO
        )

        # Limit to top-k requested after deduplication
        candidate_chunks = deduped_candidates[:top_k]

        # 4. Relevance Threshold Filtering & Diagnostic Tracking
        debug_chunks: List[DebugChunkInfo] = []
        relevant_chunks: List[Tuple[DocumentChunk, float]] = []

        for rank, (chunk, score) in enumerate(candidate_chunks, start=1):
            passed = score >= similarity_threshold
            debug_chunks.append(
                DebugChunkInfo(
                    rank=rank,
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                    score=round(score, 4),
                    passed_threshold=passed,
                    text=chunk.text
                )
            )
            if passed:
                relevant_chunks.append((chunk, score))

        # 5. Threshold Short-Circuit
        # If no chunks passed the threshold, do NOT proceed to LLM generation
        if not relevant_chunks:
            return RetrievalResult(
                has_sufficient_context=False,
                query=query_text,
                debug_chunks=debug_chunks,
                fallback_message="I couldn't find enough relevant information in the uploaded document to answer this question."
            )

        # 6. Context Assembly & Deterministic Source Provenance
        # Extract unique page numbers directly from chunk metadata
        unique_pages = sorted(list({chunk.page_number for chunk, _ in relevant_chunks}))

        context_blocks = []
        for chunk, score in relevant_chunks:
            context_blocks.append(
                f"[Page {chunk.page_number} | Chunk {chunk.chunk_id}]:\n{chunk.text}"
            )
        formatted_context = "\n\n---\n\n".join(context_blocks)

        return RetrievalResult(
            has_sufficient_context=True,
            query=query_text,
            relevant_chunks=relevant_chunks,
            debug_chunks=debug_chunks,
            sources=unique_pages,
            formatted_context=formatted_context
        )

    def get_document_full_text(self) -> str:
        """Returns the complete cleaned text of all pages for global synthesis."""
        return get_full_cleaned_text(self.pages_data)

    def clear(self):
        """Clears all indexed state."""
        self.vector_store.clear()
        self.pages_data.clear()
        self.all_chunks.clear()
        self.current_pdf_name = None
