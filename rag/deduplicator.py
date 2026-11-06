"""Lightweight retrieval deduplication module.

Filters out candidate retrieved chunks that have excessive text overlap with
already accepted higher-ranking chunks, preventing prompt pollution and redundancy.
"""

from typing import List, Tuple, Set
import re
from rag.chunker import DocumentChunk
from rag.config import MAX_CHUNK_OVERLAP_RATIO


def _tokenize_to_words(text: str) -> Set[str]:
    """Tokenizes text into a set of lowercase alphanumeric words for fast set operations."""
    return set(re.findall(r"\b\w{3,}\b", text.lower()))


def compute_token_jaccard_similarity(text1: str, text2: str) -> float:
    """Computes Jaccard similarity between two texts based on unique word tokens.
    
    J(A, B) = |A ∩ B| / |A ∪ B|
    Range: [0.0, 1.0]
    """
    tokens1 = _tokenize_to_words(text1)
    tokens2 = _tokenize_to_words(text2)

    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))

    return intersection / union if union > 0 else 0.0


def deduplicate_chunks(
    scored_chunks: List[Tuple[DocumentChunk, float]],
    max_overlap_ratio: float = MAX_CHUNK_OVERLAP_RATIO
) -> List[Tuple[DocumentChunk, float]]:
    """Deduplicates a ranked list of retrieved chunks.
    
    Iterates through candidates in order of similarity score (highest first).
    Discards any subsequent candidate that shares significant token overlap
    with an already accepted chunk.
    
    Args:
        scored_chunks: List of (DocumentChunk, score) tuples sorted by descending score.
        max_overlap_ratio: Maximum allowable Jaccard token overlap before rejection.

    Returns:
        List of deduplicated (DocumentChunk, score) tuples preserving relative rank.
    """
    if not scored_chunks:
        return []

    accepted: List[Tuple[DocumentChunk, float]] = []

    for candidate_chunk, score in scored_chunks:
        is_duplicate = False

        for accepted_chunk, _ in accepted:
            # 1. Exact identity check
            if candidate_chunk.chunk_id == accepted_chunk.chunk_id:
                is_duplicate = True
                break

            # 2. Token Jaccard overlap check
            overlap = compute_token_jaccard_similarity(candidate_chunk.text, accepted_chunk.text)
            if overlap >= max_overlap_ratio:
                is_duplicate = True
                break

        if not is_duplicate:
            accepted.append((candidate_chunk, score))

    return accepted
