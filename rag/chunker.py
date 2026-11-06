"""Document chunking module.

Implements boundary-aware recursive chunking (paragraphs -> sentences -> words)
with configurable chunk size and overlap, preserving page numbers, chunk IDs,
and character offsets.
"""

from dataclasses import dataclass
import re
from typing import List, Tuple
from rag.config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class DocumentChunk:
    """Represents a discrete semantic chunk of text with rich provenance metadata."""
    chunk_id: str
    page_number: int
    text: str
    start_char: int
    end_char: int

    def __repr__(self) -> str:
        snippet = self.text[:60] + "..." if len(self.text) > 60 else self.text
        return f"<DocumentChunk id={self.chunk_id} page={self.page_number} chars=[{self.start_char}:{self.end_char}] text={snippet!r}>"


def find_natural_split_point(text: str, start: int, target_end: int, min_end: int) -> int:
    """Finds the most natural boundary (paragraph, sentence, or word) near target_end.
    
    Searches backwards from target_end down to min_end to avoid cutting sentences mid-stream.
    """
    candidate_window = text[min_end:target_end]
    
    # 1. Paragraph boundary: double newline
    para_idx = candidate_window.rfind("\n\n")
    if para_idx != -1:
        return min_end + para_idx + 2

    # 2. Sentence boundary: punctuation (. ! ?) followed by whitespace or newline
    sentence_matches = list(re.finditer(r"[.!?](?:\s+|\n+)", candidate_window))
    if sentence_matches:
        last_match = sentence_matches[-1]
        return min_end + last_match.end()

    # 3. Single newline
    newline_idx = candidate_window.rfind("\n")
    if newline_idx != -1:
        return min_end + newline_idx + 1

    # 4. Word boundary: whitespace
    space_idx = candidate_window.rfind(" ")
    if space_idx != -1:
        return min_end + space_idx + 1

    # Fallback: slice at target_end directly
    return target_end


def chunk_page_text(
    page_number: int,
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    global_chunk_start_idx: int = 0
) -> List[DocumentChunk]:
    """Chunks the cleaned text of a single page using a boundary-aware sliding window.
    
    Args:
        page_number: 1-indexed page number.
        text: Cleaned page text.
        chunk_size: Maximum chunk length in characters (default: 900).
        chunk_overlap: Desired overlap in characters (default: 180).
        global_chunk_start_idx: Starting counter for unique chunk IDs.

    Returns:
        List of DocumentChunk instances for this page.
    """
    if not text.strip():
        return []

    text_len = len(text)
    
    # If the entire page fits in one chunk, return it as a single chunk
    if text_len <= chunk_size:
        return [
            DocumentChunk(
                chunk_id=f"chunk_{global_chunk_start_idx}",
                page_number=page_number,
                text=text.strip(),
                start_char=0,
                end_char=text_len
            )
        ]

    chunks: List[DocumentChunk] = []
    start = 0
    chunk_counter = global_chunk_start_idx

    while start < text_len:
        target_end = min(start + chunk_size, text_len)

        if target_end >= text_len:
            split_end = text_len
        else:
            # Look backwards from target_end up to half the chunk size to find a natural boundary
            min_end = max(start + (chunk_size // 2), start + 1)
            split_end = find_natural_split_point(text, start, target_end, min_end)

        chunk_text = text[start:split_end].strip()

        # Only record non-trivial chunks
        if chunk_text:
            chunks.append(
                DocumentChunk(
                    chunk_id=f"chunk_{chunk_counter}",
                    page_number=page_number,
                    text=chunk_text,
                    start_char=start,
                    end_char=split_end
                )
            )
            chunk_counter += 1

        if split_end >= text_len:
            break

        # Calculate next start point with overlap
        next_start = max(start + 1, split_end - chunk_overlap)
        
        # Advance next_start past leading whitespace to keep chunk boundaries clean
        while next_start < split_end and text[next_start].isspace():
            next_start += 1

        # Prevent infinite loops in edge cases
        if next_start <= start:
            start = split_end
        else:
            start = next_start

    return chunks


def chunk_document(
    pages_data: List[Tuple[int, str]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[DocumentChunk]:
    """Chunks an entire document given its page-level text data.
    
    Chunks are constrained to individual pages so page attribution remains exact.
    
    Args:
        pages_data: List of (page_number, cleaned_text) tuples.
        chunk_size: Character length per chunk.
        chunk_overlap: Character overlap between consecutive chunks.

    Returns:
        List of all DocumentChunk objects in sequential order.
    """
    all_chunks: List[DocumentChunk] = []
    chunk_idx = 0

    for page_num, page_text in pages_data:
        page_chunks = chunk_page_text(
            page_number=page_num,
            text=page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            global_chunk_start_idx=chunk_idx
        )
        all_chunks.extend(page_chunks)
        chunk_idx += len(page_chunks)

    return all_chunks
