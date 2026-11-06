"""PDF text extraction and cleaning module.

Responsible for extracting text page-by-page from PDFs using pdfplumber,
cleaning common PDF extraction artifacts (broken hyphens, form feeds, excessive whitespace),
and preserving page numbers for source attribution.
"""

import re
from typing import List, Tuple, Optional
import pdfplumber


def clean_page_text(text: str) -> str:
    """Cleans raw PDF text extraction artifacts while preserving semantic structure.
    
    Operations performed:
    1. Removes form feed and replacement characters (\x0c, \ufffd).
    2. Reconnects words split across lines by hyphens (e.g. 'retrie-\\nval' -> 'retrieval').
    3. Normalizes horizontal whitespace (tabs, consecutive spaces) to single spaces.
    4. Normalizes vertical whitespace (more than 2 newlines -> 2 newlines for paragraph breaks).
    5. Strips leading/trailing whitespace.
    """
    if not text:
        return ""

    # Remove form feed, null, and unicode replacement characters
    text = text.replace("\x0c", "").replace("\x00", "").replace("\ufffd", "")

    # Reconnect words broken by hyphenation at line breaks
    # Matches a word character, hyphen, newline, optional whitespace, and continuing word character
    text = re.sub(r"(\b\w+)-\n\s*(\w+\b)", r"\1\2", text)

    # Normalize carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace runs of horizontal whitespace with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Replace more than 2 consecutive newlines with 2 newlines (preserves paragraph boundaries)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_pages_from_pdf(file_input) -> List[Tuple[int, str]]:
    """Extracts and cleans text from each page of a PDF file.
    
    Args:
        file_input: Either a file-like object with a `.name` attribute (such as Gradio's
                    tempfile wrapper) or a string filesystem path.

    Returns:
        List of tuples: (page_number: int, cleaned_text: str), where page_number is 1-indexed.
        
    Raises:
        ValueError: If file is missing or has an unreadable format.
    """
    if file_input is None:
        return []

    # Support both Gradio NamedString / tempfile and direct file paths
    file_path = getattr(file_input, "name", file_input)
    if not isinstance(file_path, str):
        return []

    pages_data: List[Tuple[int, str]] = []

    try:
        with pdfplumber.open(file_path) as pdf:
            if not pdf.pages:
                return []

            for page_idx, page in enumerate(pdf.pages, start=1):
                raw_text = page.extract_text() or ""
                cleaned = clean_page_text(raw_text)
                if cleaned:
                    pages_data.append((page_idx, cleaned))

    except Exception as e:
        print(f"[text_processor] Error extracting PDF text: {e}")
        raise ValueError(f"Failed to read PDF file: {e}")

    return pages_data


def get_full_cleaned_text(pages_data: List[Tuple[int, str]]) -> str:
    """Concatenates all cleaned pages into a single document string.
    
    Useful for global document operations such as summarization and quiz generation.
    """
    return "\n\n".join([page_text for _, page_text in pages_data])
