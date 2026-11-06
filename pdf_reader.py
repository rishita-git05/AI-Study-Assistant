"""PDF Reader utility (backward-compatible adapter).

Delegates underlying extraction and cleaning to the modular `rag.text_processor`.
"""

from rag.text_processor import extract_pages_from_pdf, get_full_cleaned_text


def extract_text_from_pdf(file_obj) -> str:
    """Extracts and cleans all text from a PDF file.
    
    Maintains backward compatibility with legacy scripts while utilizing
    the robust cleaning and extraction from the RAG text_processor.
    
    Args:
        file_obj: Either a file-like object or a path string.

    Returns:
        Cleaned text string of all combined pages.
    """
    if file_obj is None:
        return ""
    try:
        pages_data = extract_pages_from_pdf(file_obj)
        return get_full_cleaned_text(pages_data)
    except Exception as e:
        print(f"[pdf_reader] Error reading PDF: {e}")
        return ""
