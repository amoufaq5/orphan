"""
chunk.py
--------
Utility for splitting long documents into overlapping chunks.
"""

from typing import List

def chunk_text(
    text: str,
    max_len: int = 512,
    overlap: int = 50,
    sep: str = " "
) -> List[str]:
    """
    Split `text` into overlapping chunks.

    Args:
        text: raw string
        max_len: max characters per chunk
        overlap: overlap characters between consecutive chunks
        sep: separator (default: space)

    Returns:
        List of chunk strings
    """
    if not text:
        return []

    text = text.strip().replace("\n", " ")
    if len(text) <= max_len:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_len, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks
