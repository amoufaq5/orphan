from __future__ import annotations
from typing import Dict, List

def simple_chunk(text: str, max_len: int = 1200, overlap: int = 100) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        j = min(len(words), i + max_len)
        chunks.append(" ".join(words[i:j]))
        i = j - overlap if j - overlap > i else j
    return chunks
