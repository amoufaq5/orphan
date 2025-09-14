from __future__ import annotations
import os
from typing import List, Tuple
from ..utils.config import load_yaml
from ..utils.logger import get_logger
from ..rag.index import load_index, search, Passage

log = get_logger("api-rag")

class RAG:
    def __init__(self, cfg_path: str):
        self.cfg = load_yaml(cfg_path)
        self.idx_path = self.cfg["rag"]["index_path"]
        if not os.path.exists(self.idx_path):
            log.warning(f"[rag] index not found at {self.idx_path}. Build it first.")
            self.index = None
        else:
            self.index = load_index(self.idx_path)
            log.info(f"[rag] index loaded: {self.idx_path}")

    def retrieve(self, query: str, top_k: int | None = None) -> List[Tuple[Passage, float]]:
        if not self.index: return []
        k = top_k or int(self.cfg["rag"]["top_k"])
        return search(self.index, query, top_k=k)

    def format_context(self, hits: List[Tuple[Passage,float]]) -> List[str]:
        max_chars = int(self.cfg["rag"]["max_ctx_chars"])
        chunks = []
        for p, score in hits:
            txt = p.text[:max_chars].strip()
            label = p.source_url or p.doc_id
            chunks.append(f"{txt}\nSource: {label}")
        return chunks
