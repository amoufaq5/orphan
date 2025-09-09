from __future__ import annotations
import orjson, torch
from typing import List, Dict, Any, Iterable
from ..tokenizer.build_corpus import _norm
from .templates import render_dialog
from ..textlm.dataset import SentencePieceWrapper
from ...utils.logger import get_logger

log = get_logger("sft-ds")

def _render_sample(item: Dict[str, Any], persona: str, enforce_citations: bool) -> str:
    ctx_chunks = [f"[{i+1}] {c['text']}\nSource: {c.get('cite','source')}" for i, c in enumerate(item.get("context", []))]
    return render_dialog(persona, item["instruction"], ctx_chunks, enforce_citations)

class SFTIterable(torch.utils.data.IterableDataset):
    def __init__(self, jsonl_path: str, tokenizer: SentencePieceWrapper, max_len: int, persona: str, enforce_citations: bool):
        super().__init__()
        self.path = jsonl_path
        self.tok = tokenizer
        self.max_len = max_len
        self.persona = persona
        self.enforce_citations = enforce_citations

    def __iter__(self):
        with open(self.path, "rb") as f:
            for line in f:
                if not line.strip(): continue
                item = orjson.loads(line)
                prompt = _render_sample(item, self.persona, self.enforce_citations)
                out = item["output"].strip()
                full = prompt + out + "</s>"
                ids = self.tok.encode(_norm(full))
                # Build labels with prompt tokens masked to -100
                p_ids = self.tok.encode(_norm(prompt))
                labels = [-100]*len(p_ids) + ids[len(p_ids):]
                if len(ids) > self.max_len:
                    ids = ids[:self.max_len]
                    labels = labels[:self.max_len]
                # right-pad
                if len(ids) < self.max_len:
                    pad = [self.tok.pad_id]*(self.max_len - len(ids))
                    pad_lab = [-100]*(self.max_len - len(labels))
                    ids += pad
                    labels += pad_lab
                yield torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
