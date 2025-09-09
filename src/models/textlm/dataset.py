from __future__ import annotations
import glob, orjson, random, torch
from typing import List, Dict, Any, Iterable, Optional
from ..tokenizer.build_corpus import _norm  # reuse normalization
from ...utils.logger import get_logger

log = get_logger("text-ds")

class SentencePieceWrapper:
    def __init__(self, spm_model_path: str):
        import sentencepiece as spm
        self.proc = spm.SentencePieceProcessor()
        self.proc.Load(spm_model_path)
        # read true IDs from the model
        self.pad_id = self.proc.pad_id() if hasattr(self.proc, "pad_id") else self.proc.PieceToId("<pad>")
        self.unk_id = self.proc.unk_id() if hasattr(self.proc, "unk_id") else self.proc.PieceToId("<unk>")
        if self.pad_id < 0:
            self.pad_id = self.proc.PieceToId("<pad>")

    def encode(self, s: str) -> List[int]:
        return self.proc.EncodeAsIds(s)

    def pad(self, ids: List[int], length: int) -> List[int]:
        if len(ids) >= length:
            return ids[:length]
        return ids + [self.pad_id] * (length - len(ids))

def _extract_text(doc: Dict[str, Any], paths: List[str]) -> str:
    def dotget(d, path):
        cur = d
        for p in path.split("."):
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return None
        return cur if isinstance(cur, str) else None
    pieces: List[str] = []
    for p in paths:
        t = dotget(doc, p)
        if t: pieces.append(t)
    return "\n\n".join(pieces).strip()

def stream_docs(globs: List[str], text_fields: List[str], shuffle_buffer: int, max_docs: Optional[int] = None) -> Iterable[str]:
    files = []
    for g in globs: files.extend(glob.glob(g))
    files.sort()
    buf: List[str] = []
    total = 0
    for fp in files:
        with open(fp, "rb") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    row = orjson.loads(line)
                except Exception:
                    continue
                text = _extract_text(row, text_fields)
                if not text: continue
                buf.append(text)
                if len(buf) >= shuffle_buffer:
                    random.shuffle(buf)
                    for s in buf:
                        yield s
                        total += 1
                        if max_docs and total >= max_docs: return
                    buf = []
    random.shuffle(buf)
    for s in buf:
        yield s
        total += 1
        if max_docs and total >= max_docs: return

class PackedLMIterable(torch.utils.data.IterableDataset):
    def __init__(self, tokenizer: SentencePieceWrapper, globs: List[str], text_fields: List[str], max_len: int, min_len: int, pack: bool, shuffle_buffer: int, max_docs: Optional[int]):
        super().__init__()
        self.tok = tokenizer
        self.globs = globs
        self.text_fields = text_fields
        self.max_len = max_len
        self.min_len = min_len
        self.pack = pack
        self.shuffle_buffer = shuffle_buffer
        self.max_docs = max_docs

    def __iter__(self):
        cur: List[int] = []
        for text in stream_docs(self.globs, self.text_fields, self.shuffle_buffer, self.max_docs):
            ids = self.tok.encode(_norm(text))
            if len(ids) < self.min_len:
                continue
            if not self.pack:
                ids = self.tok.pad(ids, self.max_len)
                yield torch.tensor(ids, dtype=torch.long)
            else:
                # pack into contiguous max_len windows using pad as a separator
                cur.extend(ids + [self.tok.pad_id])
                while len(cur) >= self.max_len:
                    chunk = cur[:self.max_len]
                    cur = cur[self.max_len:]
                    yield torch.tensor(chunk, dtype=torch.long)
        if not self.pack and cur:
            ids = self.tok.pad(cur, self.max_len)
            yield torch.tensor(ids, dtype=torch.long)
