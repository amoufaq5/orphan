from __future__ import annotations
import os, glob, pickle, orjson
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import numpy as np

from ..utils.logger import get_logger
from ..utils.io import read_jsonl, ensure_dir
from ..utils.config import load_yaml

log = get_logger("rag-index")

@dataclass
class Passage:
    pid: str
    doc_id: str
    title: str | None
    text: str
    source_url: str | None
    meta: Dict[str, Any]

def _dotget(d: Dict[str, Any], path: str) -> str | None:
    cur = d
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur if isinstance(cur, str) else None

def _chunk(text: str, max_len: int, overlap: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        j = min(len(words), i + max_len)
        chunks.append(" ".join(words[i:j]))
        i = j - overlap if (j - overlap) > i else j
    return chunks

def _collect_passages(files: List[str], include_types: List[str], section_keys: List[str], chunk_cfg: Dict[str, int]) -> List[Passage]:
    max_len = int(chunk_cfg.get("max_len", 900))
    overlap = int(chunk_cfg.get("overlap", 120))
    out: List[Passage] = []
    for fp in files:
        for row in read_jsonl(fp):
            typ = str(row.get("type", "")).lower()
            if include_types and typ not in include_types:
                continue
            title = row.get("title")
            prov = row.get("prov") or {}
            src_url = prov.get("source_url")
            did = row.get("id")
            for key in section_keys:
                txt = _dotget(row, key)
                if not txt:
                    continue
                for i, ch in enumerate(_chunk(txt, max_len=max_len, overlap=overlap)):
                    pid = f"{did}::{key}::{i:04d}"
                    out.append(Passage(pid=pid, doc_id=did, title=title, text=ch, source_url=src_url, meta={"section": key, "prov": prov, "type": typ}))
    return out

def build_index(cfg_path: str, out_path: str | None = None):
    app = load_yaml(cfg_path)
    rcfg = app["rag"]
    files: List[str] = []
    for g in rcfg["source_globs"]:
        files.extend(glob.glob(g))
    files.sort()
    if not files:
        log.warning("[rag] no canonical files matched.")
        return

    passages = _collect_passages(files, [t.lower() for t in rcfg["include_types"]], rcfg["section_keys"], rcfg["chunk"])
    if not passages:
        log.warning("[rag] no passages collected.")
        return
    texts = [p.text for p in passages]

    # TF-IDF setup (bilingual-friendly): word uni/bi-grams, light filtering
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        lowercase=True,
        strip_accents=None,
    )
    X = vec.fit_transform(texts)
    X = normalize(X, norm="l2", copy=False)

    index = {"vectorizer": vec, "matrix": X, "passages": passages}
    out_path = out_path or app["rag"]["index_path"]
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "wb") as f:
        pickle.dump(index, f)
    log.info(f"[rag] index built: {out_path} | passages={len(passages)}")

def load_index(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def search(index, query: str, top_k: int = 5) -> List[Tuple[Passage, float]]:
    """
    Rank by cosine, then de-duplicate by (doc_id, section) so we don't return
    many near-identical chunks from the same page/section.
    """
    vec = index["vectorizer"]
    M = index["matrix"]
    q = vec.transform([query])
    q = normalize(q, norm="l2", copy=False)
    sims = (M @ q.T).toarray().ravel()

    order = np.argsort(-sims)  # descending
    seen: set[tuple[str, str | None]] = set()
    hits: List[Tuple[Passage, float]] = []
    limit = max(1, top_k)
    for i in order:
        p: Passage = index["passages"][i]
        key = (p.doc_id, p.meta.get("section"))
        if key in seen:
            continue
        seen.add(key)
        hits.append((p, float(sims[i])))
        if len(hits) >= limit:
            break
    return hits
