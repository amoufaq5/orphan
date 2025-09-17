# -*- coding: utf-8 -*-
"""
Quick retrieval check over FAISS shards (with safe tie-breaking heap).

Usage:
  python -m src.rag.query_faiss -q "adult with sore throat 2 days, no fever"
  # optional:
  #   --top_k 10
  #   --faiss_glob "data/index/faiss/*.index"
  #   --sidecar_ext ".meta.jsonl"
  #   --model sentence-transformers/all-MiniLM-L6-v2
  #   --fallback_corpus data/corpus/corpus.jsonl

Notes:
- If FAISS shards aren't found (or faiss isn't installed), we fall back to scanning a small JSONL corpus.
- The priority queue (heapq) uses a numeric tie-breaker to avoid comparing dicts.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import heapq
import itertools
import json
import logging
import os
import sys
import time
from typing import Dict, Iterable, List, Tuple, Optional

# ---------- Logging ----------
log = logging.getLogger("query_faiss")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ---------- Optional FAISS ----------
try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # noqa: N816


def _load_sentence_encoder(model_name: str):
    """
    Lazy-import sentence-transformers and load the encoder.
    """
    from sentence_transformers import SentenceTransformer  # lazy import
    log.info("[encoder] Using sentence-transformers: %s", model_name)
    return SentenceTransformer(model_name)


def _read_jsonl(path: str) -> List[Dict]:
    """
    Read JSONL or JSONL.GZ into a list of dicts.
    """
    if not os.path.exists(path):
        return []
    open_fn = gzip.open if path.endswith(".gz") else open
    items: List[Dict] = []
    with open_fn(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                # best-effort
                continue
    return items


def _pair_index_and_sidecar(index_path: str, sidecar_ext: str) -> Optional[str]:
    """
    Given an index path like /foo/bar/shard_000.index, try to find a sidecar JSONL
    with the same stem plus sidecar_ext (default .meta.jsonl).
    """
    base, _ = os.path.splitext(index_path)
    candidate = base + sidecar_ext
    if os.path.exists(candidate):
        return candidate
    # Try plain .jsonl
    candidate2 = base + ".jsonl"
    if os.path.exists(candidate2):
        return candidate2
    return None


def _search_faiss_shard(
    index_path: str,
    meta_path: Optional[str],
    qvec,  # numpy array shape (1, d)
    top_k: int,
) -> List[Dict]:
    """
    Search a single FAISS shard. Returns list of dicts with _score and metadata.
    """
    res: List[Dict] = []
    if faiss is None:
        return res
    try:
        index = faiss.read_index(index_path)
    except Exception as e:
        log.warning("[faiss] Failed reading index %s: %s", index_path, e)
        return res

    # Optional sidecar rows for mapping IDs → docs
    rows: List[Dict] = _read_jsonl(meta_path) if meta_path else []

    try:
        D, I = index.search(qvec, top_k)
    except Exception as e:
        log.warning("[faiss] Search failed on %s: %s", index_path, e)
        return res

    # Map results to items
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        item: Dict = {}
        if 0 <= idx < len(rows):
            item = dict(rows[idx])
        # Attach score and index bookkeeping
        item["_score"] = float(score)
        item["_shard"] = os.path.basename(index_path)
        item["_id"] = idx
        res.append(item)

    return res


def _cosine_similarity(a, b) -> float:
    import numpy as np
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def _fallback_bruteforce(
    query: str,
    encoder,
    corpus_path: str,
    top_k: int,
) -> List[Dict]:
    """
    If no FAISS is available, do a tiny brute-force search over JSONL corpus.
    Intended only for quick sanity checks (small files).
    """
    rows = _read_jsonl(corpus_path)
    if not rows:
        log.warning("[fallback] No corpus at %s", corpus_path)
        return []

    import numpy as np

    qvec = encoder.encode([query], convert_to_numpy=True)[0]
    heap: List[Tuple[float, int, Dict]] = []
    tie = itertools.count()

    # Optionally pre-encode docs if text field exists
    # We'll build text from common fields
    def text_of(r: Dict) -> str:
        for key in ("text", "abstract", "content", "passage", "body"):
            if key in r and isinstance(r[key], str) and r[key].strip():
                return r[key]
        # Fallback: join some fields
        title = r.get("title", "")
        desc = r.get("description", "")
        return f"{title}\n{desc}"

    # Encode in small batches to avoid memory spikes
    B = 64
    for i in range(0, len(rows), B):
        batch = rows[i : i + B]
        texts = [text_of(r) for r in batch]
        vecs = encoder.encode(texts, convert_to_numpy=True, batch_size=32, show_progress_bar=False)
        for r, v in zip(batch, vecs):
            score = _cosine_similarity(qvec, v)
            heapq.heappush(heap, (-score, next(tie), {**r, "_score": float(score), "_shard": "fallback"}))

    out: List[Dict] = []
    for _ in range(min(top_k, len(heap))):
        _, _, itm = heapq.heappop(heap)
        out.append(itm)
    return out


def main():
    p = argparse.ArgumentParser(description="FAISS shard query with tie-safe heap.")
    p.add_argument("-q", "--query", required=True, help="Natural-language query text")
    p.add_argument("--top_k", type=int, default=10, help="How many results to return")
    p.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformers model name",
    )
    p.add_argument(
        "--faiss_glob",
        default="data/index/faiss/*.index",
        help="Glob pattern for FAISS shard indexes",
    )
    p.add_argument(
        "--sidecar_ext",
        default=".meta.jsonl",
        help="Sidecar extension paired with each index (fallback to .jsonl if missing)",
    )
    p.add_argument(
        "--fallback_corpus",
        default="data/corpus/corpus.jsonl",
        help="Used if FAISS shards not found or faiss missing",
    )
    args = p.parse_args()

    t0 = time.time()
    encoder = _load_sentence_encoder(args.model)
    qvec = encoder.encode([args.query], convert_to_numpy=True)

    # Discover shards
    shard_paths = sorted(glob.glob(args.faiss_glob))
    if shard_paths and faiss is not None:
        log.info("Using encoder: %s | shards=%d", args.model, len(shard_paths))
    else:
        if not shard_paths:
            log.warning("[faiss] No shards matched: %s", args.faiss_glob)
        if faiss is None:
            log.warning("[faiss] faiss module not available; using fallback.")
        results = _fallback_bruteforce(args.query, encoder, args.fallback_corpus, args.top_k)
        _print_results(results, t0)
        return

    # -------- Search all shards with tie-safe heap --------
    heap: List[Tuple[float, int, Dict]] = []
    tie = itertools.count()
    total_hits = 0

    for idx_path in shard_paths:
        meta_path = _pair_index_and_sidecar(idx_path, args.sidecar_ext)
        shard_res = _search_faiss_shard(idx_path, meta_path, qvec, args.top_k)
        total_hits += len(shard_res)
        for item in shard_res:
            # SAFE: use (-score, counter, item) so heap never compares dicts
            heapq.heappush(heap, (-item["_score"], next(tie), item))

    # Drain heap → top_k
    out: List[Dict] = []
    k = min(args.top_k, len(heap))
    for _ in range(k):
        _, _, itm = heapq.heappop(heap)
        out.append(itm)

    _print_results(out, t0, extra=f"shards={len(shard_paths)} hits={total_hits}")


def _print_results(results: List[Dict], t0: float, extra: str = ""):
    dt = time.time() - t0
    if extra:
        log.info("Done in %.3fs | %s", dt, extra)
    else:
        log.info("Done in %.3fs", dt)

    # Pretty print
    for i, r in enumerate(results, 1):
        title = r.get("title") or r.get("id") or r.get("uid") or ""
        snippet = (
            r.get("text")
            or r.get("abstract")
            or r.get("description")
            or ""
        )
        # keep snippets short
        snippet = (snippet[:160] + "…") if len(snippet) > 160 else snippet
        score = r.get("_score", 0.0)
        shard = r.get("_shard", "")
        print(f"{i:>2}. {score:7.4f}  [{shard}]  {title}\n    {snippet}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
