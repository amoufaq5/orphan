# src/rag/query_faiss.py
from __future__ import annotations
import argparse, pathlib, gzip, json, heapq, textwrap
from typing import List, Dict, Tuple
import faiss
from src.rag.embed_faiss import get_encoder
from src.utils.logger import get_logger

log = get_logger("query_faiss")
DEFAULT_INDEX_ROOT = pathlib.Path("data/index/faiss")

def _wrap(s: str, width: int = 96) -> str:
    return "\n       ".join(textwrap.wrap((s or "").replace("\n", " "), width=width))

def _load_docstore(path: pathlib.Path) -> List[Dict]:
    docs: List[Dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs

def _load_shard(shard_dir: pathlib.Path):
    """Load FAISS + docstore + metadata for a shard, but don't create encoder."""
    idx_path = shard_dir / "faiss.index"
    ds_path  = shard_dir / "docstore.jsonl.gz"
    meta_path = shard_dir / "metadata.json"
    if not idx_path.exists() or not ds_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Incomplete shard at {shard_dir}")
    index = faiss.read_index(str(idx_path))
    docs  = _load_docstore(ds_path)
    meta  = json.load(open(meta_path, "r", encoding="utf-8"))
    return index, docs, meta

def main(argv=None):
    ap = argparse.ArgumentParser(description="Query multi-shard FAISS index and merge results")
    ap.add_argument("-q", "--query", required=True, help="Query text")
    ap.add_argument("-k", "--topk", type=int, default=5, help="Global Top-K to return")
    ap.add_argument("--index_dir", default=str(DEFAULT_INDEX_ROOT), help="Root index dir containing shard_* subdirs")
    args = ap.parse_args(argv)

    root = pathlib.Path(args.index_dir)
    shards = sorted([p for p in root.glob("shard_*") if p.is_dir()])
    if not shards:
        # try single-index layout as a fallback
        try:
            idx, docs, meta = _load_shard(root)
            shards = [root]
            log.info("No shard_* dirs found; using single-index layout.")
        except Exception:
            raise SystemExit(f"No shards found in {root}. Did you run run_build_index with sharding?")

    # Boot encoder from first shard's metadata
    _, _, first_meta = _load_shard(shards[0])
    encoder_name = first_meta.get("encoder_model")
    encode, dim, _ = get_encoder(encoder_name)
    log.info(f"Using encoder: {encoder_name} | shards={len(shards)}")

    # Encode query once
    qv = encode([args.query])  # [1, D]

    # Global top-K heap (min-heap)
    K = max(1, args.topk)
    heap: List[Tuple[float, Dict]] = []  # (score, doc)

    total_vectors = 0
    for sid, sd in enumerate(shards):
        index, docs, meta = _load_shard(sd)
        total_vectors += int(meta.get("count", len(docs)))

        # Search this shard
        sims, ids = index.search(qv, K)
        sims = sims[0].tolist()
        ids  = ids[0].tolist()

        for score, idx_id in zip(sims, ids):
            if idx_id < 0 or idx_id >= len(docs):
                continue
            item = docs[idx_id].copy()
            item["_shard"] = sd.name
            item["_score"] = float(score)

            if len(heap) < K:
                heapq.heappush(heap, (item["_score"], item))
            else:
                # keep highest K scores
                if item["_score"] > heap[0][0]:
                    heapq.heapreplace(heap, (item["_score"], item))

    # Extract sorted Top-K (highest score first)
    results = [heapq.heappop(heap)[1] for _ in range(len(heap))]
    results.reverse()

    # Print
    print(f"\nTop-{len(results)} results for: {args.query}\n"
          f"(encoder={encoder_name} | shards={len(shards)} | vectors≈{total_vectors:,})\n")
    for rank, d in enumerate(results, 1):
        title  = d.get("title") or "(no title)"
        source = d.get("source") or "(source?)"
        shard  = d.get("_shard")
        score  = d.get("_score", 0.0)
        snippet = (d.get("text") or "")[:600]
        print(f"[{rank}] score={score:.3f} | {title} | {source} | {shard}")
        print(f"     {_wrap(snippet)}\n")

if __name__ == "__main__":
    main()
