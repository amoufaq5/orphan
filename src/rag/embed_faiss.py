# src/rag/query_faiss.py
from __future__ import annotations
import argparse, pathlib, textwrap
from src.rag.embed_faiss import load_index, search
from src.utils.logger import get_logger

log = get_logger("query_faiss")
INDEX_DIR = pathlib.Path("data/index/faiss")

def _wrap(s: str, width: int = 96) -> str:
    return "\n       ".join(textwrap.wrap(s.replace("\n", " "), width=width))

def main(argv=None):
    ap = argparse.ArgumentParser(description="Query FAISS index")
    ap.add_argument("-q", "--query", required=True, help="Query text")
    ap.add_argument("-k", "--topk", type=int, default=5, help="Top-K results")
    ap.add_argument("--index_dir", default=str(INDEX_DIR), help="Index directory path")
    args = ap.parse_args(argv)

    idx_dir = pathlib.Path(args.index_dir)
    index, docs, encode, meta = load_index(idx_dir)
    log.info(f"Index loaded from {idx_dir} | encoder={meta.get('encoder_model')} | N={meta.get('count')}")

    results = search(index, docs, encode, args.query, k=args.topk)
    if not results:
        print("No results.")
        return

    print(f"\nTop-{len(results)} results for: {args.query}\n")
    for rank, score, d in results:
        title = d.get("title") or "(no title)"
        source = d.get("source") or "(source?)"
        snippet = (d.get("text") or "")[:500]
        print(f"[{rank}] score={score:.3f} | {title} | {source}")
        print(f"     {_wrap(snippet)}\n")

if __name__ == "__main__":
    main()
