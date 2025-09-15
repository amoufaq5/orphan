# src/rag/query_faiss.py
import os, argparse, faiss, joblib, pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def l2_normalize(x): 
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def main():
    ap = argparse.ArgumentParser(description="Query FAISS index")
    ap.add_argument("--outdir", default="./out/faiss")
    ap.add_argument("--q", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model", default=None, help="Override model (else auto from model.txt)")
    args = ap.parse_args()

    idx_path = os.path.join(args.outdir, "index.faiss")
    meta_path = os.path.join(args.outdir, "chunks.parquet")
    model_path = os.path.join(args.outdir, "model.txt")

    index = faiss.read_index(idx_path)
    meta = pd.read_parquet(meta_path)
    model_name = args.model or open(model_path, "r", encoding="utf-8").read().strip()
    model = SentenceTransformer(model_name)

    qv = model.encode([args.q], convert_to_numpy=True).astype("float32")
    qv = l2_normalize(qv)
    scores, idxs = index.search(qv, args.k)
    idxs = idxs[0]; scores = scores[0]

    print(f"Top-{args.k} for: {args.q}\n")
    for rank, (i, s) in enumerate(zip(idxs, scores), 1):
        row = meta.iloc[i]
        print(f"{rank}. [{row['source']}] {row['title']}  (score={s:.4f})")
        print(f"   id={row['doc_id']}  seed_term={row.get('seed_term')}")
        print(f"   chunk_id={row['chunk_id']}")
        print()

if __name__ == "__main__":
    main()
