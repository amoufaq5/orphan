# src/rag/embed_faiss.py
import os, json, argparse, hashlib, faiss, pandas as pd
from tqdm import tqdm
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np

def load_corpus(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def chunk(text: str, max_words=220, overlap=40) -> List[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    out = []
    i = 0
    while i < len(words):
        seg = words[i:i+max_words]
        out.append(" ".join(seg))
        i += max_words - overlap
    return out

def hash_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def build_chunks(corpus_path: str, out_parquet: str) -> pd.DataFrame:
    docs = load_corpus(corpus_path)
    rows = []
    for d in tqdm(docs, desc="Chunking"):
        doc_id = d.get("id") or hash_id((d.get("title") or "") + d.get("text", "")[:200])
        base = {
            "doc_id": doc_id,
            "source": d["source"],
            "seed_term": d.get("seed_term"),
            "title": d.get("title"),
        }
        for idx, c in enumerate(chunk(d["text"])):
            rows.append({**base, "chunk_id": f"{doc_id}:{idx:03d}", "text": c})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    return df

def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def main():
    ap = argparse.ArgumentParser(description="Embed corpus into FAISS index")
    ap.add_argument("--corpus", default="./data/cleaned/corpus.jsonl")
    ap.add_argument("--chunks", default="./data/cleaned/chunks.parquet")
    ap.add_argument("--outdir", default="./out/faiss")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--rebuild", action="store_true", help="Ignore existing chunks & index; rebuild from scratch")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    idx_path = os.path.join(args.outdir, "index.faiss")
    meta_path = os.path.join(args.outdir, "chunks.parquet")
    model_path = os.path.join(args.outdir, "model.txt")

    # Build or load chunks
    if args.rebuild or not os.path.exists(args.chunks):
        df = build_chunks(args.corpus, args.chunks)
    else:
        df = pd.read_parquet(args.chunks)

    # If index exists and same model — incremental add only missing
    need_ids = df["chunk_id"].tolist()
    if (not args.rebuild) and os.path.exists(idx_path) and os.path.exists(meta_path) and os.path.exists(model_path):
        if open(model_path, "r", encoding="utf-8").read().strip() == args.model:
            existing = set(pd.read_parquet(meta_path)["chunk_id"].tolist())
            missing_df = df[~df["chunk_id"].isin(existing)]
            if missing_df.empty:
                print("Nothing new to embed; index is up to date.")
                return
            df_to_embed = missing_df
            base_df = pd.read_parquet(meta_path)
            incremental = True
        else:
            df_to_embed = df
            base_df = None
            incremental = False
    else:
        df_to_embed = df
        base_df = None
        incremental = False

    print(f"Embedding {len(df_to_embed)} chunks with {args.model} ...")
    model = SentenceTransformer(args.model)
    embs = model.encode(df_to_embed["text"].tolist(), batch_size=256, show_progress_bar=True, convert_to_numpy=True)
    embs = l2_normalize(embs).astype("float32")

    dim = embs.shape[1]
    if os.path.exists(idx_path) and incremental:
        index = faiss.read_index(idx_path)
        if index.d != dim:
            raise ValueError("Embedding dim changed; pass --rebuild")
        index.add(embs)
        faiss.write_index(index, idx_path)
        pd.concat([base_df, df_to_embed], ignore_index=True).to_parquet(meta_path, index=False)
    else:
        index = faiss.IndexFlatIP(dim)
        index.add(embs)
        faiss.write_index(index, idx_path)
        df.to_parquet(meta_path, index=False)
        with open(model_path, "w", encoding="utf-8") as f:
            f.write(args.model)

    print(f"OK: index → {idx_path}\nMeta → {meta_path}")

if __name__ == "__main__":
    main()
