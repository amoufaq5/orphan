# src/rag/build_index.py
import os, json, argparse, joblib
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

def load_corpus(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def main():
    ap = argparse.ArgumentParser(description="Build TF-IDF retrieval index")
    ap.add_argument("--corpus", default="./data/cleaned/corpus.jsonl")
    ap.add_argument("--outdir", default="./out/rag")
    ap.add_argument("--max_df", type=float, default=0.95)
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--max_features", type=int, default=200000)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    docs = load_corpus(args.corpus)
    texts = [d["text"] for d in docs]
    meta = [(d["id"], d["source"], d.get("seed_term"), d.get("title")) for d in docs]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,2),
            max_df=args.max_df,
            min_df=args.min_df,
            max_features=args.max_features,
            strip_accents="unicode",
            lowercase=True,
        ))
    ])
    X = pipe.fit_transform(texts)
    joblib.dump(pipe, os.path.join(args.outdir, "tfidf_pipe.joblib"))
    joblib.dump(X, os.path.join(args.outdir, "tfidf_matrix.joblib"))
    joblib.dump(meta, os.path.join(args.outdir, "meta.joblib"))
    print(f"OK: TF-IDF index built. docs={len(docs)}")

if __name__ == "__main__":
    main()
