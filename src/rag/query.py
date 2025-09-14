# src/rag/query.py
import os, argparse, joblib
from sklearn.metrics.pairwise import cosine_similarity

def main():
    ap = argparse.ArgumentParser(description="Query TF-IDF index")
    ap.add_argument("--outdir", default="./out/rag")
    ap.add_argument("--q", required=True, help="Query text")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    pipe = joblib.load(os.path.join(args.outdir, "tfidf_pipe.joblib"))
    X = joblib.load(os.path.join(args.outdir, "tfidf_matrix.joblib"))
    meta = joblib.load(os.path.join(args.outdir, "meta.joblib"))

    qv = pipe.transform([args.q])
    sims = cosine_similarity(qv, X).ravel()
    top = sims.argsort()[::-1][:args.k]

    print(f"Top-{args.k} for: {args.q}\n")
    for i, idx in enumerate(top, 1):
        id_, src, term, title = meta[idx]
        score = float(sims[idx])
        print(f"{i}. [{src}] {title}  (score={score:.4f})")
        print(f"   id={id_}  seed_term={term}")
        print()

if __name__ == "__main__":
    main()
