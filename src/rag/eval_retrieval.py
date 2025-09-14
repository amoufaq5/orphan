# src/rag/eval_retrieval.py
import os, json, argparse, joblib
from sklearn.metrics.pairwise import cosine_similarity

def main():
    ap = argparse.ArgumentParser(description="Evaluate retrieval Recall@K")
    ap.add_argument("--outdir", default="./out/rag")
    ap.add_argument("--cases", default="./data/eval/retrieval_cases.jsonl")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    pipe = joblib.load(os.path.join(args.outdir, "tfidf_pipe.joblib"))
    X = joblib.load(os.path.join(args.outdir, "tfidf_matrix.joblib"))
    meta = joblib.load(os.path.join(args.outdir, "meta.joblib"))

    # Load eval cases: {"q": "...", "must_have_id": "PMID:..."} or "must_have_term": "asthma"
    cases = []
    with open(args.cases, "r", encoding="utf-8") as f:
        for line in f:
            cases.append(json.loads(line))

    hits = 0
    for c in cases:
        q = c["q"]
        qv = pipe.transform([q])
        sims = cosine_similarity(qv, X).ravel()
        top = sims.argsort()[::-1][:args.k]

        must_id = c.get("must_have_id")
        must_term = c.get("must_have_term")
        ok = False
        for idx in top:
            id_, _, term, _ = meta[idx]
            if must_id and id_ == must_id:
                ok = True; break
            if must_term and term and must_term.lower() in term.lower():
                ok = True; break
        hits += 1 if ok else 0

    recall = hits / max(1, len(cases))
    print(f"Recall@{args.k}: {recall:.3f}  ({hits}/{len(cases)})")

if __name__ == "__main__":
    main()
