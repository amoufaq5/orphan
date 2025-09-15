# src/chat/patient_cli.py
import os, sys, faiss, pandas as pd, numpy as np, textwrap, re
from typing import List, Dict
from sentence_transformers import SentenceTransformer

FAISS_DIR = os.getenv("FAISS_DIR", "./out/faiss")  # where embed_faiss stored index/model/meta
INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
META_PATH  = os.path.join(FAISS_DIR, "chunks.parquet")
MODEL_TXT  = os.path.join(FAISS_DIR, "model.txt")

EMERGENCY_PATTERNS = [
    r"severe (chest pain|trouble breathing|shortness of breath)",
    r"faint(ed|ing)", r"stroke", r"suicid(al|e|al ideation)", r"uncontrollable bleeding",
    r"confusion.*sudden", r"one side.*weak(ness)?", r"blue lips", r"seizure(s)?"
]
EMERGENCY_HINT = (
    "This tool can only provide general information and cannot diagnose you.\n"
    "If you think this may be an emergency, please call local emergency services "
    "or go to the nearest emergency department now."
)

def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def load_retriever():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH) and os.path.exists(MODEL_TXT)):
        print("FAISS index not found. Build it first:\n"
              "  python -m src.rag.embed_faiss --corpus .\\data\\cleaned\\corpus.jsonl --outdir .\\out\\faiss", file=sys.stderr)
        sys.exit(1)
    index = faiss.read_index(INDEX_PATH)
    meta  = pd.read_parquet(META_PATH)
    model_name = open(MODEL_TXT, "r", encoding="utf-8").read().strip()
    embedder = SentenceTransformer(model_name)
    return index, meta, embedder

def retrieve(index, meta: pd.DataFrame, embedder, query: str, k: int = 6) -> List[Dict]:
    qv = embedder.encode([query], convert_to_numpy=True).astype("float32")
    qv = l2_normalize(qv)
    scores, idxs = index.search(qv, k)
    idxs = idxs[0]; scores = scores[0]
    out = []
    for i, s in zip(idxs, scores):
        row = meta.iloc[int(i)]
        out.append({
            "doc_id": row.get("doc_id"),
            "title": row.get("title"),
            "seed_term": row.get("seed_term"),
            "source": row.get("source"),
            "chunk_id": row.get("chunk_id"),
            "text": row.get("text"),
            "score": float(s),
        })
    return out

def format_context(chunks: List[Dict], max_chars_per=900) -> str:
    ctx_parts = []
    for ch in chunks:
        snippet = (ch["text"] or "")[:max_chars_per]
        ctx_parts.append(
            f"[{ch['doc_id']}] {ch['title']}\n"
            f"{snippet}\n"
        )
    return "\n".join(ctx_parts)

def detect_emergency(user_text: str) -> bool:
    t = user_text.lower()
    for pat in EMERGENCY_PATTERNS:
        if re.search(pat, t):
            return True
    return False

def try_llm(prompt: str) -> str:
    """
    Optional: call OpenAI if OPENAI_API_KEY is set. Otherwise, returns "".
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""
    try:
        # OpenAI v1+ style
        from openai import OpenAI
        client = OpenAI()
        model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=650,
            messages=[
                {"role": "system", "content": (
                    "You are a careful, empathetic health information assistant. "
                    "Speak in plain language. You are NOT a doctor and this is not medical advice. "
                    "Use ONLY the provided context to answer and include inline citations as [PMID:…] or [NCT:…]. "
                    "If the context is insufficient, say so briefly and suggest what info would help. "
                    "Encourage seeing a clinician for diagnosis/treatment. Avoid drug dosing."
                )},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(LLM call failed; showing retrieved evidence only)\n{e}"

def main():
    print("Loading retrieval index…")
    index, meta, embedder = load_retriever()
    print("Ready. Type your concern in natural language. Type 'exit' to quit.\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not user:
            continue
        if user.lower() in {"exit", "quit", ":q"}:
            print("Bye!")
            break

        if detect_emergency(user):
            print(f"\n⚠️  {EMERGENCY_HINT}\n")
            # still retrieve, but show the safety note first

        # Retrieve evidence
        hits = retrieve(index, meta, embedder, user, k=6)
        if not hits:
            print("\nI couldn’t find relevant information in the local corpus yet.\n")
            continue

        context = format_context(hits)
        citations = ", ".join(sorted({h["doc_id"] for h in hits if h.get("doc_id")}))
        prompt = (
            "Context (evidence):\n"
            + context
            + "\n\nTask: Using only the context, answer the patient's question empathetically, "
              "in simple language. Include source citations inline like [PMID:xxxxx] or [NCT:xxxxx]. "
              "Finish with a brief 'What to watch for / When to seek care' section."
            + "\n\nPatient question:\n" + user
        )

        # Try LLM; if unavailable, show a retrieval-first fallback
        answer = try_llm(prompt)
        if answer:
            print("\nAssistant:\n" + answer + "\n")
        else:
            # Retrieval-only fallback
            print("\nAssistant (evidence-based summary):")
            print("I’m not connected to a language model right now, but here are the most relevant sources I found:\n")
            for i, h in enumerate(hits, 1):
                print(f"{i}. [{h['doc_id']}] {h['title']}  — source={h['source']}  (score={h['score']:.3f})")
            print("\n" + textwrap.fill(
                "This information is educational and not a medical diagnosis. If symptoms are severe, worsening, or "
                "you’re worried, please seek in-person medical care.", width=88))
            print()

if __name__ == "__main__":
    main()
