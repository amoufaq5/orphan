"""
run_build_index.py
------------------
Build FAISS retrieval index from the unified corpus.

Corpus must be located in:
    data/corpus/corpus.jsonl.gz   OR data/corpus/corpus.jsonl

Index is saved to:
    data/index/faiss/
"""

import pathlib, json, gzip, sys
from src.rag.chunk import chunk_text
from src.rag.embed_faiss import embed_and_store
from src.utils.logger import get_logger

log = get_logger("run_build_index")

CORPUS_PATHS = [
    pathlib.Path("data/corpus/corpus.jsonl.gz"),
    pathlib.Path("data/corpus/corpus.jsonl")
]
INDEX_DIR = pathlib.Path("data/index/faiss")

def iter_jsonl(path: pathlib.Path):
    opener = gzip.open if path.suffix.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def load_corpus() -> list[dict]:
    for p in CORPUS_PATHS:
        if p.exists():
            log.info(f"Loading corpus from {p}")
            return list(iter_jsonl(p))
    log.error(f"No corpus file found at {CORPUS_PATHS}")
    sys.exit(1)

def main():
    records = load_corpus()
    if not records:
        log.error("Corpus is empty — nothing to index.")
        sys.exit(1)

    # Chunk text
    docs = []
    for rec in records:
        text = rec.get("text") or ""
        if not text.strip():
            continue
        for ch in chunk_text(text):
            docs.append({
                "id": rec.get("id"),
                "title": rec.get("title"),
                "source": rec.get("source"),
                "text": ch
            })

    log.info(f"Prepared {len(docs)} chunks from {len(records)} records")

    # Build index
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    embed_and_store(docs, INDEX_DIR)

    # Save metadata
    meta = {
        "corpus_file": str([p for p in CORPUS_PATHS if p.exists()][0]),
        "records": len(records),
        "chunks": len(docs)
    }
    with open(INDEX_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Index built successfully at {INDEX_DIR}")

if __name__ == "__main__":
    main()
