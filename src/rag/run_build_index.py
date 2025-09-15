import pathlib, json, gzip, sys
from src.rag.chunk import chunk_text
from src.rag.embed_faiss import embed_and_store
from src.utils.logger import get_logger
from src.utils.config import load_yaml  # ✅ to load app.yaml

log = get_logger("run_build_index")

APP_CFG = pathlib.Path("conf/app.yaml")
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
    # Load config if available
    chunk_size, overlap = 512, 50
    if APP_CFG.exists():
        cfg = load_yaml(APP_CFG)
        rag_cfg = cfg.get("rag", {})
        chunk_size = int(rag_cfg.get("chunk_size", chunk_size))
        overlap = int(rag_cfg.get("chunk_overlap", overlap))
        log.info(f"Using chunk_size={chunk_size}, overlap={overlap}")

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
        for ch in chunk_text(text, max_len=chunk_size, overlap=overlap):
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
        "chunks": len(docs),
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
    }
    with open(INDEX_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Index built successfully at {INDEX_DIR}")

if __name__ == "__main__":
    main()
