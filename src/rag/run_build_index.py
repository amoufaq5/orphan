"""
run_build_index.py
------------------
Shard-aware FAISS index builder with ETA + progress logs.

- Loads corpus from data/corpus/corpus.jsonl(.gz)
- Splits into shards (default: 1.5M chunks/shard)
- Each shard is saved under data/index/faiss/shard_xx/
"""

import pathlib, json, gzip, sys, time, math
from src.rag.chunk import chunk_text
from src.rag.embed_faiss import embed_and_store
from src.utils.logger import get_logger
from src.utils.config import load_yaml

log = get_logger("run_build_index")

APP_CFG = pathlib.Path("conf/app.yaml")
CORPUS_PATHS = [
    pathlib.Path("data/corpus/corpus.jsonl.gz"),
    pathlib.Path("data/corpus/corpus.jsonl")
]
INDEX_DIR = pathlib.Path("data/index/faiss")

MAX_PER_SHARD = 1_500_000  # vectors per shard (≈2.5GB)

# -------------------------------
# Helpers
# -------------------------------

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

# -------------------------------
# Main
# -------------------------------

def main():
    # Load config
    chunk_size, overlap = 512, 50
    if APP_CFG.exists():
        cfg = load_yaml(APP_CFG)
        rag_cfg = cfg.get("rag", {})
        chunk_size = int(rag_cfg.get("chunk_size", chunk_size))
        overlap = int(rag_cfg.get("chunk_overlap", overlap))
        log.info(f"Using chunk_size={chunk_size}, overlap={overlap}")

    records = load_corpus()
    if not records:
        log.error("Corpus empty — nothing to index.")
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

    total = len(docs)
    log.info(f"Prepared {total} chunks from {len(records)} records")

    # Sharding loop
    shard_id, start = 0, 0
    while start < total:
        end = min(start + MAX_PER_SHARD, total)
        shard_docs = docs[start:end]

        shard_dir = INDEX_DIR / f"shard_{shard_id:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"[shard {shard_id}] Building {len(shard_docs)} chunks "
                 f"({start:,} → {end:,} / {total:,})")

        t0 = time.time()
        embed_and_store(shard_docs, shard_dir, batch_size=512, save_every=50_000)
        dt = time.time() - t0

        # ETA calc
        done = end
        pct = (done / total) * 100
        shards_left = math.ceil((total - end) / MAX_PER_SHARD)
        est_total_time = (dt / len(shard_docs)) * total
        eta_total = est_total_time - (dt * (shard_id + 1))

        log.info(f"[shard {shard_id}] Done in {dt/60:.1f} min | "
                 f"Progress {pct:.2f}% | "
                 f"ETA total ~{eta_total/3600:.2f}h remaining")

        start = end
        shard_id += 1

    log.info(f"All shards complete. {total:,} chunks indexed into {shard_id} shards.")

if __name__ == "__main__":
    main()
