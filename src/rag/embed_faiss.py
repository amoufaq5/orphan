# src/rag/embed_faiss.py
from __future__ import annotations
import os, json, gzip, pathlib, faiss, numpy as np
from typing import List, Dict
from src.utils.logger import get_logger

log = get_logger("embed_faiss")

# --------------------------------------------------------------------
# Encoder factory
# --------------------------------------------------------------------
def get_encoder(model_name: str | None = None):
    model_name = model_name or os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    try:
        from sentence_transformers import SentenceTransformer
        _m = SentenceTransformer(model_name)
        log.info(f"[encoder] Using sentence-transformers: {model_name}")

        def _encode(texts: List[str]) -> np.ndarray:
            vecs = _m.encode(texts, batch_size=256, convert_to_numpy=True, normalize_embeddings=True)
            return vecs.astype("float32", copy=False)

        dim = int(_m.get_sentence_embedding_dimension())
        return _encode, dim, model_name
    except Exception as e:
        log.warning(f"[encoder] ST not available ({e}); using HF fallback")
        from transformers import AutoTokenizer, AutoModel
        import torch
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModel.from_pretrained(model_name)
        mdl.eval()

        @torch.no_grad()
        def _encode(texts: List[str]) -> np.ndarray:
            outs: List[np.ndarray] = []
            for i in range(0, len(texts), 32):
                enc = tok(texts[i:i+32], padding=True, truncation=True, return_tensors="pt")
                last = mdl(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1)
                pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = pooled.cpu().numpy()
                pooled /= (np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-12)
                outs.append(pooled.astype("float32", copy=False))
            return np.vstack(outs)

        dim = int(mdl.config.hidden_size)
        return _encode, dim, model_name

# --------------------------------------------------------------------
# Resume-capable embedding + indexing
# --------------------------------------------------------------------
def embed_and_store(docs: List[Dict], out_dir: pathlib.Path,
                    model_name: str | None = None,
                    batch_size: int = 512,
                    save_every: int = 100_000) -> None:
    """
    Incremental index builder with resume support.
    - Saves after every `save_every` chunks.
    - If files already exist, resumes from last saved point.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    encode, dim, used_model = get_encoder(model_name)

    idx_path = out_dir / "faiss.index"
    ds_path  = out_dir / "docstore.jsonl.gz"
    meta_path = out_dir / "metadata.json"

    # If resuming
    start = 0
    if idx_path.exists() and ds_path.exists():
        index = faiss.read_index(str(idx_path))
        with gzip.open(ds_path, "rt", encoding="utf-8") as f:
            for _ in f:
                start += 1
        log.info(f"[resume] Found existing index/docstore with {start} vectors. Resuming…")
        f_doc = gzip.open(ds_path, "at", encoding="utf-8")  # append mode
    else:
        index = faiss.IndexFlatIP(dim)
        f_doc = gzip.open(ds_path, "wt", encoding="utf-8")
        log.info("[index] Starting fresh index build")

    total = len(docs)
    added = start
    buffer_texts, buffer_docs = [], []

    for i, d in enumerate(docs[start:], start=start):
        buffer_texts.append(d["text"])
        buffer_docs.append(d)

        if len(buffer_texts) >= batch_size:
            vecs = encode(buffer_texts)
            index.add(vecs)
            for dd in buffer_docs:
                f_doc.write(json.dumps(dd, ensure_ascii=False) + "\n")
            added += len(buffer_texts)
            buffer_texts, buffer_docs = [], []

            if added % save_every == 0:
                faiss.write_index(index, str(idx_path))
                log.info(f"[index] Saved checkpoint @ {added}/{total} vectors")

    # Flush remainder
    if buffer_texts:
        vecs = encode(buffer_texts)
        index.add(vecs)
        for dd in buffer_docs:
            f_doc.write(json.dumps(dd, ensure_ascii=False) + "\n")
        added += len(buffer_texts)

    f_doc.close()
    faiss.write_index(index, str(idx_path))

    meta = {"encoder_model": used_model, "dim": dim, "count": int(index.ntotal)}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log.info(f"[index] Complete: {index.ntotal}/{total} vectors indexed.")
