# src/rag/embed_faiss.py
from __future__ import annotations
import os, json, gzip, pathlib, faiss, numpy as np
from typing import List, Dict, Tuple
from src.utils.logger import get_logger

log = get_logger("embed_faiss")

# --------------------------------------------------------------------
# Encoder factory
# --------------------------------------------------------------------

def get_encoder(model_name: str | None = None):
    """
    Returns: (encode_fn, dim, model_name)
    encode_fn: List[str] -> np.ndarray [N, D] float32
    """
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
        log.warning(f"[encoder] sentence-transformers unavailable ({e}); falling back to HF mean pooling")
        from transformers import AutoTokenizer, AutoModel
        import torch
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModel.from_pretrained(model_name)
        mdl.eval()
        log.info(f"[encoder] Using HF fallback: {model_name}")

        @torch.no_grad()
        def _encode(texts: List[str]) -> np.ndarray:
            outs: List[np.ndarray] = []
            for i in range(0, len(texts), 32):
                batch = texts[i:i+32]
                enc = tok(batch, padding=True, truncation=True, return_tensors="pt")
                last = mdl(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1)
                summed = (last * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1)
                pooled = (summed / denom).cpu().numpy()
                # normalize
                pooled = pooled / (np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-12)
                outs.append(pooled.astype("float32", copy=False))
            return np.vstack(outs)

        dim = int(mdl.config.hidden_size)
        return _encode, dim, model_name

# --------------------------------------------------------------------
# Build & store
# --------------------------------------------------------------------

def embed_and_store(docs: List[Dict], out_dir: pathlib.Path,
                    model_name: str | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    encode, dim, used_model = get_encoder(model_name)

    texts = [d["text"] for d in docs]
    log.info(f"[index] Encoding {len(texts)} chunks …")
    vecs = encode(texts)
    assert vecs.shape[1] == dim

    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    faiss.write_index(index, str(out_dir / "faiss.index"))
    log.info(f"[index] Saved FAISS index ({vecs.shape[0]} vectors, dim={dim})")

    with gzip.open(out_dir / "docstore.jsonl.gz", "wt", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    meta = {"encoder_model": used_model, "dim": dim, "count": int(vecs.shape[0])}
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log.info(f"[index] Saved metadata")

# --------------------------------------------------------------------
# Load & search
# --------------------------------------------------------------------

def _load_docstore(path: pathlib.Path) -> List[Dict]:
    docs: List[Dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs

def load_index(index_dir: pathlib.Path):
    idx_path = index_dir / "faiss.index"
    meta_path = index_dir / "metadata.json"
    ds_path   = index_dir / "docstore.jsonl.gz"
    if not idx_path.exists() or not meta_path.exists() or not ds_path.exists():
        raise FileNotFoundError(f"Incomplete index at {index_dir}")
    index = faiss.read_index(str(idx_path))
    meta = json.load(open(meta_path, "r", encoding="utf-8"))
    docs = _load_docstore(ds_path)
    enc, dim, used_model = get_encoder(meta.get("encoder_model"))
    return index, docs, enc, meta

def search(index, docs: List[Dict], encode, query: str, k: int = 5) -> List[Tuple[int, float, Dict]]:
    qv = encode([query])
    sims, ids = index.search(qv, k)
    sims = sims[0].tolist()
    ids  = ids[0].tolist()
    out = []
    for rank, (i, s) in enumerate(zip(ids, sims), 1):
        if i < 0 or i >= len(docs):
            continue
        out.append((rank, float(s), docs[i]))
    return out
