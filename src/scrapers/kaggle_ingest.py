import csv, json, gzip, pathlib, random
from typing import List, Dict, Any, Iterable
from src.utils.config import load_yaml
from src.utils.logger import get_logger
from src.utils.io import write_jsonl_gz  # if you don't have it, we’ll inline fallbacks
from src.utils.pii import mask_text      # fallback: identity if you prefer

log = get_logger("kaggle_ingest")

def _iter_rows(path: pathlib.Path, fmt: str) -> Iterable[Dict[str, Any]]:
    if fmt == "csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            yield from csv.DictReader(f)
    elif fmt == "tsv":
        with path.open("r", encoding="utf-8", newline="") as f:
            yield from csv.DictReader(f, delimiter="\t")
    elif fmt == "jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

def _coalesce_text(row: Dict[str, Any], cols: Any) -> str:
    if isinstance(cols, list):
        return "\n".join(str(row.get(c, "")).strip() for c in cols if row.get(c))
    return str(row.get(cols, "")).strip()

def _map_record(cfg: Dict[str, Any], row: Dict[str, Any], slug: str) -> Dict[str, Any]:
    mp = cfg.get("map", {})
    text = _coalesce_text(row, mp.get("text") or cfg.get("text_cols"))
    label = row.get(mp.get("label")) if mp.get("label") else None
    title = str(row.get(mp.get("title"))) if mp.get("title") else None

    rec = {
        "id": f"kaggle::{slug}::{hash(text) & 0xffffffff:x}",
        "title": title or (cfg.get("title") or slug),
        "text": text,
        "label": label,
        "source": f"kaggle/{slug}",
        "tags": cfg.get("tags", []),
        "lang": cfg.get("lang", "en"),
        "meta": {k: row.get(k) for k in (cfg.get("extra_cols") or [])}
    }
    return rec

def ingest(catalog_path: str = "conf/kaggle_catalog.yaml",
           root_raw: str = "data/raw/kaggle",
           out_path: str = "data/clean/kaggle_merged.jsonl.gz"):

    cat = load_yaml(catalog_path)
    defaults = cat.get("defaults", {})
    items = cat.get("datasets", [])

    out: List[Dict[str, Any]] = []
    for ds in items:
        slug = ds["slug"]
        cfg = {**defaults, **ds}
        fmt = cfg.get("format", "csv")
        max_rows = int(cfg.get("max_rows", 0))
        pii_mask = bool(cfg.get("pii_mask", True))

        for rel in cfg["files"]:
            p = pathlib.Path(root_raw) / slug / rel
            if not p.exists():
                log.warning(f"Missing file: {p}")
                continue
            n = 0
            for row in _iter_rows(p, fmt):
                rec = _map_record(cfg, row, slug)
                if pii_mask:
                    rec["text"] = mask_text(rec["text"])
                    if rec.get("title"):
                        rec["title"] = mask_text(rec["title"])
                out.append(rec)
                n += 1
                if max_rows and n >= max_rows:
                    break
        log.info(f"[{slug}] collected {sum(1 for r in out if r['source']==f'kaggle/{slug}')} records")

    if defaults.get("shuffle", True):
        random.shuffle(out)

    # write gzip jsonl
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"Wrote {out_path}  (total={len(out)})")

if __name__ == "__main__":
    ingest()
