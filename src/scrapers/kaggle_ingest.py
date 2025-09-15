# src/scrapers/kaggle_ingest.py
import csv, json, gzip, pathlib, random
from typing import List, Dict, Any, Iterable, Iterator, Optional
from src.utils.config import load_yaml
from src.utils.logger import get_logger
from src.utils.pii import mask_text  # if missing, create a no-op that returns input

log = get_logger("kaggle_ingest")

def _iter_csv(path: pathlib.Path, delimiter: str = ",") -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter=delimiter)

def _iter_tsv(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    return _iter_csv(path, delimiter="\t")

def _iter_jsonl(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def _iter_pubmed200k(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    """
    Each non-empty line: SECTION<TAB>SENTENCE
    Abstracts separated by blank lines.
    Emits rows with: {"abstract_id": int, "section": str, "sentence": str, "pmid": str}
    """
    aid = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                aid += 1
                continue
            # Expect "LABEL\tsentence"
            if "\t" not in line:
                # fallback: treat entire line as sentence, unknown section
                section, sent = "UNKNOWN", line
            else:
                section, sent = line.split("\t", 1)
            yield {
                "abstract_id": aid,
                "section": section.strip(),
                "sentence": sent.strip(),
                "pmid": f"pm200k:{aid}"
            }
        # final abstract id increment not needed; last group handled by blank line

def _iter_rows(path: pathlib.Path, fmt: str, cfg: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    if fmt == "csv":
        delim = cfg.get("csv_delimiter", ",")
        yield from _iter_csv(path, delimiter=delim)
    elif fmt == "tsv":
        yield from _iter_tsv(path)
    elif fmt == "jsonl":
        yield from _iter_jsonl(path)
    elif fmt == "pubmed200k":
        yield from _iter_pubmed200k(path)
    else:
        raise ValueError(f"Unsupported format: {fmt} for {path}")

def _coalesce_text(row: Dict[str, Any], cols: Any) -> str:
    if isinstance(cols, list):
        parts = [str(row.get(c, "")).strip() for c in cols if row.get(c)]
        return "\n".join(p for p in parts if p)
    return str(row.get(cols, "")).strip()

def _map_record(cfg: Dict[str, Any], row: Dict[str, Any], slug: str) -> Dict[str, Any]:
    mp = cfg.get("map", {})
    # Prefer explicit map.text; else fall back to text_cols
    text = _coalesce_text(row, mp.get("text") or cfg.get("text_cols"))
    label_key = mp.get("label")
    label = row.get(label_key) if label_key else None
    title_key = mp.get("title")
    title = str(row.get(title_key)) if title_key else None

    # fallback title
    if not title:
        title = cfg.get("title") or slug

    rec = {
        "id": f"kaggle::{slug}::{hash(text) & 0xffffffff:x}",
        "title": title,
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

        total_for_ds = 0
        for rel in cfg["files"]:
            p = pathlib.Path(root_raw) / slug / rel
            if not p.exists():
                log.warning(f"Missing file: {p}")
                continue
            n = 0
            for row in _iter_rows(p, fmt, cfg):
                rec = _map_record(cfg, row, slug)
                if not rec["text"] or len(rec["text"]) < 10:
                    continue  # drop trivial lines
                if pii_mask:
                    rec["text"] = mask_text(rec["text"])
                    if rec.get("title"):
                        rec["title"] = mask_text(rec["title"])
                out.append(rec)
                n += 1
                total_for_ds += 1
                if max_rows and n >= max_rows:
                    break
        log.info(f"[{slug}] collected {total_for_ds} records")

    if defaults.get("shuffle", True):
        random.shuffle(out)

    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"Wrote {out_path}  (total={len(out)})")

if __name__ == "__main__":
    ingest()
