# src/scrapers/kaggle_ingest.py
from __future__ import annotations

import csv
import json
import gzip
import pathlib
import random
from typing import Dict, Any, Iterable, List

from src.utils.config import load_yaml
from src.utils.logger import get_logger

log = get_logger("kaggle_ingest")

# ---- Optional PII masking (safe no-op fallback if module absent) ----
try:
    from src.utils.pii import mask_text  # type: ignore
except Exception:
    def mask_text(s: str) -> str:  # noqa: E302
        return s

# ---------------------- Row Readers ---------------------- #
def _iter_csv(path: pathlib.Path, delimiter: str = ",") -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter=delimiter)

def _iter_tsv(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    yield from _iter_csv(path, delimiter="\t")

def _iter_jsonl(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def _iter_pubmed200k(path: pathlib.Path) -> Iterable[Dict[str, Any]]:
    """
    PubMed RCT format: 'LABEL<TAB>SENTENCE', abstracts separated by blank lines.
    Emits: {"abstract_id": int, "section": str, "sentence": str, "pmid": str}
    """
    aid = 0
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                aid += 1
                continue
            if "\t" in line:
                section, sent = line.split("\t", 1)
            else:
                section, sent = "UNKNOWN", line
            yield {
                "abstract_id": aid,
                "section": section.strip(),
                "sentence": sent.strip(),
                "pmid": f"pm200k:{aid}",
            }

def _iter_excel(path: pathlib.Path, sheet: str | int | None = None, sheets: List[str | int] | None = None) -> Iterable[Dict[str, Any]]:
    """
    Excel reader using pandas (openpyxl/xlrd under the hood).
    If 'sheets' provided -> iterate and concatenate; else read 'sheet' (default 0 = first).
    """
    import pandas as pd  # heavy import here to avoid overhead when unused

    def _df_records(df):
        for rec in df.fillna("").to_dict(orient="records"):
            yield rec

    if sheets:
        for s in sheets:
            df = pd.read_excel(path, sheet_name=s, engine=None)
            yield from _df_records(df)
    else:
        df = pd.read_excel(path, sheet_name=(0 if sheet is None else sheet), engine=None)
        yield from _df_records(df)

# ---------------------- Format Router ---------------------- #
def _guess_format_from_suffix(p: pathlib.Path) -> str:
    suf = p.suffix.lower()
    if suf == ".csv":
        return "csv"
    if suf == ".tsv":
        return "tsv"
    if suf == ".jsonl":
        return "jsonl"
    if suf in {".xlsx", ".xls"}:
        return "xlsx"
    if suf == ".txt":
        return "pubmed200k"  # our default for your PubMed dumps
    return "csv"

def _iter_rows(path: pathlib.Path, cfg: Dict[str, Any], declared_fmt: str | None) -> Iterable[Dict[str, Any]]:
    fmt = (declared_fmt or _guess_format_from_suffix(path)).lower()
    if fmt == "csv":
        delim = cfg.get("csv_delimiter", ",")
        yield from _iter_csv(path, delimiter=delim)
    elif fmt == "tsv":
        yield from _iter_tsv(path)
    elif fmt == "jsonl":
        yield from _iter_jsonl(path)
    elif fmt in {"xlsx", "excel"}:
        yield from _iter_excel(path, sheet=cfg.get("sheet"), sheets=cfg.get("sheets"))
    elif fmt == "pubmed200k":
        yield from _iter_pubmed200k(path)
    else:
        raise ValueError(f"Unsupported format '{fmt}' for file {path}")

# ---------------------- Mapping Helpers ---------------------- #
def _coalesce_text(row: Dict[str, Any], cols: Any) -> str:
    if isinstance(cols, list):
        parts = [str(row.get(c, "")).strip() for c in cols if row.get(c)]
        return "\n".join(p for p in parts if p)
    return str(row.get(cols, "")).strip()

def _map_record(cfg: Dict[str, Any], row: Dict[str, Any], slug: str) -> Dict[str, Any]:
    mp = cfg.get("map", {})
    text = _coalesce_text(row, mp.get("text") or cfg.get("text_cols"))
    if not text:
        return {}
    label_key = mp.get("label") or cfg.get("label_col")
    title_key = mp.get("title")
    label = row.get(label_key) if label_key else None
    title = str(row.get(title_key)) if title_key else (cfg.get("title") or slug)

    return {
        "id": f"kaggle::{slug}::{hash(text) & 0xffffffff:x}",
        "title": title,
        "text": text,
        "label": label,
        "source": f"kaggle/{slug}",
        "tags": cfg.get("tags", []),
        "lang": cfg.get("lang", "en"),
        "meta": {k: row.get(k) for k in (cfg.get("extra_cols") or [])},
    }

# ---------------------- Ingest ---------------------- #
def ingest(
    catalog_path: str = "conf/kaggle_catalog.yaml",
    root_raw: str = "data/raw/kaggle",
    out_path: str = "data/clean/kaggle_merged.jsonl.gz",
) -> None:
    """
    Reads datasets from catalog and writes a single gzipped JSONL with canonical records.
    Search order per file:
      1) <root>/<slug>/<file>
      2) <root>/<slug>/files/<file>
      3) <root>/<file>              # for your Excel files stored directly in root
      4) <root>/files/<file>
    """
    cat = load_yaml(catalog_path)
    defaults = cat.get("defaults", {})
    items = cat.get("datasets", [])

    out: List[Dict[str, Any]] = []

    for ds in items:
        slug = ds["slug"]
        cfg = {**defaults, **ds}
        declared_fmt = cfg.get("format")
        max_rows = int(cfg.get("max_rows", 0))
        pii_mask_flag = bool(cfg.get("pii_mask", True))

        total_for_ds = 0
        for rel in cfg["files"]:
            candidates = [
                pathlib.Path(root_raw) / slug / rel,
                pathlib.Path(root_raw) / slug / "files" / rel,
                pathlib.Path(root_raw) / rel,  # root-level file (your Excel case)
                pathlib.Path(root_raw) / "files" / rel,
            ]
            p = next((c for c in candidates if c.exists() and c.is_file()), None)
            if p is None:
                log.warning(f"Missing file for slug '{slug}': tried {', '.join(str(c) for c in candidates)}")
                continue

            n = 0
            for row in _iter_rows(p, cfg, declared_fmt):
                rec = _map_record(cfg, row, slug)
                if not rec:
                    continue
                # basic quality filter: drop trivially short rows
                if len(rec["text"]) < 10:
                    continue
                if pii_mask_flag:
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

    log.info(f"Wrote {out_path} (total={len(out)})")

# ---------------------- CLI ---------------------- #
if __name__ == "__main__":
    ingest()
