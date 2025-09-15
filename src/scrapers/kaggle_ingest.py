# src/scrapers/kaggle_ingest.py
# Ingest Kaggle datasets (CSV/TSV/JSONL/XLSX + PubMed200k TXT) into canonical JSONL.GZ
# Reads mapping from conf/kaggle_catalog.yaml

from __future__ import annotations
import csv, json, gzip, pathlib, random
from typing import List, Dict, Any, Iterable

from src.utils.config import load_yaml
from src.utils.logger import get_logger

log = get_logger("kaggle_ingest")

# -------------------- PII mask (no-op fallback) --------------------
try:
    from src.utils.pii import mask_text  # type: ignore
except Exception:
    def mask_text(s: str) -> str:  # noqa: E302
        return s

# --------------------------- Readers ------------------------------
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
    PubMed RCT format: 'SECTION<TAB>SENTENCE'; abstracts separated by blank lines.
    Emits rows: {'abstract_id', 'section', 'sentence', 'pmid'}
    """
    aid = 0
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                aid += 1
                continue
            section, sent = ("UNKNOWN", line) if "\t" not in line else line.split("\t", 1)
            yield {
                "abstract_id": aid,
                "section": section.strip(),
                "sentence": sent.strip(),
                "pmid": f"pm200k:{aid}",
            }

# --------------------- Excel support (engine select) --------------
def _pick_excel_engine(path: pathlib.Path) -> str:
    """
    Choose a pandas Excel engine based on file suffix.
    .xlsx/.xlsm -> openpyxl
    .xls        -> xlrd
    """
    suf = path.suffix.lower()
    if suf in {".xlsx", ".xlsm"}:
        return "openpyxl"
    if suf == ".xls":
        return "xlrd"
    # default to openpyxl for unknown modern formats
    return "openpyxl"

def _iter_excel(path: pathlib.Path, sheet: Any = None, sheets: List[str] | None = None) -> Iterable[Dict[str, Any]]:
    import pandas as pd

    engine = _pick_excel_engine(path)

    def _df_iter(df):
        for rec in df.fillna("").to_dict(orient="records"):
            yield rec

    def _read(sheet_name):
        # Try chosen engine; if it fails, attempt a reasonable fallback.
        try:
            return pd.read_excel(path, sheet_name=sheet_name, engine=engine)
        except Exception:
            # Fallback order: openpyxl -> xlrd (helps if user installed only one)
            for eng in (["openpyxl", "xlrd"] if engine != "openpyxl" else ["xlrd"]):
                try:
                    return pd.read_excel(path, sheet_name=sheet_name, engine=eng)
                except Exception:
                    continue
            # Re-raise original if all fail
            raise

    if sheets:
        for s in sheets:
            try:
                df = _read(s)
                yield from _df_iter(df)
            except Exception as e:
                log.warning(f"Excel sheet '{s}' not readable in {path.name} (engine={engine}): {e}")
    else:
        df = _read(sheet or 0)  # 0 = first sheet
        yield from _df_iter(df)

# --------------------- Format routing helpers ---------------------
def _guess_format(p: pathlib.Path) -> str:
    s = p.suffix.lower()
    if s == ".csv":
        return "csv"
    if s == ".tsv":
        return "tsv"
    if s == ".jsonl":
        return "jsonl"
    if s in {".xlsx", ".xls", ".xlsm"}:
        return "xlsx"
    if s == ".txt":
        return "pubmed200k"
    return "csv"

def _iter_rows(path: pathlib.Path, cfg: Dict[str, Any], declared_fmt: str | None) -> Iterable[Dict[str, Any]]:
    fmt = (declared_fmt or _guess_format(path)).lower()
    if fmt == "csv":
        yield from _iter_csv(path, delimiter=cfg.get("csv_delimiter", ","))
    elif fmt == "tsv":
        yield from _iter_tsv(path)
    elif fmt == "jsonl":
        yield from _iter_jsonl(path)
    elif fmt in {"xlsx", "excel"}:
        yield from _iter_excel(path, sheet=cfg.get("sheet"), sheets=cfg.get("sheets"))
    elif fmt == "pubmed200k":
        yield from _iter_pubmed200k(path)
    else:
        raise ValueError(f"Unsupported format '{fmt}' for {path}")

# -------------------------- Mapping -------------------------------
CANDIDATE_TEXT_KEYS = [
    "text", "Text", "TEXT",
    "review", "Review", "content", "COMMENTS", "comment",
    "Description", "DESC",
    "abstract", "sentence",
    "Symptom", "symptom",
    "Measure_Description",
    "Indications",
    "Notes", "note",
    "title", "Title",
]

def _coalesce_text(row: Dict[str, Any], cols: Any) -> str:
    # Preferred mapping / list of columns
    if isinstance(cols, list) and cols:
        parts = [str(row.get(c, "")).strip() for c in cols if row.get(c)]
        if parts:
            return "\n".join(parts)
    elif cols:
        val = str(row.get(cols, "")).strip()
        if val:
            return val
    # Auto-fallback: choose longest plausible string field
    best = ""
    for k, v in row.items():
        if isinstance(v, str):
            vs = v.strip()
            if (k in CANDIDATE_TEXT_KEYS or len(vs) > 20) and len(vs) > len(best):
                best = vs
    return best

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
        "id": f"kaggle::{slug}::{hash(text) & 0xFFFFFFFF:x}",
        "title": title,
        "text": text,
        "label": label,
        "source": f"kaggle/{slug}",
        "tags": cfg.get("tags", []),
        "lang": cfg.get("lang", "en"),
        "meta": {k: row.get(k) for k in (cfg.get("extra_cols") or [])},
    }

# --------------------- Robust path resolver ----------------------
def _resolve_paths(root_raw: pathlib.Path, slug: str, rel: str) -> List[pathlib.Path]:
    """
    Resolve dataset file paths robustly:
      1) Try common exact locations.
      2) If not found, try safe glob patterns (without creating illegal '**').
    """
    out: List[pathlib.Path] = []
    tried: List[pathlib.Path] = []

    rel = (rel or "").strip()
    base_slug = root_raw / slug
    base_files = base_slug / "files"

    # 1) Exact candidates
    exacts = [
        base_slug / rel,
        base_files / rel,
        root_raw / rel,
        root_raw / "files" / rel,
    ]
    for p in exacts:
        tried.append(p)
        if p.exists() and p.is_file():
            out.append(p)

    if out:
        uniq, seen = [], set()
        for p in out:
            s = str(p)
            if s not in seen:
                uniq.append(p)
                seen.add(s)
        return uniq

    # 2) Safe globbing (only if rel has no wildcards and yields a non-empty stem)
    patterns: List[str] = []
    if rel and not any(ch in rel for ch in ["*", "?", "[", "]"]):
        name = pathlib.Path(rel).name
        stem = name.rsplit(".", 1)[0]
        if stem:
            patterns = [f"*{stem}*"]

    candidates: List[pathlib.Path] = []
    for pat in patterns:
        for base in (base_slug, base_files, root_raw):
            try:
                candidates += [p for p in base.glob(pat) if p.is_file()]
            except ValueError as e:
                log.warning(f"Bad glob pattern '{pat}' for slug '{slug}': {e}")

    # Dedup
    uniq, seen = [], set()
    for p in candidates:
        s = str(p)
        if s not in seen:
            uniq.append(p)
            seen.add(s)

    if not uniq:
        tried_str = ", ".join(str(p) for p in tried[:4])
        log.warning(f"Missing file for slug '{slug}': tried {tried_str} and safe globs {patterns or '[]'}")
    return uniq

# --------------------------- Ingest -------------------------------
def ingest(catalog_path: str = "conf/kaggle_catalog.yaml",
           root_raw: str = "data/raw/kaggle",
           out_path: str = "data/clean/kaggle_merged.jsonl.gz") -> None:
    cat = load_yaml(catalog_path)
    defaults = cat.get("defaults", {})
    items = cat.get("datasets", [])

    out: List[Dict[str, Any]] = []
    root = pathlib.Path(root_raw)

    for ds in items:
        slug = ds["slug"]
        cfg = {**defaults, **ds}
        declared_fmt = cfg.get("format")
        max_rows = int(cfg.get("max_rows", 0))
        pii_mask_flag = bool(cfg.get("pii_mask", True))

        total_for_ds = 0
        for rel in cfg["files"]:
            paths = _resolve_paths(root, slug, rel)
            if not paths:
                continue
            for p in paths:
                n = 0
                for row in _iter_rows(p, cfg, declared_fmt):
                    rec = _map_record(cfg, row, slug)
                    if not rec or len(rec["text"]) < 10:
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
    log.info(f"Wrote {out_path}  (total={len(out)})")

if __name__ == "__main__":
    ingest()
