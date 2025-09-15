import csv, json, gzip, pathlib, random, fnmatch
from typing import List, Dict, Any, Iterable
from src.utils.config import load_yaml
from src.utils.logger import get_logger

log = get_logger("kaggle_ingest")

# PII mask fallback
try:
    from src.utils.pii import mask_text  # type: ignore
except Exception:
    def mask_text(s: str) -> str: return s

# ---------- readers ----------
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
    aid = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                aid += 1
                continue
            section, sent = ("UNKNOWN", line) if "\t" not in line else line.split("\t", 1)
            yield {"abstract_id": aid, "section": section.strip(), "sentence": sent.strip(), "pmid": f"pm200k:{aid}"}

def _iter_excel(path: pathlib.Path, sheet: Any = None, sheets: List[str] | None = None) -> Iterable[Dict[str, Any]]:
    import pandas as pd
    def _df_iter(df):
        for rec in df.fillna("").to_dict(orient="records"):
            yield rec
    if sheets:
        for s in sheets:
            try:
                df = pd.read_excel(path, sheet_name=s, engine=None)
                yield from _df_iter(df)
            except Exception as e:
                log.warning(f"Excel sheet '{s}' missing in {path.name}: {e}")
    else:
        df = pd.read_excel(path, sheet_name=sheet or 0, engine=None)
        yield from _df_iter(df)

def _guess_format(p: pathlib.Path) -> str:
    s = p.suffix.lower()
    if s == ".csv": return "csv"
    if s == ".tsv": return "tsv"
    if s == ".jsonl": return "jsonl"
    if s in {".xlsx",".xls"}: return "xlsx"
    if s == ".txt": return "pubmed200k"
    return "csv"

def _iter_rows(path: pathlib.Path, cfg: Dict[str, Any], declared_fmt: str | None) -> Iterable[Dict[str, Any]]:
    fmt = (declared_fmt or _guess_format(path)).lower()
    if fmt == "csv":
        yield from _iter_csv(path, delimiter=cfg.get("csv_delimiter", ","))
    elif fmt == "tsv":
        yield from _iter_tsv(path)
    elif fmt == "jsonl":
        yield from _iter_jsonl(path)
    elif fmt in {"xlsx","excel"}:
        yield from _iter_excel(path, sheet=cfg.get("sheet"), sheets=cfg.get("sheets"))
    elif fmt == "pubmed200k":
        yield from _iter_pubmed200k(path)
    else:
        raise ValueError(f"Unsupported format '{fmt}' for {path}")

# ---------- mapping ----------
CANDIDATE_TEXT_KEYS = [
    "text","Text","TEXT","review","Review","content","COMMENTS","comment","Description","DESC",
    "abstract","sentence","Symptom","symptom","Measure_Description","Indications","Notes","note","title"
]

def _coalesce_text(row: Dict[str, Any], cols: Any) -> str:
    if isinstance(cols, list):
        parts = [str(row.get(c, "")).strip() for c in cols if row.get(c)]
        if parts: return "\n".join(parts)
        # fall through to auto
    elif cols:
        val = str(row.get(cols, "")).strip()
        if val: return val
    # auto-fallback: choose the *longest* string-like field
    best = ""
    for k, v in row.items():
        if isinstance(v, str) and len(v) > len(best) and (k in CANDIDATE_TEXT_KEYS or len(v) > 20):
            best = v.strip()
    return best

def _map_record(cfg: Dict[str, Any], row: Dict[str, Any], slug: str) -> Dict[str, Any]:
    mp = cfg.get("map", {})
    text = _coalesce_text(row, mp.get("text") or cfg.get("text_cols"))
    if not text:  # still empty → skip
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
        "meta": {k: row.get(k) for k in (cfg.get("extra_cols") or [])}
    }

# ---------- file resolver with globs ----------
def _resolve_paths(root_raw: pathlib.Path, slug: str, rel: str) -> List[pathlib.Path]:
    """
    Try common locations AND globbing to survive small naming differences.
    """
    candidates = []
    # exact paths
    candidates += [
        root_raw / slug / rel,
        root_raw / slug / "files" / rel,
        root_raw / rel,                       # file directly under kaggle/
        root_raw / "files" / rel
    ]
    # glob patterns (case-insensitive-ish by matching *)
    # if a bare filename like "webmd.xlsx", also try "*webmd*.xlsx" in both places
    name = pathlib.Path(rel).name
    stem = name.rsplit(".", 1)[0]
    patterns = [f"*{stem}*"]
    for pat in patterns:
        candidates += list((root_raw / slug).glob(pat + "*"))
        candidates += list((root_raw / slug / "files").glob(pat + "*"))
        candidates += list(root_raw.glob(pat + "*"))

    # dedup preserving order
    uniq, seen = [], set()
    for p in candidates:
        if p.exists() and str(p) not in seen and p.is_file():
            uniq.append(p)
            seen.add(str(p))
    return uniq

# ---------- main ----------
def ingest(catalog_path: str = "conf/kaggle_catalog.yaml",
           root_raw: str = "data/raw/kaggle",
           out_path: str = "data/clean/kaggle_merged.jsonl.gz"):
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
                log.warning(f"Missing file for slug '{slug}': tried {root/slg if (slg:=slug) else ''}/{rel} and globs")
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
