# src/scrapers/kaggle_ingest.py
from __future__ import annotations
import csv, json, gzip, pathlib, random
from typing import Dict, Any, Iterable, List
from src.utils.config import load_yaml
from src.utils.logger import get_logger

log = get_logger("kaggle_ingest")

try:
    from src.utils.pii import mask_text
except Exception:
    def mask_text(s: str) -> str: return s

# --- Readers ---
def _iter_csv(path: pathlib.Path, delimiter: str = ",") -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter=delimiter)

def _iter_tsv(path: pathlib.Path): yield from _iter_csv(path, delimiter="\t")

def _iter_jsonl(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def _iter_pubmed200k(path: pathlib.Path):
    aid = 0
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                aid += 1; continue
            section, sent = (line.split("\t",1)+["UNKNOWN"])[:2] if "\t" in line else ("UNKNOWN", line)
            yield {"abstract_id": aid,"section":section.strip(),"sentence":sent.strip(),"pmid":f"pm200k:{aid}"}

def _iter_excel(path: pathlib.Path, sheet=None, sheets=None):
    import pandas as pd
    def _df_records(df): yield from df.fillna("").to_dict(orient="records")
    if sheets:
        for s in sheets: yield from _df_records(pd.read_excel(path, sheet_name=s))
    else:
        yield from _df_records(pd.read_excel(path, sheet_name=(0 if sheet is None else sheet)))

# --- Format router ---
def _guess_format_from_suffix(p: pathlib.Path)->str:
    s=p.suffix.lower()
    return {"csv":"csv","tsv":"tsv",".jsonl":"jsonl",".xlsx":"xlsx",".xls":"xlsx",".txt":"pubmed200k"}.get(s,"csv")

def _iter_rows(path,cfg,declared_fmt):
    fmt=(declared_fmt or _guess_format_from_suffix(path)).lower()
    if fmt=="csv": yield from _iter_csv(path,cfg.get("csv_delimiter",","))
    elif fmt=="tsv": yield from _iter_tsv(path)
    elif fmt=="jsonl": yield from _iter_jsonl(path)
    elif fmt in {"xlsx","excel"}: yield from _iter_excel(path,sheet=cfg.get("sheet"),sheets=cfg.get("sheets"))
    elif fmt=="pubmed200k": yield from _iter_pubmed200k(path)
    else: raise ValueError(f"Unsupported format {fmt} for {path}")

# --- Mapping ---
def _coalesce_text(row,cols):
    if isinstance(cols,list): return "\n".join(str(row.get(c,"")).strip() for c in cols if row.get(c))
    return str(row.get(cols,"")).strip()

def _map_record(cfg,row,slug):
    mp=cfg.get("map",{})
    text=_coalesce_text(row,mp.get("text") or cfg.get("text_cols"))
    if not text: return {}
    label_key=mp.get("label") or cfg.get("label_col")
    title_key=mp.get("title")
    label=row.get(label_key) if label_key else None
    title=str(row.get(title_key)) if title_key else (cfg.get("title") or slug)
    return {"id":f"kaggle::{slug}::{hash(text)&0xffffffff:x}","title":title,"text":text,"label":label,
            "source":f"kaggle/{slug}","tags":cfg.get("tags",[]),"lang":cfg.get("lang","en"),
            "meta":{k:row.get(k) for k in (cfg.get("extra_cols") or [])}}

# --- Ingest ---
def ingest(catalog_path="conf/kaggle_catalog.yaml",root_raw="data/raw/kaggle",out_path="data/clean/kaggle_merged.jsonl.gz"):
    cat=load_yaml(catalog_path); defaults=cat.get("defaults",{}); items=cat.get("datasets",[]); out=[]
    for ds in items:
        slug=ds["slug"]; cfg={**defaults,**ds}; declared_fmt=cfg.get("format")
        max_rows=int(cfg.get("max_rows",0)); min_len=int(cfg.get("min_text_len",10))
        pii_mask_flag=bool(cfg.get("pii_mask",True)); total=0
        for rel in cfg["files"]:
            candidates=[pathlib.Path(root_raw)/slug/rel,pathlib.Path(root_raw)/slug/"files"/rel,
                        pathlib.Path(root_raw)/rel,pathlib.Path(root_raw)/"files"/rel]
            p=next((c for c in candidates if c.exists() and c.is_file()),None)
            if not p: log.warning(f"Missing file for slug '{slug}': tried {', '.join(str(c) for c in candidates)}"); continue
            n=0
            for row in _iter_rows(p,cfg,declared_fmt):
                rec=_map_record(cfg,row,slug)
                if not rec or len(rec["text"])<min_len: continue
                if pii_mask_flag:
                    rec["text"]=mask_text(rec["text"])
                    if rec.get("title"): rec["title"]=mask_text(rec["title"])
                out.append(rec); n+=1; total+=1
                if max_rows and n>=max_rows: break
        log.info(f"[{slug}] collected {total} records")
    if defaults.get("shuffle",True): random.shuffle(out)
    pathlib.Path(out_path).parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(out_path,"wt",encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    log.info(f"Wrote {out_path} (total={len(out)})")

if __name__=="__main__": ingest()
