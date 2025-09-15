# src/clean/make_corpus.py
"""
Builds a unified corpus from cleaned datasets.
Supports both explicit args and -c conf/data.yaml style.
"""
import argparse, pathlib, json, gzip, random, sys
from typing import List, Dict, Any
from src.utils.config import load_yaml
from src.utils.logger import get_logger

log = get_logger("make_corpus")

def iter_jsonl(path: pathlib.Path):
    if not path.exists(): return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def build_corpus(inputs: List[pathlib.Path], out_path: pathlib.Path, min_chars: int, lang: str):
    out: List[Dict[str, Any]] = []
    for p in inputs:
        for row in iter_jsonl(p):
            txt = (row.get("text") or "").strip()
            if len(txt) < min_chars: continue
            if lang and row.get("lang") and row["lang"].lower() != lang.lower():
                continue
            out.append(row)
    random.shuffle(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if out_path.suffix.endswith(".gz") else open
    with opener(out_path, "wt", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"Wrote {out_path} (n={len(out)})")

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pubmed_dir")
    ap.add_argument("--ctgov_dir")
    ap.add_argument("--openfda_dir")
    ap.add_argument("--kaggle_dir")
    ap.add_argument("--out", required=False)
    ap.add_argument("--min_chars", type=int, default=20)
    ap.add_argument("--lang", default="en")
    ap.add_argument("-c", "--config", help="YAML config path")
    args = ap.parse_args(argv)

    if args.config:
        cfg = load_yaml(args.config)
        paths = []
        for key in ["pubmed", "ctgov", "openfda", "kaggle"]:
            d = pathlib.Path(cfg["clean"].get(key,""))
            if d and d.exists():
                for f in d.glob("*.jsonl*"):
                    paths.append(f)
        out = pathlib.Path(cfg["corpus"]["out"])
        build_corpus(paths, out, cfg["corpus"].get("min_chars", 20), cfg["corpus"].get("lang","en"))
    else:
        paths = []
        for d in [args.pubmed_dir, args.ctgov_dir, args.openfda_dir, args.kaggle_dir]:
            if d:
                for f in pathlib.Path(d).glob("*.jsonl*"):
                    paths.append(f)
        out = pathlib.Path(args.out or "data/corpus/corpus.jsonl.gz")
        build_corpus(paths, out, args.min_chars, args.lang)

if __name__ == "__main__":
    main()
