"""
make_corpus.py
--------------
Builds a unified training corpus from cleaned datasets.

✅ Supports:
  - Config-driven mode (-c conf/data.yaml)
  - Explicit CLI args
  - Both directories (glob *.jsonl*) and single files

Example:
    python -m src.clean.make_corpus -c conf/data.yaml
    python -m src.clean.make_corpus --files data/cleaned/corpus.jsonl data/clean/kaggle_merged.jsonl --out data/corpus/corpus.jsonl.gz
"""

import argparse, pathlib, json, gzip, random, sys
from typing import List, Dict, Any
from src.utils.config import load_yaml
from src.utils.logger import get_logger

log = get_logger("make_corpus")

# -------------------------------
# Helpers
# -------------------------------

def iter_jsonl(path: pathlib.Path):
    opener = gzip.open if path.suffix.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def _gather_inputs(entries: Dict[str, str]) -> List[pathlib.Path]:
    """Accept both dirs and single files."""
    inputs: List[pathlib.Path] = []
    for name, path_str in entries.items():
        if not path_str:
            continue
        p = pathlib.Path(path_str)
        if not p.exists():
            log.warning(f"[{name}] not found: {p}")
            continue
        if p.is_dir():
            for f in p.glob("*.jsonl*"):
                inputs.append(f)
        elif p.is_file() and (".jsonl" in p.suffixes or p.suffix.endswith(".jsonl") or p.suffix.endswith(".gz")):
            inputs.append(p)
        else:
            log.warning(f"[{name}] path not recognized: {p}")
    return inputs

def build_corpus(inputs: List[pathlib.Path], out_path: pathlib.Path,
                 min_chars: int, lang: str, shuffle: bool):
    records: List[Dict[str, Any]] = []
    for p in inputs:
        log.info(f"Reading {p}")
        for row in iter_jsonl(p):
            txt = (row.get("text") or "").strip()
            if len(txt) < min_chars:
                continue
            if lang and row.get("lang") and row["lang"].lower() != lang.lower():
                continue
            records.append(row)

    if not records:
        log.error("No valid records found. Check your input files.")
        sys.exit(1)

    if shuffle:
        random.shuffle(records)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if out_path.suffix.endswith(".gz") else open
    with opener(out_path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log.info(f"Corpus built: {out_path} (total={len(records)})")

# -------------------------------
# Main
# -------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", help="Explicit list of cleaned files or dirs")
    ap.add_argument("--out", help="Output path (jsonl/jsonl.gz)")
    ap.add_argument("--min_chars", type=int, default=20)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("-c", "--config", help="Path to YAML config (e.g., conf/data.yaml)")
    args = ap.parse_args(argv)

    if args.config:
        cfg = load_yaml(args.config)
        if not isinstance(cfg, dict) or "clean" not in cfg or "corpus" not in cfg:
            raise SystemExit("[make_corpus] YAML missing required top-level keys: 'clean' and 'corpus'.")

        inputs = _gather_inputs(cfg["clean"])
        out = pathlib.Path(cfg["corpus"].get("out", "data/corpus/corpus.jsonl.gz"))
        build_corpus(
            inputs,
            out,
            int(cfg["corpus"].get("min_chars", 20)),
            cfg["corpus"].get("lang", "en"),
            bool(cfg["corpus"].get("shuffle", True)),
        )
    else:
        if not args.files:
            raise SystemExit("Must supply either -c conf/data.yaml or --files paths")
        inputs = []
        for path in args.files:
            inputs.extend(_gather_inputs({path: path}))
        out = pathlib.Path(args.out or "data/corpus/corpus.jsonl.gz")
        build_corpus(inputs, out, args.min_chars, args.lang, args.shuffle)

if __name__ == "__main__":
    main()
