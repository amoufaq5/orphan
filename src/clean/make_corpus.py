# src/clean/make_corpus.py
import argparse, pathlib, json, gzip, random, sys
from typing import List, Dict, Any
from src.utils.config import load_yaml
from src.utils.logger import get_logger

log = get_logger("make_corpus")

def iter_jsonl(path: pathlib.Path):
    opener = gzip.open if path.suffix.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def build_corpus(inputs: List[pathlib.Path], out_path: pathlib.Path,
                 min_chars: int, lang: str, shuffle: bool):
    records: List[Dict[str, Any]] = []
    for p in inputs:
        if not p.exists():
            log.warning(f"Missing file skipped: {p}")
            continue
        for row in iter_jsonl(p):
            txt = (row.get("text") or "").strip()
            if len(txt) < min_chars:
                continue
            if lang and row.get("lang") and row["lang"].lower() != lang.lower():
                continue
            records.append(row)

    if shuffle:
        random.shuffle(records)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if out_path.suffix.endswith(".gz") else open
    with opener(out_path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log.info(f"Corpus built: {out_path} (total={len(records)})")

def _gather_from_dirs(dir_map: Dict[str, str]) -> List[pathlib.Path]:
    inputs: List[pathlib.Path] = []
    for name, d in dir_map.items():
        if not d:
            continue
        dpath = pathlib.Path(d)
        if not dpath.exists():
            log.warning(f"[{name}] folder not found: {dpath}")
            continue
        for f in dpath.glob("*.jsonl*"):
            inputs.append(f)
    return inputs

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pubmed_dir")
    ap.add_argument("--ctgov_dir")
    ap.add_argument("--openfda_dir")
    ap.add_argument("--kaggle_dir")
    ap.add_argument("--out")
    ap.add_argument("--min_chars", type=int, default=20)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("-c", "--config", help="Path to YAML config (e.g., conf/data.yaml)")
    args = ap.parse_args(argv)

    if args.config:
        cfg = load_yaml(args.config)
        if not isinstance(cfg, dict):
            raise SystemExit(f"[make_corpus] Config is not a dict. Loaded type={type(cfg)} from {args.config}")

        # Helpful debug in case of YAML surprises
        log.info(f"[make_corpus] Loaded keys from {args.config}: {list(cfg.keys())}")

        if "clean" not in cfg or "corpus" not in cfg:
            raise SystemExit(
                "[make_corpus] YAML missing required top-level keys. "
                "Expected 'clean' and 'corpus'. Example:\n"
                "clean:\n  pubmed: data/clean/pubmed\n  ctgov: data/clean/ctgov\n"
                "corpus:\n  out: data/corpus/corpus.jsonl.gz\n  min_chars: 20\n  lang: en"
            )

        dirs = {
            "pubmed":  cfg["clean"].get("pubmed", ""),
            "ctgov":   cfg["clean"].get("ctgov", ""),
            "openfda": cfg["clean"].get("openfda", ""),
            "kaggle":  cfg["clean"].get("kaggle", ""),
        }
        inputs = _gather_from_dirs(dirs)
        if not inputs:
            raise SystemExit("[make_corpus] No input files found under specified clean dirs.")

        out = pathlib.Path(cfg["corpus"].get("out", "data/corpus/corpus.jsonl.gz"))
        build_corpus(
            inputs,
            out,
            int(cfg["corpus"].get("min_chars", 20)),
            cfg["corpus"].get("lang", "en"),
            bool(cfg["corpus"].get("shuffle", True)),
        )
    else:
        dirs = {
            "pubmed":  args.pubmed_dir,
            "ctgov":   args.ctgov_dir,
            "openfda": args.openfda_dir,
            "kaggle":  args.kaggle_dir,
        }
        inputs = _gather_from_dirs(dirs)
        if not inputs:
            raise SystemExit("[make_corpus] No input files found; supply directories or use -c conf/data.yaml.")
        out = pathlib.Path(args.out or "data/corpus/corpus.jsonl.gz")
        build_corpus(inputs, out, args.min_chars, args.lang, args.shuffle)

if __name__ == "__main__":
    main()
