from __future__ import annotations
import argparse
from ..utils.config import load_yaml
from ..utils.logger import get_logger
from .enrich_labels import enrich_dir

log = get_logger("enrich-run")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="conf/data.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.data)
    canonical_dir = cfg["data"]["canonical_dir"]
    out_dir = canonical_dir  # in-place shards with a new prefix; keep same dir
    shard_max = cfg["data"].get("shard_max_records", 5000)
    enrich_dir(canonical_dir, out_dir, shard_max=shard_max)

if __name__ == "__main__":
    main()
