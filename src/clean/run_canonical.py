from __future__ import annotations
import argparse
from .to_canonical import convert_shards
from ..utils.config import load_yaml
from ..utils.logger import get_logger

log = get_logger("canonical-run")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="conf/data.yaml")
    ap.add_argument("--prefix", nargs="*", default=None, help="Shard filename prefixes to include")
    args = ap.parse_args()

    cfg = load_yaml(args.data)
    shards_dir = cfg["data"]["shards_dir"]
    out_dir = cfg["data"]["canonical_dir"]
    shard_max = cfg["data"].get("shard_max_records", 5000)
    convert_shards(shards_dir, out_dir, prefix_filters=args.prefix, shard_max=shard_max)

if __name__ == "__main__":
    main()
