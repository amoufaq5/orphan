from __future__ import annotations
import argparse
from .index import build_index

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/app.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    build_index(args.config, args.out)

if __name__ == "__main__":
    main()
