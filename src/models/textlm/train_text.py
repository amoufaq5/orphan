"""
train_text.py — robust launcher with loud errors + optional debug.
Usage: python -m src.models.textlm.train_text -c conf/train_text.yaml [--debug]
"""

import argparse, sys, traceback, os, pathlib
from src.utils.config import load_yaml
from src.utils.logger import get_logger
from src.models.textlm.text_trainer import TextTrainer

log = get_logger("train_text")

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="YAML config path (e.g., conf/train_text.yaml)")
    ap.add_argument("--debug", action="store_true", help="Print environment & versions and exit")
    args = ap.parse_args(argv)

    if args.debug:
        print("PYTHON:", sys.version)
        try:
            import torch, transformers, datasets
            print("torch:", torch.__version__)
            print("transformers:", transformers.__version__)
            print("datasets:", datasets.__version__)
        except Exception as e:
            print("Version check error:", repr(e))
        print("CWD:", os.getcwd())
        print("Exists config?", os.path.exists(args.config))
        print("TOKENIZER DIR exists?", os.path.exists("out/tokenizer"))
        print("MODEL CONFIG exists?", os.path.exists("conf/model_config.yaml"))
        print("CORPUS exists (gz)?", os.path.exists("data/corpus/corpus.jsonl.gz"))
        print("CORPUS exists (plain)?", os.path.exists("data/corpus/corpus.jsonl"))
        sys.exit(0)

    cfg_path = args.config
    if not os.path.exists(cfg_path):
        print(f"[train_text] Config not found: {cfg_path}")
        sys.exit(2)

    cfg = load_yaml(cfg_path)
    if not isinstance(cfg, dict):
        print(f"[train_text] Config is not a dict: {cfg_path}")
        sys.exit(3)

    # Loud log so we know we started
    log.info(f"Loaded config: {cfg_path}")

    try:
        trainer = TextTrainer(cfg)
        trainer.train()
    except SystemExit:
        raise
    except Exception as e:
        # Loud, explicit error with traceback
        print("\n[train_text] FATAL: Uncaught exception during training\n", flush=True)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
