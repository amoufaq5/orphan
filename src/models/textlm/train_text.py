"""
train_text.py
-------------
Train or continue-pretrain a base TextLM model.
"""

import argparse, sys
from src.utils.config import load_yaml
from src.utils.logger import get_logger
from src.models.textlm.text_trainer import TextTrainer

log = get_logger("train_text")

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="YAML config path (e.g., conf/train_text.yaml)")
    args = ap.parse_args(argv)

    cfg_path = args.config
    cfg = load_yaml(cfg_path)
    if not isinstance(cfg, dict):
        log.error(f"Config {cfg_path} is not a valid dict.")
        sys.exit(1)

    # Validate required keys
    required = ["train_path", "out_dir", "tokenizer_path"]
    missing = [k for k in required if k not in cfg]
    if missing:
        log.error(f"Config {cfg_path} missing required keys: {missing}")
        log.error("Example:\ntrain_path: data/corpus/corpus.jsonl.gz\nout_dir: out/models/textlm\ntokenizer_path: out/tokenizer")
        sys.exit(1)

    log.info(f"Loaded config: {cfg_path}")
    trainer = TextTrainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
