# src/models/textlm/train_text.py
# -*- coding: utf-8 -*-
"""
Train a text language model (OrphGPT text branch).

Usage:
    python -m src.models.textlm.train_text -c conf/train_text.yaml
"""

import argparse
import logging
import os
import sys
import yaml
import torch

# Bootstrap sys.path so "src" is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from src.utils.logger import get_logger
except Exception:
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    def get_logger(name): 
        return _logging.getLogger(name)

from src.models.textlm.text_trainer import TextTrainer  # <-- now exists

log = get_logger("train_text")

def parse_args():
    p = argparse.ArgumentParser(description="Train Orph text language model")
    p.add_argument("-c", "--config", required=True, help="Path to YAML config file")
    p.add_argument("--resume", default=None, help="Optional checkpoint path to resume from")
    return p.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.config):
        log.error("Config file not found: %s", args.config)
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    log.info("Loaded config: %s", args.config)
    trainer = TextTrainer(cfg)

    if args.resume:
        log.info("Resume requested: %s (Trainer/Transformers handles checkpointing internally)", args.resume)

    trainer.train()

    final_path = os.path.join(cfg.get("output_dir", "out/text_model"), "final.pt")
    try:
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        torch.save(trainer.model.state_dict(), final_path)
        log.info("Copied model state_dict → %s", final_path)
    except Exception as e:
        log.warning("Skip extra save (%s); artifacts already saved in output_dir", e)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
        sys.exit(130)
