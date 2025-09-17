# -*- coding: utf-8 -*-
"""
Train a text language model (OrphGPT text branch).

Usage:
    python -m src.models.textlm.train_text -c conf/train_text.yaml
    # or equivalently:
    python -m src.models.textlm.train_text --config conf/train_text.yaml
"""

import argparse
import logging
import os
import sys
import yaml
import torch

from src.utils.logger import get_logger
from src.training.text_trainer import TextTrainer


log = get_logger("train_text")


def parse_args():
    p = argparse.ArgumentParser(description="Train Orph text language model")
    p.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to YAML config file",
    )
    p.add_argument(
        "--resume",
        default=None,
        help="Optional checkpoint path to resume from",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # --- Load config ---
    if not os.path.exists(args.config):
        log.error("Config file not found: %s", args.config)
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    log.info("Loaded config: %s", args.config)

    # --- Init trainer ---
    trainer = TextTrainer(cfg)

    if args.resume:
        log.info("Resuming from checkpoint: %s", args.resume)
        trainer.load_checkpoint(args.resume)

    # --- Train ---
    trainer.train()

    # --- Save final model ---
    out_dir = cfg.get("output_dir", "out/text_model")
    os.makedirs(out_dir, exist_ok=True)
    final_path = os.path.join(out_dir, "final.pt")
    torch.save(trainer.model.state_dict(), final_path)
    log.info("Training complete. Final model saved to %s", final_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warning("Interrupted by user")
        sys.exit(130)
