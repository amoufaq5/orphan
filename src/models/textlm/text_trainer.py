"""
text_trainer.py
---------------
Train a GPT-style language model from scratch on your custom corpus.

Features:
- Tokenizer: out/tokenizer/ (trained via train_tokenizer.py)
- Model config: conf/model_config.yaml
- Data: data/corpus/corpus.jsonl.gz
- Tracks perplexity on eval split
- Saves checkpoints + final model
- Auto-resumes from latest checkpoint
"""

import pathlib, math
from typing import Dict, Any
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel, GPT2Config,
    GPT2TokenizerFast,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from transformers.trainer_utils import get_last_checkpoint

from src.utils.logger import get_logger
from src.utils.config import load_yaml

log = get_logger("text_trainer")


class TextTrainer:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

        # ---------------------------
        # Load tokenizer
        # ---------------------------
        tok_path = pathlib.Path(cfg["tokenizer_path"])
        if not tok_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {tok_path}")

        self.tokenizer = GPT2TokenizerFast.from_pretrained(
            str(tok_path),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
        )
        log.info(f"Loaded tokenizer from {tok_path}")

        # ---------------------------
        # Load
