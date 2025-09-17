# src/models/textlm/text_trainer.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
)

# Optional logger; fall back to print if your util isn't present
try:
    from src.utils.logger import get_logger
    log = get_logger("text_trainer")
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("text_trainer")


@dataclass
class TextTrainerConfig:
    model_name: str
    train_path: str
    eval_path: Optional[str] = None
    output_dir: str = "out/text_model"
    block_size: int = 512
    epochs: int = 1
    train_batch_size: int = 2
    eval_batch_size: int = 2
    lr: float = 5e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    gradient_accumulation_steps: int = 1
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500
    seed: int = 42
    fp16: bool = False
    bf16: bool = False


class JsonlLMDataset(Dataset):
    """
    Expects a JSONL with a 'text' field per line. Tokenizes into fixed-length blocks.
    """
    def __init__(self, path: str, tokenizer, block_size: int):
        import json, gzip
        self.examples: List[Dict] = []
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")
        open_fn = gzip.open if path.endswith(".gz") else open
        texts: List[str] = []
        with open_fn(path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                t = obj.get("text") or obj.get("abstract") or obj.get("content") or ""
                if t and isinstance(t, str):
                    texts.append(t)

        if not texts:
            raise ValueError(f"No 'text' found in {path}")

        joined = "\n\n".join(texts)
        toks = tokenizer(
            joined,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False,
        )["input_ids"][0]

        # chunk into block_size
        self.examples = [
            {"input_ids": toks[i:i+block_size]}
            for i in range(0, toks.size(0) - block_size + 1, block_size)
        ]
        log.info("Built dataset %s → %d blocks of size %d", os.path.basename(path), len(self.examples), block_size)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = self.examples[idx]["input_ids"]
        return {"input_ids": ids, "labels": ids.clone()}


class TextTrainer:
    def __init__(self, cfg_dict: Dict):
        # map dict → dataclass with defaults
        cfg = TextTrainerConfig(
            model_name=cfg_dict.get("model_name", "gpt2"),
            train_path=cfg_dict["train_path"],
            eval_path=cfg_dict.get("eval_path"),
            output_dir=cfg_dict.get("output_dir", "out/text_model"),
            block_size=int(cfg_dict.get("block_size", 512)),
            epochs=int(cfg_dict.get("epochs", 1)),
            train_batch_size=int(cfg_dict.get("train_batch_size", 2)),
            eval_batch_size=int(cfg_dict.get("eval_batch_size", 2)),
            lr=float(cfg_dict.get("lr", 5e-5)),
            weight_decay=float(cfg_dict.get("weight_decay", 0.0)),
            warmup_ratio=float(cfg_dict.get("warmup_ratio", 0.0)),
            gradient_accumulation_steps=int(cfg_dict.get("gradient_accumulation_steps", 1)),
            logging_steps=int(cfg_dict.get("logging_steps", 50)),
            save_steps=int(cfg_dict.get("save_steps", 500)),
            eval_steps=int(cfg_dict.get("eval_steps", 500)),
            seed=int(cfg_dict.get("seed", 42)),
            fp16=bool(cfg_dict.get("fp16", False)),
            bf16=bool(cfg_dict.get("bf16", False)),
        )
        self.cfg = cfg

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Loading tokenizer/model: %s (device=%s)", cfg.model_name, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            # GPT2-like models need an explicit pad token
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
        self.model.to(self.device)

        self.collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        os.makedirs(cfg.output_dir, exist_ok=True)

        self.train_ds = JsonlLMDataset(cfg.train_path, self.tokenizer, cfg.block_size)
        self.eval_ds = JsonlLMDataset(cfg.eval_path, self.tokenizer, cfg.block_size) if cfg.eval_path else None

        self.args = TrainingArguments(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.train_batch_size,
            per_device_eval_batch_size=cfg.eval_batch_size,
            learning_rate=cfg.lr,
            weight_decay=cfg.weight_decay,
            warmup_ratio=cfg.warmup_ratio,
            evaluation_strategy="steps" if self.eval_ds is not None else "no",
            eval_steps=cfg.eval_steps if self.eval_ds is not None else None,
            logging_steps=cfg.logging_steps,
            save_steps=cfg.save_steps,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            seed=cfg.seed,
            fp16=cfg.fp16,
            bf16=cfg.bf16,
            report_to="none",
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=self.train_ds,
            eval_dataset=self.eval_ds,
            data_collator=self.collator,
            tokenizer=self.tokenizer,
        )

    def load_checkpoint(self, ckpt_dir_or_file: str):
        # Transformers Trainer automatically resumes from --resume_from_checkpoint,
        # but we expose a method for parity with your earlier code.
        pass

    def train(self):
        log.info("Starting training for %d epoch(s)…", self.cfg.epochs)
        result = self.trainer.train()
        log.info("Training done. %s", str(result))

        # final eval (optional)
        if self.eval_ds is not None:
            metrics = self.trainer.evaluate()
            ppl = math.exp(metrics["eval_loss"]) if "eval_loss" in metrics else float("nan")
            log.info("Eval: %s | ppl=%.2f", metrics, ppl)

        self.trainer.save_model(self.cfg.output_dir)
        self.tokenizer.save_pretrained(self.cfg.output_dir)
        log.info("Saved model+tokenizer → %s", self.cfg.output_dir)
