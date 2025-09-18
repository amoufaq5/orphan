"""
text_trainer.py
---------------
Train a GPT-style language model from scratch on your custom corpus.

- Tokenizer: out/tokenizer/ (trained via train_tokenizer.py)
- Model config: conf/model_config.yaml
- Data: data/corpus/corpus.jsonl.gz
- Tracks perplexity on eval split
- Saves checkpoints + final model
- Resumes automatically from last checkpoint if available
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
        # Load model config
        # ---------------------------
        mc_path = pathlib.Path(cfg.get("model_config", "conf/model_config.yaml"))
        if not mc_path.exists():
            raise FileNotFoundError(f"Model config not found at {mc_path}")
        mc = load_yaml(str(mc_path))
        log.info(f"Loaded model config from {mc_path}")

        model_config = GPT2Config(**mc)
        self.model = GPT2LMHeadModel(model_config)
        log.info("Initialized model from scratch (no pretrained weights)")

        # ---------------------------
        # Load dataset
        # ---------------------------
        train_path = pathlib.Path(cfg["train_path"])
        if not train_path.exists():
            raise FileNotFoundError(f"Train path not found: {train_path}")

        dataset = load_dataset("json", data_files=str(train_path), split="train")
        eval_split = float(cfg.get("eval_split", 0.01))
        ds = dataset.train_test_split(test_size=eval_split, seed=cfg.get("seed", 42))
        self.train_ds, self.eval_ds = ds["train"], ds["test"]

        # Downsample for debugging (optional in YAML)
        if cfg.get("max_train_samples"):
            self.train_ds = self.train_ds.select(range(cfg["max_train_samples"]))
        if cfg.get("max_eval_samples"):
            self.eval_ds = self.eval_ds.select(range(cfg["max_eval_samples"]))

        # Tokenize
        def tok_fn(batch):
            return self.tokenizer(
                batch["text"],
                max_length=cfg.get("max_length", 512),
                truncation=True,
                padding="max_length"
            )

        self.train_ds = self.train_ds.map(tok_fn, batched=True, remove_columns=["text"])
        self.eval_ds = self.eval_ds.map(tok_fn, batched=True, remove_columns=["text"])

        # ---------------------------
        # Collator
        # ---------------------------
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # ---------------------------
        # Training args
        # ---------------------------
        self.training_args = TrainingArguments(
            output_dir=cfg["out_dir"],
            overwrite_output_dir=True,
            num_train_epochs=cfg.get("epochs", 3),
            per_device_train_batch_size=cfg.get("train_batch_size", 8),
            gradient_accumulation_steps=cfg.get("grad_accum", 1),
            per_device_eval_batch_size=cfg.get("train_batch_size", 8),
            evaluation_strategy="steps" if "evaluation_strategy" in TrainingArguments.__init__.__code__.co_varnames else "no",
            eval_steps=cfg.get("logging_steps", 100),
            save_steps=cfg.get("save_steps", 500),
            logging_steps=cfg.get("logging_steps", 100),
            learning_rate=cfg.get("lr", 5e-5),
            weight_decay=cfg.get("weight_decay", 0.01),
            warmup_ratio=cfg.get("warmup_ratio", 0.05),
            seed=cfg.get("seed", 42),
            save_total_limit=3,
            report_to="none"
        )

        # ---------------------------
        # Trainer
        # ---------------------------
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_ds,
            eval_dataset=self.eval_ds,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        # ---------------------------
        # Resume checkpoint if exists
        # ---------------------------
        self.last_checkpoint = None
        output_dir = pathlib.Path(self.cfg["out_dir"])
        if output_dir.exists():
            last_ckpt = get_last_checkpoint(str(output_dir))
            if last_ckpt is not None:
                log.info(f"Resuming training from checkpoint: {last_ckpt}")
                self.last_checkpoint = last_ckpt

    # ---------------------------
    # Metrics (perplexity)
    # ---------------------------
    def compute_metrics(self, eval_pred):
        metrics = {}
        if "eval_loss" in eval_pred.metrics:
            loss = eval_pred.metrics["eval_loss"]
            metrics["perplexity"] = math.exp(loss) if loss < 100 else float("inf")
        return metrics

    # ---------------------------
    # Training entrypoint
    # ---------------------------
    def train(self):
        log.info("Starting training…")
        self.trainer.train(resume_from_checkpoint=self.last_checkpoint)
        log.info("Saving final model…")
        self.trainer.save_model(self.cfg["out_dir"])
        self.tokenizer.save_pretrained(self.cfg["out_dir"])
        log.info(f"Model + tokenizer saved to {self.cfg['out_dir']}")
