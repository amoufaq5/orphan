"""
text_trainer.py
---------------
Train a GPT-style language model from scratch on your custom corpus.

- Tokenizer: out/tokenizer/ (trained via train_tokenizer.py)
- Model config: conf/model_config.yaml
- Data: data/corpus/corpus.jsonl(.gz)
- Tracks perplexity when eval_loss is available
- Saves checkpoints + final model
- Resumes automatically from last checkpoint if available
- Type coercion on YAML numbers to avoid str/float errors
- Drops ALL non-tensor columns before collation (fixes "too many dimensions 'str'")
"""

import pathlib, math, inspect
from typing import Dict, Any
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel, GPT2Config,
    GPT2TokenizerFast,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import get_last_checkpoint

from src.utils.logger import get_logger
from src.utils.config import load_yaml

log = get_logger("text_trainer")


# ---------------------------
# Helpers for safe type coercion
# ---------------------------
def _as_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return float(default) if default is not None else None

def _as_int(v, default=None):
    try:
        return int(v)
    except Exception:
        return int(default) if default is not None else None


# ---------------------------
# TrainingArguments builder (version-safe)
# ---------------------------
def _build_training_args(cfg: Dict[str, Any]) -> TrainingArguments:
    sig = inspect.signature(TrainingArguments.__init__).parameters
    def has(p: str) -> bool: return p in sig

    kw = dict(output_dir=cfg["out_dir"], overwrite_output_dir=True)

    # Core knobs
    if has("num_train_epochs"): kw["num_train_epochs"] = _as_float(cfg.get("epochs", 3), 3)
    if has("per_device_train_batch_size"): kw["per_device_train_batch_size"] = _as_int(cfg.get("train_batch_size", 8), 8)
    if has("gradient_accumulation_steps"): kw["gradient_accumulation_steps"] = _as_int(cfg.get("grad_accum", 1), 1)
    if has("per_device_eval_batch_size"): kw["per_device_eval_batch_size"] = _as_int(cfg.get("train_batch_size", 8), 8)

    # Optimizer
    if has("learning_rate"): kw["learning_rate"] = _as_float(cfg.get("lr", 5e-5), 5e-5)
    if has("weight_decay"): kw["weight_decay"] = _as_float(cfg.get("weight_decay", 0.01), 0.01)
    if has("warmup_ratio"): kw["warmup_ratio"] = _as_float(cfg.get("warmup_ratio", 0.05), 0.05)

    # Misc
    if has("seed"): kw["seed"] = _as_int(cfg.get("seed", 42), 42)
    if has("save_total_limit"): kw["save_total_limit"] = 3
    if has("report_to"): kw["report_to"] = "none"

    # Logging/saving
    if has("logging_steps"): kw["logging_steps"] = _as_int(cfg.get("logging_steps", 100), 100)
    if has("save_steps"): kw["save_steps"] = _as_int(cfg.get("save_steps", 500), 500)

    # Evaluation (newer versions)
    if has("evaluation_strategy"):
        kw["evaluation_strategy"] = "steps"
        if has("eval_steps"): kw["eval_steps"] = _as_int(cfg.get("logging_steps", 100), 100)
        if has("load_best_model_at_end"): kw["load_best_model_at_end"] = False
        if has("metric_for_best_model"): kw["metric_for_best_model"] = "loss"
        if has("greater_is_better"): kw["greater_is_better"] = False
    else:
        # Older versions: enable eval if supported
        if has("do_eval"): kw["do_eval"] = True
        if has("eval_steps"): kw["eval_steps"] = _as_int(cfg.get("logging_steps", 100), 100)

    return TrainingArguments(**(kw))


# ---------------------------
# Trainer wrapper
# ---------------------------
class TextTrainer:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

        # --- Tokenizer ---
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
        # Ensure pad token is set (some GPT tokenizers may not have one)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        log.info(f"Loaded tokenizer from {tok_path}")

        # --- Model config ---
        mc_path = pathlib.Path(cfg.get("model_config", "conf/model_config.yaml"))
        if not mc_path.exists():
            raise FileNotFoundError(f"Model config not found at {mc_path}")
        mc = load_yaml(str(mc_path))
        log.info(f"Loaded model config from {mc_path}")

        model_config = GPT2Config(**mc)
        self.model = GPT2LMHeadModel(model_config)
        log.info("Initialized model from scratch (no pretrained weights)")

        # --- Dataset ---
        train_path = pathlib.Path(cfg["train_path"])
        if not train_path.exists():
            raise FileNotFoundError(f"Train path not found: {train_path}")

        dataset = load_dataset("json", data_files=str(train_path), split="train")
        eval_split = _as_float(cfg.get("eval_split", 0.01), 0.01)
        ds = dataset.train_test_split(test_size=eval_split, seed=_as_int(cfg.get("seed", 42), 42))
        self.train_ds, self.eval_ds = ds["train"], ds["test"]

        # Optional downsampling for quick tests
        if cfg.get("max_train_samples"):
            self.train_ds = self.train_ds.select(range(_as_int(cfg["max_train_samples"])))
        if cfg.get("max_eval_samples"):
            self.eval_ds = self.eval_ds.select(range(_as_int(cfg["max_eval_samples"])))

        # --- Tokenize & DROP ALL original columns ---
        train_remove_cols = list(self.train_ds.column_names)
        eval_remove_cols  = list(self.eval_ds.column_names)

        def tok_fn(batch):
            return self.tokenizer(
                batch["text"],
                max_length=_as_int(self.cfg.get("max_length", 512), 512),
                truncation=True,
                padding="max_length",
                return_attention_mask=True,
            )

        self.train_ds = self.train_ds.map(
            tok_fn, batched=True, remove_columns=train_remove_cols
        )
        self.eval_ds = self.eval_ds.map(
            tok_fn, batched=True, remove_columns=eval_remove_cols
        )

        # --- Collator ---
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # --- Training args (version-safe) ---
        self.training_args = _build_training_args(cfg)

        # --- Trainer ---
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_ds,
            eval_dataset=self.eval_ds,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,  # keep for older versions; newer raise FutureWarning only
            compute_metrics=self.compute_metrics,
        )

        # --- Resume checkpoint if exists ---
        self.last_checkpoint = None
        output_dir = pathlib.Path(self.cfg["out_dir"])
        if output_dir.exists():
            last_ckpt = get_last_checkpoint(str(output_dir))
            if last_ckpt is not None:
                log.info(f"Resuming training from checkpoint: {last_ckpt}")
                self.last_checkpoint = last_ckpt

    # --- Metrics (perplexity) ---
    def compute_metrics(self, eval_pred):
        try:
            metrics = {}
            if hasattr(eval_pred, "metrics") and "eval_loss" in eval_pred.metrics:
                loss = eval_pred.metrics["eval_loss"]
                metrics["perplexity"] = math.exp(loss) if loss < 100 else float("inf")
            return metrics
        except Exception:
            return {}

    # --- Training entrypoint ---
    def train(self):
        log.info("Starting training…")
        self.trainer.train(resume_from_checkpoint=self.last_checkpoint)
        log.info("Saving final model…")
        self.trainer.save_model(self.cfg["out_dir"])
        self.tokenizer.save_pretrained(self.cfg["out_dir"])
        log.info(f"Model + tokenizer saved to {self.cfg['out_dir']}")
