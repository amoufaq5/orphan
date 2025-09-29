"""
H100-Optimized Medical Language Model Trainer

This module provides a high-performance trainer specifically optimized for H100 GPUs
with medical domain adaptations and advanced optimization techniques.
"""

import os
import math
import time
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist

from transformers import (
    TrainingArguments,
    Trainer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_scheduler
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging as hf_logging

import wandb
from accelerate import Accelerator
from deepspeed import DeepSpeedEngine
import deepspeed

from ...utils.config.loader import load_config
from ...utils.logging.logger import get_logger
from .dataset import MedicalDataset, MedicalDataCollator
from .callbacks import MedicalTrainingCallbacks
from ..architecture.transformer import MedicalTransformer

logger = get_logger(__name__)


@dataclass
class H100TrainingArguments:
    """H100-specific training arguments for optimal performance."""
    
    # Model and data
    model_name_or_path: str = field(default="")
    tokenizer_name_or_path: str = field(default="")
    dataset_path: str = field(default="")
    
    # H100 optimizations
    use_flash_attention: bool = field(default=True)
    use_torch_compile: bool = field(default=True)
    compile_mode: str = field(default="max-autotune")
    
    # Training parameters
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=16)
    per_device_eval_batch_size: int = field(default=32)
    gradient_accumulation_steps: int = field(default=4)
    
    # Learning rate and scheduling
    learning_rate: float = field(default=2e-4)
    warmup_steps: int = field(default=1000)
    lr_scheduler_type: str = field(default="cosine")
    min_lr_ratio: float = field(default=0.1)
    
    # Mixed precision
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    
    # Memory optimizations
    gradient_checkpointing: bool = field(default=True)
    dataloader_num_workers: int = field(default=8)
    dataloader_pin_memory: bool = field(default=True)
    
    # Checkpointing and logging
    output_dir: str = field(default="./outputs")
    logging_steps: int = field(default=100)
    save_steps: int = field(default=1000)
    eval_steps: int = field(default=1000)
    save_total_limit: int = field(default=3)
    
    # Distributed training
    local_rank: int = field(default=-1)
    deepspeed: Optional[str] = field(default=None)
    
    # Medical-specific
    max_seq_length: int = field(default=2048)
    medical_vocab_size: int = field(default=200000)
    
    # Monitoring
    report_to: str = field(default="wandb")
    run_name: Optional[str] = field(default=None)


class H100MedicalTrainer:
    """
    High-performance medical language model trainer optimized for H100 GPUs.
    
    Features:
    - Flash Attention 2 for memory efficiency
    - Torch compilation for speed optimization
    - Mixed precision training with bfloat16
    - DeepSpeed integration for large models
    - Medical domain-specific optimizations
    - Advanced monitoring and checkpointing
    """
    
    def __init__(
        self,
        args: H100TrainingArguments,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: MedicalDataset,
        eval_dataset: Optional[MedicalDataset] = None,
        config_path: Optional[str] = None
    ):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Load H100 configuration
        self.h100_config = load_config(config_path or "conf/h100_config.yaml")
        
        # Initialize distributed training
        self._setup_distributed()
        
        # Setup device and model optimizations
        self._setup_device()
        self._setup_model_optimizations()
        
        # Initialize training components
        self._setup_data_loaders()
        self._setup_optimizer_and_scheduler()
        self._setup_mixed_precision()
        
        # Initialize monitoring
        self._setup_monitoring()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        logger.info(f"H100MedicalTrainer initialized with {self._count_parameters()} parameters")
    
    def _setup_distributed(self):
        """Setup distributed training environment."""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.args.local_rank = int(os.environ['LOCAL_RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.rank = int(os.environ['RANK'])
            
            # Initialize process group
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=self.world_size,
                rank=self.rank
            )
            
            self.is_distributed = True
            logger.info(f"Distributed training: rank {self.rank}/{self.world_size}")
        else:
            self.is_distributed = False
            self.rank = 0
            self.world_size = 1
            logger.info("Single GPU training")
    
    def _setup_device(self):
        """Setup device and CUDA optimizations."""
        if torch.cuda.is_available():
            if self.is_distributed:
                torch.cuda.set_device(self.args.local_rank)
                self.device = torch.device(f'cuda:{self.args.local_rank}')
            else:
                self.device = torch.device('cuda:0')
            
            # H100 specific optimizations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            
            logger.info(f"Using device: {self.device}")
            logger.info(f"GPU: {torch.cuda.get_device_name(self.device)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(self.device).total_memory / 1e9:.1f} GB")
        else:
            self.device = torch.device('cpu')
            logger.warning("CUDA not available, using CPU")
    
    def _setup_model_optimizations(self):
        """Setup model-specific optimizations for H100."""
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # Flash Attention optimization
        if self.args.use_flash_attention and hasattr(self.model.config, 'use_flash_attention_2'):
            self.model.config.use_flash_attention_2 = True
            logger.info("Flash Attention 2 enabled")
        
        # Torch compilation for H100
        if self.args.use_torch_compile:
            try:
                self.model = torch.compile(
                    self.model,
                    mode=self.args.compile_mode,
                    dynamic=False  # Static shapes for better optimization
                )
                logger.info(f"Model compiled with mode: {self.args.compile_mode}")
            except Exception as e:
                logger.warning(f"Torch compilation failed: {e}")
        
        # Distributed Data Parallel
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=False,
                bucket_cap_mb=25
            )
            logger.info("Model wrapped with DistributedDataParallel")
    
    def _setup_data_loaders(self):
        """Setup optimized data loaders for H100."""
        # Data collator for medical text
        self.data_collator = MedicalDataCollator(
            tokenizer=self.tokenizer,
            max_length=self.args.max_seq_length,
            pad_to_multiple_of=8,  # Optimize for tensor cores
            return_tensors="pt"
        )
        
        # Training data loader
        train_sampler = None
        if self.is_distributed:
            train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=True,
            prefetch_factor=4
        )
        
        # Evaluation data loader
        if self.eval_dataset:
            eval_sampler = None
            if self.is_distributed:
                eval_sampler = DistributedSampler(
                    self.eval_dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=False
                )
            
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                sampler=eval_sampler,
                shuffle=False,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory
            )
        
        logger.info(f"Data loaders created - Train: {len(self.train_dataloader)} batches")
    
    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler."""
        # AdamW optimizer with medical-specific parameters
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=True  # H100 optimization
        )
        
        # Learning rate scheduler
        num_training_steps = len(self.train_dataloader) * self.args.num_train_epochs // self.args.gradient_accumulation_steps
        
        self.lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        logger.info(f"Optimizer and scheduler setup - Total steps: {num_training_steps}")
    
    def _setup_mixed_precision(self):
        """Setup mixed precision training."""
        if self.args.bf16:
            self.scaler = None  # bfloat16 doesn't need scaling
            self.autocast_dtype = torch.bfloat16
            logger.info("Using bfloat16 mixed precision")
        elif self.args.fp16:
            self.scaler = GradScaler()
            self.autocast_dtype = torch.float16
            logger.info("Using float16 mixed precision")
        else:
            self.scaler = None
            self.autocast_dtype = torch.float32
            logger.info("Using float32 precision")
    
    def _setup_monitoring(self):
        """Setup training monitoring and logging."""
        if self.rank == 0 and self.args.report_to == "wandb":
            wandb.init(
                project="orphan-medical-llm",
                name=self.args.run_name or f"h100-training-{int(time.time())}",
                config=self.args.__dict__
            )
            logger.info("Wandb monitoring initialized")
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train(self):
        """Main training loop optimized for H100."""
        logger.info("Starting H100-optimized training")
        
        # Training metrics
        total_loss = 0.0
        log_loss = 0.0
        start_time = time.time()
        
        self.model.train()
        
        for epoch in range(self.args.num_train_epochs):
            self.epoch = epoch
            
            if self.is_distributed:
                self.train_dataloader.sampler.set_epoch(epoch)
            
            epoch_start_time = time.time()
            
            for step, batch in enumerate(self.train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                
                # Forward pass with autocast
                with autocast(device_type='cuda', dtype=self.autocast_dtype, enabled=self.autocast_dtype != torch.float32):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.args.gradient_accumulation_steps
                
                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                total_loss += loss.item()
                log_loss += loss.item()
                
                # Gradient accumulation step
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                    
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.args.logging_steps == 0:
                        self._log_training_metrics(log_loss, start_time)
                        log_loss = 0.0
                    
                    # Evaluation
                    if self.global_step % self.args.eval_steps == 0 and self.eval_dataset:
                        self._evaluate()
                    
                    # Checkpointing
                    if self.global_step % self.args.save_steps == 0:
                        self._save_checkpoint()
            
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        
        # Final evaluation and save
        if self.eval_dataset:
            self._evaluate()
        self._save_checkpoint(is_final=True)
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
    
    def _log_training_metrics(self, log_loss: float, start_time: float):
        """Log training metrics."""
        avg_loss = log_loss / self.args.logging_steps
        current_lr = self.lr_scheduler.get_last_lr()[0]
        elapsed_time = time.time() - start_time
        
        # Calculate throughput
        samples_per_second = (self.global_step * self.args.per_device_train_batch_size * 
                             self.args.gradient_accumulation_steps * self.world_size) / elapsed_time
        
        metrics = {
            "train/loss": avg_loss,
            "train/learning_rate": current_lr,
            "train/global_step": self.global_step,
            "train/epoch": self.epoch,
            "train/samples_per_second": samples_per_second,
            "train/gpu_memory_allocated": torch.cuda.memory_allocated(self.device) / 1e9,
            "train/gpu_memory_reserved": torch.cuda.memory_reserved(self.device) / 1e9,
        }
        
        if self.rank == 0:
            logger.info(f"Step {self.global_step}: loss={avg_loss:.4f}, lr={current_lr:.2e}, "
                       f"samples/s={samples_per_second:.1f}")
            
            if self.args.report_to == "wandb":
                wandb.log(metrics, step=self.global_step)
    
    def _evaluate(self):
        """Evaluate the model."""
        logger.info("Running evaluation...")
        
        self.model.eval()
        total_eval_loss = 0.0
        num_eval_steps = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                
                with autocast(device_type='cuda', dtype=self.autocast_dtype, enabled=self.autocast_dtype != torch.float32):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                
                total_eval_loss += loss.item()
                num_eval_steps += 1
        
        avg_eval_loss = total_eval_loss / num_eval_steps
        perplexity = math.exp(avg_eval_loss)
        
        # Save best model
        if avg_eval_loss < self.best_eval_loss:
            self.best_eval_loss = avg_eval_loss
            self._save_checkpoint(is_best=True)
        
        metrics = {
            "eval/loss": avg_eval_loss,
            "eval/perplexity": perplexity,
            "eval/global_step": self.global_step,
        }
        
        if self.rank == 0:
            logger.info(f"Evaluation: loss={avg_eval_loss:.4f}, perplexity={perplexity:.2f}")
            
            if self.args.report_to == "wandb":
                wandb.log(metrics, step=self.global_step)
        
        self.model.train()
    
    def _save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint."""
        if self.rank != 0:
            return
        
        output_dir = Path(self.args.output_dir)
        
        if is_final:
            checkpoint_dir = output_dir / "final"
        elif is_best:
            checkpoint_dir = output_dir / "best"
        else:
            checkpoint_dir = output_dir / f"checkpoint-{self.global_step}"
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        torch.save({
            'global_step': self.global_step,
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'best_eval_loss': self.best_eval_loss,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
        }, checkpoint_dir / "training_state.pt")
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
        
        # Clean up old checkpoints
        if not (is_best or is_final):
            self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Clean up old checkpoints to save disk space."""
        output_dir = Path(self.args.output_dir)
        checkpoints = [d for d in output_dir.iterdir() if d.name.startswith("checkpoint-")]
        
        if len(checkpoints) > self.args.save_total_limit:
            # Sort by step number and remove oldest
            checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
            for checkpoint in checkpoints[:-self.args.save_total_limit]:
                import shutil
                shutil.rmtree(checkpoint)
                logger.info(f"Removed old checkpoint: {checkpoint}")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to training config")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    training_args = H100TrainingArguments(**config.get("training", {}))
    training_args.local_rank = args.local_rank
    
    # Initialize model, tokenizer, and datasets
    # This would be implemented based on your specific model architecture
    # model = MedicalTransformer.from_pretrained(training_args.model_name_or_path)
    # tokenizer = AutoTokenizer.from_pretrained(training_args.tokenizer_name_or_path)
    # train_dataset = MedicalDataset(training_args.dataset_path, tokenizer)
    
    # Initialize trainer
    # trainer = H100MedicalTrainer(
    #     args=training_args,
    #     model=model,
    #     tokenizer=tokenizer,
    #     train_dataset=train_dataset,
    #     config_path=args.config
    # )
    
    # Start training
    # trainer.train()


if __name__ == "__main__":
    main()
