from __future__ import annotations
import os, math, json, torch, random, numpy as np
from typing import Dict, Any
from ...utils.logger import get_logger

log = get_logger("text-utils")

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def num_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def save_checkpoint(path: str, payload: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)

def load_checkpoint(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")

class CosineWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                lrs.append(base_lr * step / max(1, self.warmup_steps))
            else:
                progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                lrs.append(0.5 * base_lr * (1.0 + math.cos(math.pi * progress)))
        return lrs
