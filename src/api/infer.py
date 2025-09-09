from __future__ import annotations
import os, torch
from typing import Optional
from ..models.textlm.model import OrphGPT
from ..models.textlm.utils import load_checkpoint
from ..models.tokenizer.build_corpus import _norm
from ..models.sft.templates import render_dialog
from ..utils.logger import get_logger

log = get_logger("infer")

class SPTokenizer:
    def __init__(self, spm_model: str):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_model)
        # assume standard ids from our training setup
        self.pad_id = self.sp.PieceToId("<pad>")
        self.unk_id = self.sp.PieceToId("<unk>")
        if self.pad_id < 0: self.pad_id = 0
        if self.unk_id < 0: self.unk_id = 3

    def encode(self, s: str): return self.sp.EncodeAsIds(s)
    def decode(self, ids):    return self.sp.DecodeIds([i for i in ids if i != self.pad_id])

def load_model(spm_model_path: str, ckpt_path: str, max_seq_len: int):
    # infer vocab size from .vocab
    vocab_path = os.path.splitext(spm_model_path)[0] + ".vocab"
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_size = sum(1 for _ in f)
    from ..utils.config import load_yaml
    base = load_yaml("conf/train_text.yaml")["train"]["textlm"]
    model = OrphGPT(
        vocab_size=vocab_size,
        d_model=int(base["model_dim"]),
        n_layers=int(base["n_layers"]),
        n_heads=int(base["n_heads"]),
        ffn_mult=int(base["ffn_mult"]),
        max_seq_len=int(max_seq_len),
        dropout=float(base["dropout"]),
        norm_type=str(base.get("norm","rms")),
        use_rope=bool(base.get("rope", True)),
    )
    if ckpt_path and os.path.exists(ckpt_path):
        ck = load_checkpoint(ckpt_path)
        model.load_state_dict(ck["model"], strict=False)
        log.info(f"[infer] loaded checkpoint: {ckpt_path}")
    model.eval()
    return model

@torch.no_grad()
def generate(model: OrphGPT, tok: SPTokenizer, prompt: str, max_new_tokens: int = 256, temperature: float = 0.8, top_p: float = 0.9):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ids = tok.encode(_norm(prompt))
    ids = ids[-model.max_seq_len:]  # trim prompt if too long
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits, _ = model(x)
        next_logits = logits[:, -1, :]
        if temperature > 0:
            next_logits = next_logits / max(1e-6, temperature)
            probs = torch.softmax(next_logits, dim=-1)
            # nucleus (top-p) sampling
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum <= top_p
            mask[..., 0] = True
            filtered = sorted_probs * mask
            filtered = filtered / filtered.sum(dim=-1, keepdim=True)
            next_id = torch.multinomial(filtered, num_samples=1)
            next_token = sorted_idx.gather(-1, next_id)
        else:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        x = torch.cat([x, next_token], dim=1)
        if x.size(1) >= model.max_seq_len: break
    out_ids = x[0].tolist()
    return tok.decode(out_ids)

def build_prompt(persona: str, instruction: str, context_snippets: list[str] | None, enforce_citations: bool):
    return render_dialog(persona, instruction, context_snippets or [], enforce_citations)
