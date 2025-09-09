from __future__ import annotations
import torch, torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def _rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):  # type: ignore
        return _rms_norm(x, self.weight, self.eps)

class RotaryPositionalEmbedding:
    """RoPE helper; dim must be per-head. Caches cos/sin up to max_seq_len."""
    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        assert dim % 2 == 0, "RoPE head_dim must be even"
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, T: int):
        inv = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        t = torch.arange(T, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv)  # [T, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, dim]
        self.cos_cached = emb.cos()[None, None, :, :]  # [1,1,T,dim]
        self.sin_cached = emb.sin()[None, None, :, :]  # [1,1,T,dim]

    def maybe_grow(self, needed_T: int, device: torch.device, dtype: torch.dtype):
        if needed_T > self.max_seq_len:
            self.max_seq_len = needed_T
            self._build_cache(needed_T)
        # move to model device/dtype lazily
        self.cos_cached = self.cos_cached.to(device=device, dtype=dtype)
        self.sin_cached = self.sin_cached.to(device=device, dtype=dtype)

    def apply(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        # x: [B, H, T, head_dim]
        cos = self.cos_cached[..., :seq_len, :]
        sin = self.sin_cached[..., :seq_len, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
        return (x * cos) + (x_rot * sin)

class MLP(nn.Module):
    def __init__(self, d_model: int, mult: int, dropout: float):
        super().__init__()
        hidden = d_model * mult
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))

class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, use_rope: bool, max_seq_len: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.use_rope = use_rope
        self.rope: Optional[RotaryPositionalEmbedding] = None
        if self.use_rope:
            self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

    def forward(self, x, attn_mask=None):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # [B,H,T,head_dim]
        if self.use_rope and self.rope is not None:
            # ensure cache/device/dtype are aligned & big enough
            self.rope.maybe_grow(T, x.device, x.dtype)
            q = self.rope.apply(q, T)
            k = self.rope.apply(k, T)
        # Memory-friendly SDPA with causal masking
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)  # [B,H,T,head_dim]
        y = y.transpose(1,2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))

class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int, dropout: float, norm_type: str, use_rope: bool, max_seq_len: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model) if norm_type == "rms" else nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, dropout, use_rope, max_seq_len)
        self.norm2 = RMSNorm(d_model) if norm_type == "rms" else nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, ffn_mult, dropout)
    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask=None)  # SDPA handles causal masking
        x = x + self.mlp(self.norm2(x))
        return x

class OrphGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, ffn_mult: int, max_seq_len: int, dropout: float, norm_type: str = "rms", use_rope: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = None if use_rope else nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, ffn_mult, dropout, norm_type, use_rope, max_seq_len)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model) if norm_type == "rms" else nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.max_seq_len = max_seq_len
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(T, device=idx.device).unsqueeze(0)
            x = x + self.pos_emb(pos)
        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x, attn_mask=None)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100
            )
        return logits, loss
