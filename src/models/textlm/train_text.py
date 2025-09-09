from __future__ import annotations
import os, glob, time, math, argparse, torch
from typing import List
from .model import OrphGPT
from .dataset import SentencePieceWrapper, PackedLMIterable
from .utils import set_seed, num_params, save_checkpoint, load_checkpoint, CosineWithWarmup
from ...utils.config import load_yaml
from ...utils.logger import get_logger

log = get_logger("train")

def build_dataloader(cfg, spm_path: str, world_size: int = 1):
    data_cfg = cfg["train"]["data"]
    globs = [data_cfg["train_glob"]] if isinstance(data_cfg["train_glob"], str) else data_cfg["train_glob"]
    # UPDATED: read pad/unk ids from the model
    tok = SentencePieceWrapper(spm_path)
    ds = PackedLMIterable(
        tokenizer=tok,
        globs=globs,
        text_fields=list(data_cfg.get("text_fields", ["sections.body"])),
        max_len=int(data_cfg.get("max_seq_len", 2048)),
        min_len=int(data_cfg.get("min_seq_len", 16)),
        pack=bool(data_cfg.get("pack_sequences", True)),
        shuffle_buffer=int(data_cfg.get("shuffle_buffer", 20000)),
        max_docs=None,
    )
    bs = int(cfg["train"]["optimization"]["batch_size"])
    return torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=0)

def build_model(cfg, vocab_size: int):
    mcfg = cfg["train"]["textlm"]
    model = OrphGPT(
        vocab_size=vocab_size,
        d_model=int(mcfg["model_dim"]),
        n_layers=int(mcfg["n_layers"]),
        n_heads=int(mcfg["n_heads"]),
        ffn_mult=int(mcfg["ffn_mult"]),
        max_seq_len=int(mcfg["context_len"]),
        dropout=float(mcfg["dropout"]),
        norm_type=str(mcfg.get("norm", "rms")),
        use_rope=bool(mcfg.get("rope", True)),
    )
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/train_text.yaml")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    io = cfg["train"]["io"]
    tok_files = cfg["train"]["tokenizer_files"]
    spm_model = tok_files["spm_model"]
    if not os.path.exists(spm_model):
        log.error(f"SPM model not found at {spm_model}.")
        return

    # infer vocab size from .vocab file
    vocab_path = os.path.splitext(spm_model)[0] + ".vocab"
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_size = sum(1 for _ in f)

    set_seed(int(cfg["train"]["optimization"].get("seed", 1337)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data & model
    loader = build_dataloader(cfg, spm_model)
    model = build_model(cfg, vocab_size).to(device)
    log.info(f"Model params: {num_params(model)/1e6:.2f}M | device={device}")

    # opt & sched
    opt_cfg = cfg["train"]["optimization"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg["lr"]),
        weight_decay=float(opt_cfg["weight_decay"]),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.95])),
        eps=float(opt_cfg.get("eps", 1e-8)),
    )
    total_steps = int(opt_cfg.get("epochs", 1) * 100000)  # rough placeholder
    warmup_ratio = float(opt_cfg.get("warmup_ratio", 0.06))
    scheduler = CosineWithWarmup(optimizer, warmup_steps=int(total_steps*warmup_ratio), total_steps=total_steps)

    grad_accum = int(opt_cfg.get("grad_accum", 1))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(opt_cfg.get("amp", True)))
    max_norm = float(opt_cfg.get("max_grad_norm", 1.0))

    # resume
    step = 0
    best_eval = None
    if args.resume and os.path.exists(args.resume):
        ckpt = load_checkpoint(args.resume)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        step = ckpt.get("step", 0)
        best_eval = ckpt.get("best_eval", None)
        log.info(f"Resumed from {args.resume} at step={step}")

    model.train()
    save_every = int(io.get("save_every_steps", 1000))
    eval_every = int(io.get("eval_every_steps", 1000))
    log_every  = int(io.get("log_every_steps", 100))
    out_dir = io.get("out_dir", "out/text_orphgpt")
    keep_n = int(io.get("keep_last_n", 3))
    os.makedirs(out_dir, exist_ok=True)

    running = 0.0
    start = time.time()

    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            logits, loss = model(batch, labels=batch)
        loss = loss / grad_accum
        scaler.scale(loss).backward()
        running += loss.item()

        if (step + 1) % grad_accum == 0:
            if max_norm > 0:
                scaler.unscale_gradients(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        if step % log_every == 0 and step > 0:
            tokps = (batch.numel() * log_every) / (time.time() - start + 1e-6)
            import math
            ppl = math.exp(min(20.0, running / log_every))
            log.info(f"step={step} loss={running/log_every:.4f} ppl≈{ppl:.2f} tok/s≈{tokps:.0f}")
            running = 0.0
            start = time.time()

        if step % eval_every == 0 and step > 0:
            import math
            model.eval()
            with torch.no_grad():
                eval_loss, n = 0.0, 0
                for i, eb in enumerate(loader):
                    if i >= 20: break
                    eb = eb.to(device)
                    _, l = model(eb, labels=eb)
                    eval_loss += float(l.item())
                    n += 1
                eval_loss /= max(1, n)
                eval_ppl = math.exp(min(20.0, eval_loss))
                log.info(f"[eval] step={step} loss={eval_loss:.4f} ppl≈{eval_ppl:.2f}")
            model.train()

        if step % save_every == 0 and step > 0:
            ck = os.path.join(out_dir, f"step_{step}.pt")
            save_checkpoint(ck, {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "best_eval": best_eval,
                "cfg": cfg,
            })
            # prune old
            snaps = sorted(glob.glob(os.path.join(out_dir, "step_*.pt")), key=os.path.getmtime)
            if len(snaps) > keep_n:
                for s in snaps[:-keep_n]:
                    try: os.remove(s)
                    except Exception: pass

        step += 1

if __name__ == "__main__":
    main()
