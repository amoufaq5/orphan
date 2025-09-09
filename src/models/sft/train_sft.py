from __future__ import annotations
import os, glob, time, math, argparse, torch
from ..textlm.model import OrphGPT
from ..textlm.utils import set_seed, save_checkpoint, load_checkpoint, CosineWithWarmup, num_params
from ..textlm.dataset import SentencePieceWrapper
from .dataset import SFTIterable
from ...utils.config import load_yaml
from ...utils.logger import get_logger

log = get_logger("sft-train")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/train_sft.yaml")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["sft"]["optimization"].get("seed", 1337)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer
    spm_path = cfg["sft"]["model"]["spm_model"]
    tok = SentencePieceWrapper(spm_path)

    # data
    persona = cfg["sft"]["templates"]["persona"]
    enforce_citations = bool(cfg["sft"]["templates"].get("enforce_citations", True))
    ds = SFTIterable(
        cfg["sft"]["data"]["train_path"],
        tok,
        cfg["sft"]["data"]["max_seq_len"],
        persona,
        enforce_citations,
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=int(cfg["sft"]["data"]["batch_size"]),
        num_workers=0,
    )

    # model (init from pretrain if provided)
    ckpt_in = cfg["sft"]["model"].get("ckpt_in")
    # infer vocab size from spm vocab file
    vocab_path = os.path.splitext(spm_path)[0] + ".vocab"
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_size = sum(1 for _ in f)
    # mirror dims from pretrain config or reuse train_text.yaml; for simplicity reuse train_text defaults
    from ...utils.config import load_yaml as load_train_cfg
    base_cfg = load_train_cfg("conf/train_text.yaml")
    mcfg = base_cfg["train"]["textlm"]
    model = OrphGPT(
        vocab_size=vocab_size,
        d_model=int(mcfg["model_dim"]),
        n_layers=int(mcfg["n_layers"]),
        n_heads=int(mcfg["n_heads"]),
        ffn_mult=int(mcfg["ffn_mult"]),
        max_seq_len=int(cfg["sft"]["data"]["max_seq_len"]),
        dropout=float(mcfg["dropout"]),
        norm_type=str(mcfg.get("norm","rms")),
        use_rope=bool(mcfg.get("rope", True)),
    ).to(device)

    if ckpt_in and os.path.exists(ckpt_in):
        ck = load_checkpoint(ckpt_in)
        model.load_state_dict(ck["model"], strict=False)
        log.info(f"Loaded base weights from {ckpt_in}")
    log.info(f"Model params: {num_params(model)/1e6:.2f}M")

    # opt/sched
    opt_cfg = cfg["sft"]["optimization"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg["lr"]),
        weight_decay=float(opt_cfg["weight_decay"]),
        betas=tuple(opt_cfg.get("betas",[0.9,0.95])),
    )
    total_steps = int(opt_cfg.get("epochs",1) * 100000)  # rough placeholder
    warmup_ratio = float(opt_cfg.get("warmup_ratio",0.06))
    scheduler = CosineWithWarmup(
        optimizer,
        warmup_steps=int(total_steps*warmup_ratio),
        total_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(opt_cfg.get("amp", True)))
    grad_accum = int(cfg["sft"]["data"].get("grad_accum", 1))
    max_norm = float(opt_cfg.get("max_grad_norm", 1.0))

    # resume
    out_dir = cfg["sft"]["model"]["ckpt_out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    step = 0
    best = None
    if args.resume and os.path.exists(args.resume):
        ck = load_checkpoint(args.resume)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        step = ck.get("step", 0)
        best = ck.get("best", None)
        log.info(f"Resumed from {args.resume}")

    save_every = 1000
    log_every  = 100
    keep_n     = int(cfg["sft"]["model"].get("keep_last_n", 3))
    running = 0.0
    t0 = time.time()

    model.train()
    for batch in loader:
        ids, labels = batch
        ids = ids.to(device)
        labels = labels.to(device)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            logits, _ = model(ids, labels=None)
            # shift one for teacher forcing
            logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            target = labels[:, 1:].contiguous().view(-1)
            loss = torch.nn.functional.cross_entropy(
                logits, target, ignore_index=-100
            )
        loss = loss / grad_accum
        scaler.scale(loss).backward()
        running += loss.item()

        if (step+1) % grad_accum == 0:
            if max_norm > 0:
                scaler.unscale_gradients(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        if step % log_every == 0 and step > 0:
            tokps = (ids.numel() * log_every) / (time.time() - t0 + 1e-6)
            log.info(f"step={step} sft_loss={running/log_every:.4f} tok/s≈{tokps:.0f}")
            running = 0.0
            t0 = time.time()

        if step % save_every == 0 and step > 0:
            path = os.path.join(out_dir, f"sft_step_{step}.pt")
            save_checkpoint(
                path,
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "best": best,
                    "cfg": cfg,
                },
            )
            # prune old
            snaps = sorted(glob.glob(os.path.join(out_dir, "sft_step_*.pt")), key=os.path.getmtime)
            if len(snaps) > keep_n:
                for s in snaps[:-keep_n]:
                    try: os.remove(s)
                    except Exception: pass

        step += 1

if __name__ == "__main__":
    main()
