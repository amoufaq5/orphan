from __future__ import annotations
import os, sys, glob
from typing import List
from ..utils.config import load_yaml
from ..utils.logger import get_logger
from .infer import SPTokenizer, load_model, generate, build_prompt
from .rag import RAG

log = get_logger("offline")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/app.yaml")
    ap.add_argument("--persona", default=None, help="patient|doctor|pharmacist")
    ap.add_argument("--instruction", required=False, default="List key contraindications and serious warnings for ibuprofen.")
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top_p", type=float, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    rt = cfg["model_runtime"]
    persona = (args.persona or rt.get("persona") or "doctor").lower()

    # 1) tokenizer
    spm = rt["spm_model"]
    if not os.path.exists(spm):
        print(f"[offline] tokenizer not found: {spm}\n"
              f"→ run: python -m src.models.tokenizer.build_corpus --config conf/train_text.yaml\n"
              f"        python -m src.models.tokenizer.train_tokenizer_spm --config conf/train_text.yaml")
        sys.exit(2)
    tok = SPTokenizer(spm)

    # 2) model (will load your sft checkpoint if present; otherwise random weights)
    ckpt = rt.get("sft_checkpoint","")
    if not ckpt or not os.path.exists(ckpt):
        print(f"[offline] WARNING: SFT checkpoint not found at {ckpt!r}. Using random weights for a smoke test.")
        ckpt = None
    model = load_model(spm, ckpt, rt["max_seq_len"])

    # 3) RAG (optional but recommended)
    rag = RAG(args.config)
    hits = rag.retrieve(args.instruction, top_k=args.top_k)
    ctx_snips = rag.format_context(hits)
    if not hits:
        print("[offline] RAG returned 0 hits. If this is your first run, build the index:\n"
              "  python -m src.rag.run_build_index --config conf/app.yaml")

    # 4) build prompt & generate
    prompt = build_prompt(persona, args.instruction, ctx_snips, enforce_citations=True)
    text = generate(
        model, tok, prompt,
        max_new_tokens=args.max_new_tokens or rt["max_new_tokens"],
        temperature=args.temperature if args.temperature is not None else rt["temperature"],
        top_p=args.top_p if args.top_p is not None else rt["top_p"],
    )
    try:
        ans = text.split("[ASSISTANT]", 1)[1].strip()
    except Exception:
        ans = text

    # 5) print result + short evidence manifest
    print("\n=== ANSWER ===\n")
    print(ans)
    if hits:
        print("\n=== EVIDENCE ===")
        for i,(p,score) in enumerate(hits,1):
            print(f"[{i}] {p.meta.get('section')}  score={score:.3f}")
            print(f"    {p.source_url or p.doc_id}")
    print("")

if __name__ == "__main__":
    main()
