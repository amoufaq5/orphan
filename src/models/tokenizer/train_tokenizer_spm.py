from __future__ import annotations
import os, sentencepiece as spm
from ...utils.logger import get_logger
from ...utils.config import load_yaml
from ...utils.io import ensure_dir

log = get_logger("tok-train")

def _spm_model_type(kind: str) -> str:
    kind = (kind or "").lower()
    if kind in ("spm_unigram", "unigram", "unigram_lm"):
        return "unigram"
    if kind in ("spm_bpe", "bpe"):
        return "bpe"
    # default
    return "unigram"

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/train_text.yaml")
    ap.add_argument("--corpus", default=None, help="Override path to corpus.txt")
    ap.add_argument("--out_dir", default="out/tokenizer")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    tcfg = cfg["train"]["tokenizer"]
    cset = cfg["train"]["tokenizer_corpus"]
    corpus = args.corpus or cset["output_txt"]

    ensure_dir(args.out_dir)
    model_prefix = os.path.join(args.out_dir, "orph_spm")

    vocab_size = int(tcfg.get("vocab_size", 52000))
    character_coverage = float(tcfg.get("character_coverage", 0.9995))
    model_type = _spm_model_type(tcfg.get("model_type", "spm_unigram"))
    special_tokens = tcfg.get("special_tokens") or []

    # Write user_defined_symbols (special tokens) for SPM
    # SPM will preserve these exact strings
    user_symbols = ",".join(special_tokens)

    log.info(f"[tokenizer] Training SentencePiece: {model_type} vocab={vocab_size} coverage={character_coverage}")
    spm.SentencePieceTrainer.Train(
        input=corpus,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        input_sentence_size=20000000,  # enable sampling for large corpora (no-op for small)
        shuffle_input_sentence=True,
        byte_fallback=True if tcfg.get("byte_fallback", True) else False,
        normalization_rule_name="nfkc",
        user_defined_symbols=user_symbols,
        train_extremely_large_corpus=True,
        hard_vocab_limit=False,   # allow a tiny overflow instead of failing
        max_sentence_length=8192
    )

    spm_model = model_prefix + ".model"
    spm_vocab  = model_prefix + ".vocab"
    if os.path.exists(spm_model) and os.path.exists(spm_vocab):
        log.info(f"[tokenizer] Files ready: {spm_model}, {spm_vocab}")
    else:
        log.error("[tokenizer] Expected SPM outputs not found.")

if __name__ == "__main__":
    main()
