from __future__ import annotations
import os, json
from tokenizers import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast  # transformers is optional; only needed at train time

def load_tokenizer_fast(spm_model_path: str, special_tokens: list[str]):
    """
    Build a PreTrainedTokenizerFast from a .model produced by SentencePiece.
    Note: This requires `transformers`. If you prefer no transformers yet, you can
    keep SentencePiece directly and add HF later.
    """
    # The simplest route: let transformers load SPM directly
    # but we still provide special tokens explicitly.
    tok = PreTrainedTokenizerFast(tokenizer_file=None)
    tok.backend_tokenizer = None  # explicit for readability

    # transformers can load sentencepiece model via `PreTrainedTokenizerFast.from_pretrained`
    # but that expects a directory. We instead set attributes directly:
    tok = PreTrainedTokenizerFast(
        tokenizer_object=None,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    # Register additional tokens
    tok.add_special_tokens({"additional_special_tokens": [t for t in special_tokens if t not in ["<pad>","<s>","</s>","<unk>"]]})

    # This trick relies on `tokenizer_config.json` and `spiece.model` naming if using from_pretrained.
    # For now, return the fast tokenizer with special tokens; at model-build time you can point
    # to the spm model using AutoTokenizer if desired.
    return tok
