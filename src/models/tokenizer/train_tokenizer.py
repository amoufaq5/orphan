"""
train_tokenizer.py
------------------
Train a Byte-Pair Encoding (BPE) tokenizer on your unified corpus.

- Reads from data/corpus/corpus.jsonl.gz (or .jsonl)
- Saves tokenizer to out/tokenizer/
"""

import pathlib, gzip, json
from tokenizers import ByteLevelBPETokenizer
from src.utils.logger import get_logger

log = get_logger("train_tokenizer")

CORPUS_PATHS = [
    pathlib.Path("data/corpus/corpus.jsonl.gz"),
    pathlib.Path("data/corpus/corpus.jsonl")
]
OUT_DIR = pathlib.Path("out/tokenizer")

def iter_texts():
    for p in CORPUS_PATHS:
        if p.exists():
            log.info(f"Reading corpus from {p}")
            opener = gzip.open if p.suffix.endswith(".gz") else open
            with opener(p, "rt", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): 
                        continue
                    try:
                        j = json.loads(line)
                        txt = j.get("text")
                        if txt and len(txt.strip()) > 0:
                            yield txt
                    except Exception as e:
                        continue
            return
    raise FileNotFoundError(f"No corpus found at {CORPUS_PATHS}")

def main():
    # Collect texts to temp file (tokenizers API requires file input)
    tmp_file = pathlib.Path("data/corpus/_tmp_texts.txt")
    tmp_file.parent.mkdir(parents=True, exist_ok=True)

    with tmp_file.open("w", encoding="utf-8") as out:
        for i, txt in enumerate(iter_texts()):
            out.write(txt.replace("\n", " ") + "\n")
            if (i+1) % 100000 == 0:
                log.info(f"Wrote {i+1} lines to temporary training file…")

    log.info(f"Starting tokenizer training on {tmp_file}")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(tmp_file)],
        vocab_size=32000,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(str(OUT_DIR))
    log.info(f"Tokenizer saved to {OUT_DIR}")

    # Cleanup
    tmp_file.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
