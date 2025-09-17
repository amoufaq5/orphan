import pathlib, gzip, json
from tokenizers import ByteLevelBPETokenizer

# Simple logger substitute (avoid pulling project logger if not initialized)
def log(msg): print(msg)

CORPUS_PATHS = [
    pathlib.Path("data/corpus/corpus.jsonl.gz"),
    pathlib.Path("data/corpus/corpus.jsonl"),
]
OUT_DIR = pathlib.Path("out/tokenizer")

def iter_texts():
    for p in CORPUS_PATHS:
        if p.exists():
            log(f"[train_tokenizer] Reading corpus from {p}")
            opener = gzip.open if p.suffix.endswith(".gz") else open
            with opener(p, "rt", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        j = json.loads(line)
                        txt = j.get("text")
                        if txt and txt.strip():
                            yield txt
                    except Exception:
                        continue
            return
    raise FileNotFoundError(f"No corpus found at {CORPUS_PATHS}")

def main():
    tmp_file = pathlib.Path("data/corpus/_tmp_texts.txt")
    tmp_file.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with tmp_file.open("w", encoding="utf-8") as out:
        for n, txt in enumerate(iter_texts(), start=1):
            out.write(txt.replace("\n", " ") + "\n")
            if n % 100000 == 0:
                log(f"[train_tokenizer] Wrote {n} lines...")
    if n == 0:
        raise RuntimeError("No lines found in corpus to train tokenizer.")

    log(f"[train_tokenizer] Starting tokenizer training on {tmp_file} (lines={n})")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(tmp_file)],
        vocab_size=32000,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(str(OUT_DIR))
    log(f"[train_tokenizer] Tokenizer saved to {OUT_DIR}")

    try:
        tmp_file.unlink()
    except Exception:
        pass

if __name__ == "__main__":
    main()
