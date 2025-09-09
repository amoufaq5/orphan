from __future__ import annotations
import os, glob, unicodedata, orjson
from typing import Iterable, Dict, Any, Set, List
from ...utils.logger import get_logger
from ...utils.io import read_jsonl, ensure_dir
from ...utils.config import load_yaml

log = get_logger("tok-corpus")

def _norm(text: str, form: str = "NFKC") -> str:
    try:
        return unicodedata.normalize(form, text)
    except Exception:
        return text

def _lang_hint(s: str) -> str:
    # extremely light heuristic: presence of Arabic unicode block
    for ch in s:
        if '\u0600' <= ch <= '\u06FF' or '\u0750' <= ch <= '\u077F' or '\u08A0' <= ch <= '\u08FF':
            return "ar"
    return "en"

def iter_canonical_text(files: List[str], include_types: List[str] | None, dedupe: bool, max_docs: int, norm_form: str):
    seen: Set[int] = set()
    count = 0
    for path in files:
        for row in read_jsonl(path):
            t = str(row.get("type", "")).lower()
            if include_types and t not in include_types:
                continue
            sections = row.get("sections") or {}
            # concatenate known section fields; default to "body"
            pieces = []
            if isinstance(sections, dict):
                # prefer structured sections when present (labels)
                # else fallback to body
                if any(k for k in sections.keys()):
                    for k, v in sections.items():
                        if isinstance(v, str) and v.strip():
                            pieces.append(v.strip())
                else:
                    body = sections.get("body")
                    if isinstance(body, str) and body.strip():
                        pieces.append(body.strip())
            text = "\n\n".join(pieces).strip()
            if not text:
                continue
            text_n = _norm(text, norm_form)
            if dedupe:
                h = hash(text_n)
                if h in seen:
                    continue
                seen.add(h)
            yield text_n
            count += 1
            if count >= max_docs:
                return

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/train_text.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    tcfg = cfg["train"]["tokenizer"]
    cset = cfg["train"]["tokenizer_corpus"]
    files = []
    for g in cset["input_globs"]:
        files.extend(glob.glob(g))
    files = sorted(files)
    if not files:
        log.warning("[tokenizer] No input files matched.")
        return

    out_txt = cset["output_txt"]
    ensure_dir(os.path.dirname(out_txt))

    include_types = [s.lower() for s in (cset.get("include_types") or [])]
    dedupe = bool(cset.get("dedupe_by_hash", True))
    max_docs = int(cset.get("max_docs", 100000))
    norm_form = tcfg.get("normalization", "NFKC")

    ar_hint = cset.get("arabic_hint_token", "<arabic>")
    en_hint = cset.get("english_hint_token", "<english>")

    rows = 0
    with open(out_txt, "w", encoding="utf-8", newline="\n") as f:
        for txt in iter_canonical_text(files, include_types, dedupe, max_docs, norm_form):
            hint = ar_hint if _lang_hint(txt) == "ar" else en_hint
            # add a language hint token at line start to help tokenizer learn both scripts
            f.write(hint + " " + txt.replace("\r", " ") + "\n")
            rows += 1

    log.info(f"[tokenizer] Wrote corpus: {out_txt} (lines={rows})")

if __name__ == "__main__":
    main()
