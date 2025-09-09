from __future__ import annotations
import re
from typing import Dict

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"\+?\d[\d\s\-]{7,}\d")
IDNUM = re.compile(r"\b[0-9]{8,}\b")

def scrub(text: str) -> str:
    text = EMAIL.sub("[EMAIL]", text)
    text = PHONE.sub("[PHONE]", text)
    text = IDNUM.sub("[ID]", text)
    return text

def scrub_record(rec: Dict) -> Dict:
    out = {}
    for k, v in rec.items():
        if isinstance(v, str):
            out[k] = scrub(v)
        else:
            out[k] = v
    return out
