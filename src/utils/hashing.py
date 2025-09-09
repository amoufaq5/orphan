from __future__ import annotations
import hashlib

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def text_hash(s: str) -> str:
    return sha256_hex(s.encode("utf-8"))
