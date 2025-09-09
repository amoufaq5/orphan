from __future__ import annotations
import os, json, time, orjson
from typing import Iterable, Dict, Any, Optional

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "rb") as f:
        for line in f:
            if not line.strip(): continue
            yield orjson.loads(line)

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]], indent: int = 0) -> int:
    cnt = 0
    with open(path, "wb") as f:
        for r in rows:
            f.write(orjson.dumps(r, option=(orjson.OPT_APPEND_NEWLINE)))
            cnt += 1
    return cnt

def shard_writer(base_dir: str, base_name: str, max_records: int):
    """
    Returns (write, close) where write(item) appends to current shard.
    """
    ensure_dir(base_dir)
    shard_idx, written = 0, 0
    f = None

    def _open_next():
        nonlocal f, shard_idx, written
        if f: f.close()
        shard_path = os.path.join(base_dir, f"{base_name}.{shard_idx:05d}.jsonl")
        f = open(shard_path, "ab")
        written = 0
        shard_idx += 1
        return shard_path

    _open_next()

    def write(item: Dict[str, Any]):
        nonlocal f, written
        f.write(orjson.dumps(item, option=orjson.OPT_APPEND_NEWLINE))
        written += 1
        if written >= max_records:
            _open_next()

    def close():
        if f: f.close()

    return write, close
