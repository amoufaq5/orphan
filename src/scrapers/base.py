from __future__ import annotations
import time, json, math, itertools
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from pathlib import Path

UA = "OrphBot/1.0 (+research)"

@dataclass
class ShardWriter:
path: Path
max_rows: int = 10000
_fh: Optional[open] = None
_rows: int = 0
_idx: int = 0

def _open_new(self):
self._idx += 1
self._rows = 0
self.path.parent.mkdir(parents=True, exist_ok=True)
if self._fh: self._fh.close()
fn = self.path.with_name(f"{self.path.stem}-{self._idx:05d}.jsonl")
self._fh = open(fn,'w',encoding='utf-8')

def write(self, obj):
if not self._fh or self._rows >= self.max_rows:
self._open_new()
self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
self._rows += 1

def close(self):
if self._fh: self._fh.close()


def session_with_retries(total=5, backoff=0.5) -> requests.Session:
s = requests.Session()
s.headers.update({"User-Agent": UA})
retry = Retry(
total=total, read=total, connect=total, status=total,
status_forcelist=(429,500,502,503,504),
backoff_factor=backoff,
allowed_methods=("GET","POST"),
)
s.mount('https://', HTTPAdapter(max_retries=retry))
s.mount('http://', HTTPAdapter(max_retries=retry))
return s


def rate_limited(iterable: Iterable, per_sec: float) -> Iterator:
interval = 1.0 / max(per_sec, 1e-6)
last = 0.0
for item in iterable:
now = time.time()
if last:
sleep = interval - (now - last)
if sleep > 0: time.sleep(sleep)
last = time.time()
yield item
