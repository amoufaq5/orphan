from __future__ import annotations

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

from src.utils.logger import get_logger

log = get_logger("scraper.base")


@dataclass
class ScrapeResult:
    total_fetched: int
    shards_path: str
    metadata: Optional[Dict[str, Any]] = None


class TokenBucket:
    """Synchronous token bucket used to throttle outbound requests."""

    def __init__(self, rate_per_sec: float, capacity: Optional[int] = None):
        self.rate_per_sec = max(rate_per_sec, 1e-6)
        self.capacity = capacity if capacity is not None else max(1, int(self.rate_per_sec * 5))
        self.tokens = self.capacity
        self.timestamp = time.monotonic()

    def take(self, tokens: int = 1) -> None:
        tokens = max(1, tokens)
        now = time.monotonic()
        elapsed = now - self.timestamp
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_sec)
        if self.tokens < tokens:
            deficit = tokens - self.tokens
            sleep_for = deficit / self.rate_per_sec
            time.sleep(max(sleep_for, 0.0))
            now = time.monotonic()
            elapsed = now - self.timestamp
            if elapsed > 0:
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_sec)
        self.tokens = max(self.tokens - tokens, 0)
        self.timestamp = time.monotonic()


class BaseScraper:
    """Common functionality shared across async scraper implementations."""

    name: str = "base"

    def __init__(
        self,
        *,
        out_dir: str = "data/scraped/raw",
        max_docs: int = 500,
        max_pages: int = 50,
        shard_max_records: int = 500,
        rate_limit_per_sec: float = 1.0,
        request_headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shards_dir = str(self.out_dir)
        self.max_docs = max_docs
        self.max_pages = max_pages
        self.shard_max_records = shard_max_records
        self.timeout = timeout
        self.request_headers = request_headers or {}
        self.bucket = TokenBucket(rate_limit_per_sec)

    @property
    def base_headers(self) -> Dict[str, str]:
        headers = {
            "User-Agent": "orphan-medical-ai-scraper/1.0 (+https://github.com/amoufaq5/orphan)",
            "Accept": "application/json, text/plain, */*",
        }
        headers.update(self.request_headers)
        return headers

    @asynccontextmanager
    async def _client(self, extra_headers: Optional[Dict[str, str]] = None) -> httpx.AsyncClient:
        headers = self.base_headers.copy()
        if extra_headers:
            headers.update(extra_headers)
        async with httpx.AsyncClient(headers=headers, timeout=self.timeout) as client:
            yield client

    async def _get_json(self, client: httpx.AsyncClient, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.bucket.take(1)
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    async def _get_text(self, client: httpx.AsyncClient, url: str, params: Optional[Dict[str, Any]] = None) -> str:
        self.bucket.take(1)
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.text

    async def _get_bytes(self, client: httpx.AsyncClient, url: str, params: Optional[Dict[str, Any]] = None) -> bytes:
        self.bucket.take(1)
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.content

    async def run(self) -> ScrapeResult:  # pragma: no cover - to be implemented by subclasses
        raise NotImplementedError
