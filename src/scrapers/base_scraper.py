from __future__ import annotations
import asyncio
import time
from typing import Any, Dict, Iterable, Tuple, Optional

import httpx
from httpx import HTTPStatusError
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from ..utils.logger import get_logger
from ..utils.rate_limit import TokenBucket
from ..utils.io import shard_writer
from ..utils.schemas import RawDoc
from ..utils.hashing import text_hash

log = get_logger("scraper")


class ScrapeResult:
    def __init__(self, total_fetched: int, shards_path: str):
        self.total_fetched = total_fetched
        self.shards_path = shards_path


class BaseScraper:
    """
    Production-minded base scraper:
      - Session reuse (httpx.AsyncClient)
      - Token-bucket rate limiting
      - Tenacity retries with jitter backoff
      - Graceful handling of 4xx/5xx (no process crash)
      - Sharded JSONL output
      - Soft page/document caps
      - Optional extra headers (e.g., 'X-Api-Key')
      - Hooks to customize behavior in subclasses

    Subclasses must override:
      - name (str)
      - build_requests(self) -> Iterable[Tuple[url, params]]
      - parse(self, payload: Dict[str, Any]) -> Iterable[RawDoc]
    """

    name: str = "base"

    def __init__(
        self,
        user_agent: str,
        rps: float = 3.0,
        burst: int = 3,
        timeout: int = 30,
        total_retries: int = 5,
        max_pages: int = 50,
        max_docs: int = 200_000,
        shards_dir: str = "data/shards",
        shard_max_records: int = 5000,
        start_page: int = 0,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        self.ua = user_agent
        self.bucket = TokenBucket(rps, burst)
        self.timeout = timeout
        self.total_retries = total_retries
        self.max_pages = max(0, int(max_pages))
        self.max_docs = max_docs
        self.shards_dir = shards_dir
        self.shard_max_records = shard_max_records
        self.start_page = max(0, int(start_page))
        self.extra_headers = extra_headers or {}

    # ------------ Hooks (subclasses may override) ------------

    def build_requests(self) -> Iterable[Tuple[str, Dict[str, Any]]]:
        """Yield (url, params) for each page/request."""
        raise NotImplementedError

    def parse(self, payload: Dict[str, Any]) -> Iterable[RawDoc]:
        """Yield RawDoc for a decoded JSON payload."""
        raise NotImplementedError

    def on_before_request(self, url: str, params: Dict[str, Any]) -> None:
        """Hook called before each HTTP request."""
        return

    def on_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Hook to modify/normalize payload before parse()."""
        return payload

    def on_http_error(self, exc: HTTPStatusError, url: str, params: Dict[str, Any]) -> bool:
        """
        Return True if caller should CONTINUE (skip this page) after logging,
        False if it should STOP the scraper early.
        Default: continue on client/server errors (skip page).
        """
        status = exc.response.status_code if exc.response is not None else "?"
        log.error(f"[{self.name}] HTTP {status} for {url} params={params}")
        # If 401/403 and subclass cannot recover, continuing may loop; default to continue once (skip this page).
        return True

    # ------------ HTTP client & retry core ------------

    async def _client(self, extra_headers: dict | None = None) -> httpx.AsyncClient:
        headers = {"User-Agent": self.ua}
        headers.update(self.extra_headers)
        if extra_headers:
            headers.update(extra_headers)
        return httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(self.timeout, connect=max(1.0, self.timeout / 2.0)),
        )

    @retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(initial=0.5, max=8))
    async def _get_json(self, client: httpx.AsyncClient, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self.bucket.take(1)
        r = await client.get(url, params=params)
        # If rate-limited with Retry-After, be polite
        if r.status_code == 429:
            ra = r.headers.get("Retry-After")
            try:
                sleep_s = float(ra)
            except (TypeError, ValueError):
                sleep_s = 2.0
            log.warning(f"[{self.name}] 429 Too Many Requests → sleeping {sleep_s:.1f}s")
            await asyncio.sleep(sleep_s)
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        if "application/json" not in ctype and not r.text.strip().startswith("{"):
            # Some APIs mislabel; try best-effort parse
            try:
                return r.json()
            except Exception:
                log.error(f"[{self.name}] Non-JSON response for {url} (ctype={ctype})")
                return {}
        return r.json()

    # ------------ Main run loop ------------

    async def run(self) -> ScrapeResult:
        write, close = shard_writer(self.shards_dir, f"{self.name}", self.shard_max_records)
        total = 0
        page_idx = -1

        async with await self._client() as client:
            for i, (url, params) in enumerate(self.build_requests()):
                page_idx = i
                if i < self.start_page:
                    continue
                if i >= self.max_pages:
                    log.warning(f"[{self.name}] Reached max_pages={self.max_pages}, stopping.")
                    break
                if self.max_docs is not None and total >= self.max_docs:
                    log.warning(f"[{self.name}] Reached max_docs={self.max_docs}, stopping.")
                    break

                try:
                    self.on_before_request(url, params)
                    payload = await self._get_json(client, url, params)
                except HTTPStatusError as e:
                    # Let subclass decide whether to continue or stop
                    should_continue = self.on_http_error(e, url, params)
                    if should_continue:
                        continue
                    else:
                        break
                except Exception as e:
                    log.error(f"[{self.name}] Request error for {url}: {e}")
                    # Skip just this page
                    continue

                payload = self.on_payload(payload) or payload

                try:
                    for raw in self.parse(payload):
                        # provenance hash
                        if raw.text:
                            raw.prov.hash = text_hash(raw.text)
                        write(raw.model_dump(mode="json"))
                        total += 1
                        if self.max_docs is not None and total >= self.max_docs:
                            log.warning(f"[{self.name}] Reached max_docs={self.max_docs}, stopping.")
                            close()
                            return ScrapeResult(total, self.shards_dir)
                except Exception as e:
                    log.error(f"[{self.name}] Parse error on page {i}: {e}")
                    # Skip parse errors, proceed to next page
                    continue

        close()
        return ScrapeResult(total, self.shards_dir)
