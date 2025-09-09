from __future__ import annotations
import time, asyncio, httpx, math
from typing import Any, Dict, Iterable, List, Optional, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from ..utils.logger import get_logger
from ..utils.rate_limit import TokenBucket
from ..utils.io import shard_writer
from ..utils.schemas import RawDoc, Provenance
from ..utils.hashing import text_hash

log = get_logger("scraper")

class ScrapeResult:
    def __init__(self, total_fetched: int, shards_path: str):
        self.total_fetched = total_fetched
        self.shards_path = shards_path

class BaseScraper:
    """
    Base class: handles session mgmt, retries, rate limiting, paging, sharding.
    Override:
      - name (str)
      - build_requests(self) -> Iterable[Tuple[url, params]]
      - parse(self, response_json) -> Iterable[RawDoc]
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
    ):
        self.ua = user_agent
        self.bucket = TokenBucket(rps, burst)
        self.timeout = timeout
        self.total_retries = total_retries
        self.max_pages = max_pages
        self.max_docs = max_docs
        self.shards_dir = shards_dir
        self.shard_max_records = shard_max_records

    async def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={"User-Agent": self.ua},
            timeout=httpx.Timeout(self.timeout, connect=self.timeout/2),
        )

    @retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(initial=0.5, max=8))
    async def _get_json(self, client: httpx.AsyncClient, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self.bucket.take(1)
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()

    def build_requests(self) -> Iterable[Tuple[str, Dict[str, Any]]]:
        """Override with (url, params) pages."""
        raise NotImplementedError

    def parse(self, payload: Dict[str, Any]) -> Iterable[RawDoc]:
        """Override to yield RawDoc instances from response JSON."""
        raise NotImplementedError

    async def run(self) -> ScrapeResult:
        write, close = shard_writer(self.shards_dir, f"{self.name}", self.shard_max_records)
        total = 0
        async with await self._client() as client:
            for i, (url, params) in enumerate(self.build_requests()):
                if i >= self.max_pages:
                    log.warning(f"[{self.name}] Reached max_pages={self.max_pages}, stopping.")
                    break
                payload = await self._get_json(client, url, params)
                for raw in self.parse(payload):
                    # provenance hash (content-based)
                    if raw.text:
                        raw.prov.hash = text_hash(raw.text)
                    write(raw.model_dump(mode="json"))
                    total += 1
                    if total >= self.max_docs:
                        log.warning(f"[{self.name}] Reached max_docs={self.max_docs}, stopping.")
                        close()
                        return ScrapeResult(total, self.shards_dir)
        close()
        return ScrapeResult(total, self.shards_dir)
