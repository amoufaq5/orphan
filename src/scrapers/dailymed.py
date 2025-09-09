from __future__ import annotations
import asyncio, math, datetime as dt
from typing import Any, Dict, Iterable, List, Tuple
from dateutil.parser import isoparse
from .base_scraper import BaseScraper, ScrapeResult
from .registry import register
from ..utils.logger import get_logger
from ..utils.schemas import RawDoc, Provenance

log = get_logger("dailymed")

BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

@register("dailymed_spls")
class DailyMedSPLScraper(BaseScraper):
    """
    Two-phase scraper:
      1) page through /spls to collect SETIDs (metadata)
      2) fetch details /spls/{SETID}.json for full sections (when available)
    Writes RawDoc shards with full provenance.
    """

    name = "dailymed_spls"

    def __init__(self, *args, pagesize: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self.pagesize = max(1, min(250, pagesize))  # DailyMed typically allows up to 250

    # We override run() because we need an index phase then detail phase
    async def run(self) -> ScrapeResult:
        write, close = None, None
        total = 0

        async with await self._client() as client:
            # Phase 1: enumerate SPL SETIDs
            setids: List[Dict[str, Any]] = []
            page = 1
            while page <= self.max_pages:
                params = {"pagesize": self.pagesize, "page": page}
                self.bucket.take(1)
                r = await client.get(f"{BASE}/spls.json", params=params)
                r.raise_for_status()
                payload = r.json()
                data = payload.get("data") or []
                if not data:
                    log.info(f"[dailymed] empty page {page}, stopping.")
                    break

                for item in data:
                    # Minimal row to move into detail fetch
                    setid = item.get("setid") or item.get("setId") or item.get("set_id")
                    if not setid:
                        continue
                    setids.append(item)
                    if len(setids) >= self.max_docs:
                        log.warning("[dailymed] hit max_docs during index phase.")
                        break

                if len(setids) >= self.max_docs:
                    break

                total_count = (payload.get("metadata") or {}).get("total", None)
                if total_count:
                    max_pages_est = math.ceil(total_count / self.pagesize)
                    if page >= max_pages_est:
                        break
                page += 1

            if not setids:
                from ..utils.io import shard_writer
                write, close = shard_writer(self.shards_dir, f"{self.name}", self.shard_max_records)
                close()
                return ScrapeResult(0, self.shards_dir)

            # Phase 2: detail fetch for each SETID
            from ..utils.io import shard_writer
            write, close = shard_writer(self.shards_dir, f"{self.name}", self.shard_max_records)
            fetched = 0
            for idx, meta in enumerate(setids):
                if fetched >= self.max_docs:
                    log.warning("[dailymed] max_docs reached in detail phase.")
                    break
                setid = meta.get("setid") or meta.get("setId") or meta.get("set_id")
                if not setid:
                    continue
                # detail endpoint
                detail_url = f"{BASE}/spls/{setid}.json"
                payload = await self._get_json(client, detail_url, params={})
                for doc in self._parse_detail(payload, meta):
                    write(doc.model_dump(mode="json"))
                    fetched += 1
                    if fetched >= self.max_docs:
                        break

            close()
            return ScrapeResult(fetched, self.shards_dir)

    def _parse_detail(self, payload: Dict[str, Any], index_meta: Dict[str, Any]) -> Iterable[RawDoc]:
        """
        Parse detail JSON. DailyMed returns a structure containing sections.
        We conservatively join textual parts while keeping full payload in meta.
        """
        data = payload.get("data") or []
        if not isinstance(data, list):
            data = [data]

        for item in data:
            setid = item.get("setid") or item.get("setId") or index_meta.get("setid")
            title = (item.get("title") or index_meta.get("title") or "").strip() or None
            # Try to gather text bodies from known keys if present
            text_parts: List[str] = []
            # Some responses contain a 'sections' array
            sections = item.get("sections") or item.get("splSections") or []
            if isinstance(sections, list):
                for s in sections:
                    t = s.get("text") or s.get("content") or ""
                    if t:
                        text_parts.append(t)
            # Fallback: if no sections, try 'text' or other fields
            if not text_parts:
                for k in ("text", "content", "body"):
                    if k in item and isinstance(item[k], str):
                        text_parts.append(item[k])

            text = "\n\n".join([p.strip() for p in text_parts if p and isinstance(p, str)]) or None

            # Provenance
            url = item.get("spl_url") or f"{BASE.replace('/services/v2','')}/search.cfm?setid={setid}"
            eff = item.get("effective_time") or index_meta.get("effective_time")
            try:
                eff_iso = isoparse(str(eff)).isoformat() if eff else None
            except Exception:
                eff_iso = None

            prov = Provenance(
                source="dailymed",
                source_url=url,
                license="U.S. National Library of Medicine (DailyMed) – see site terms",
                retrieved_at=dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                hash=None,
            )

            yield RawDoc(
                id=f"dailymed:{setid}",
                title=title,
                text=text,
                meta={"type": "drug_label", "raw": item, "index_meta": index_meta, "lang": "en"},
                prov=prov,
            )
