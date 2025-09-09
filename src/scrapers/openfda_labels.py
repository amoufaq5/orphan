from __future__ import annotations
import os, math, datetime as dt
from typing import Any, Dict, Iterable, List, Tuple, Optional
from httpx import HTTPStatusError
from .base_scraper import BaseScraper
from .registry import register
from ..utils.logger import get_logger
from ..utils.schemas import RawDoc, Provenance

log = get_logger("openfda")

API = "https://api.fda.gov/drug/label.json"

@register("openfda_labels")
class OpenFDALabelsScraper(BaseScraper):
    name = "openfda_labels"

    def __init__(self, *args, limit: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self.limit = max(1, min(100, limit))
        self.api_key = os.getenv("OPENFDA_API_KEY", "").strip()

    def build_requests(self) -> Iterable[Tuple[str, Dict[str, Any]]]:
        # use existence query (broad and accepted)
        skip = 0
        pages = 0
        while pages < self.max_pages and skip < self.max_docs:
            params: Dict[str, Any] = {
                "search": "_exists_:effective_time",
                "limit": self.limit,
                "skip": skip,
            }
            if self.api_key:
                params["api_key"] = self.api_key
            yield API, params
            pages += 1
            skip += self.limit

    async def run(self):
        write, close = None, None
        from ..utils.io import shard_writer
        write, close = shard_writer(self.shards_dir, f"{self.name}", self.shard_max_records)
        total = 0
        async with await self._client(extra_headers={"X-Api-Key": self.api_key} if self.api_key else None) as client:
            for i, (url, params) in enumerate(self.build_requests()):
                if i >= self.max_pages:
                    log.warning(f"[{self.name}] Reached max_pages={self.max_pages}, stopping.")
                    break
                try:
                    payload = await self._get_json(client, url, params)
                except HTTPStatusError as e:
                    if e.response.status_code == 403:
                        log.warning("[openfda] 403 Forbidden — retrying WITHOUT api key and safer search…")
                        # Retry once without key
                        params.pop("api_key", None)
                        try:
                            payload = await self._get_json(client, url, params)
                        except Exception as e2:
                            log.error(f"[openfda] still failing after removing key: {e2}")
                            break
                    else:
                        log.error(f"[openfda] HTTP error: {e}")
                        break

                for raw in self.parse(payload):
                    write(raw.model_dump(mode="json"))
                    total += 1
                    if total >= self.max_docs:
                        log.warning(f"[{self.name}] Reached max_docs={self.max_docs}, stopping.")
                        close()
                        return type("ScrapeResult", (), {"total_fetched": total, "shards_path": self.shards_dir})()
        close()
        return type("ScrapeResult", (), {"total_fetched": total, "shards_path": self.shards_dir})()

    def parse(self, payload: Dict[str, Any]) -> Iterable[RawDoc]:
        results = payload.get("results") or []
        for r in results:
            rid = r.get("id") or r.get("set_id") or r.get("spl_id") or None
            openfda = r.get("openfda") or {}
            brand = (openfda.get("brand_name") or [None])[0]
            substant = (openfda.get("substance_name") or [None])[0]
            title = " / ".join([x for x in [brand, substant] if x]) or None

            sections_order = [
                "package_label_principal_display_panel",
                "indications_and_usage",
                "dosage_and_administration",
                "contraindications",
                "warnings",
                "warnings_and_cautions",
                "precautions",
                "adverse_reactions",
                "drug_interactions",
                "use_in_specific_populations",
                "clinical_pharmacology",
                "clinical_studies",
                "patient_counseling_information",
                "information_for_patients",
                "storage_and_handling",
            ]
            text_parts: List[str] = []
            for key in sections_order:
                val = r.get(key)
                if isinstance(val, list):
                    text_parts.extend(v for v in val if isinstance(v, str) and v.strip())
                elif isinstance(val, str):
                    text_parts.append(val)
            text = "\n\n".join([t.strip() for t in text_parts if t])

            prov = Provenance(
                source="openfda_labels",
                source_url=f"https://labels.fda.gov/?id={rid}" if rid else "https://labels.fda.gov/",
                license="openFDA terms of service",
                retrieved_at=dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                hash=None,
            )
            yield RawDoc(
                id=f"openfda:{rid}" if rid else f"openfda:auto:{abs(hash(str(r)))%10**12}",
                title=title,
                text=text or None,
                meta={"type": "drug_label", "raw": r, "lang": "en", "source": "openfda"},
                prov=prov,
            )
