from __future__ import annotations
import os, math, datetime as dt
from typing import Any, Dict, Iterable, List, Tuple
from .base_scraper import BaseScraper
from .registry import register
from ..utils.logger import get_logger
from ..utils.schemas import RawDoc, Provenance

log = get_logger("openfda")

API = "https://api.fda.gov/drug/label.json"

@register("openfda_labels")
class OpenFDALabelsScraper(BaseScraper):
    """
    Paginates with skip/limit. Max limit per call = 100 (openFDA).
    Respects API key if provided via env OPENFDA_API_KEY.
    """

    name = "openfda_labels"

    def __init__(self, *args, limit: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self.limit = max(1, min(100, limit))
        self.api_key = os.getenv("OPENFDA_API_KEY", "")

    def build_requests(self) -> Iterable[Tuple[str, Dict[str, Any]]]:
        # Use a broad search to fetch everything; you can refine later by updated dates etc.
        # openFDA supports meta.results.total; we’ll iterate up to caps/max_pages.
        # We don't know total without first call; so we build a generator that yields progressively.
        total = None
        skip = 0
        pages = 0
        while (pages < self.max_pages) and (self.max_docs is None or skip < self.max_docs):
            params: Dict[str, Any] = {
                "search": "effective_time:[* TO *]",
                "limit": self.limit,
                "skip": skip,
            }
            if self.api_key:
                params["api_key"] = self.api_key
            yield API, params
            pages += 1
            skip += self.limit

    def parse(self, payload: Dict[str, Any]) -> Iterable[RawDoc]:
        results = payload.get("results") or []
        for r in results:
            rid = r.get("id") or r.get("set_id") or r.get("spl_id") or None
            title = None
            # try to compose a human title
            openfda = r.get("openfda") or {}
            brand = (openfda.get("brand_name") or [None])[0]
            substant = (openfda.get("substance_name") or [None])[0]
            manufacturer = (openfda.get("manufacturer_name") or [None])[0]
            if brand or substant:
                title = " / ".join([x for x in [brand, substant] if x])

            # build a readable text body by joining common sections if present
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
                    text_parts.extend(v for v in val if isinstance(v, str))
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
                meta={"type": "drug_label", "raw": r, "lang": "en"},
                prov=prov,
            )
