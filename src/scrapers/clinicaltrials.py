# src/scrapers/clinicaltrials.py
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from .base_scraper import BaseScraper, ScrapeResult
from .registry import register
from ..utils.logger import get_logger
from ..utils.schemas import RawDoc, Provenance

log = get_logger("ctgov")

API_V2 = "https://clinicaltrials.gov/api/v2/studies"


def _get(d: Dict[str, Any], path: str, default=None):
    """
    Safe nested getter: _get(obj, "a.b[0].c").
    Supports simple dotted paths and list indices in square brackets.
    """
    cur: Any = d
    for part in path.split("."):
        if not part:
            return default
        if "[" in part and part.endswith("]"):
            # e.g., "items[0]"
            name, idx_str = part[:-1].split("[", 1)
            if name:
                if not isinstance(cur, dict) or name not in cur:
                    return default
                cur = cur.get(name)
            try:
                idx = int(idx_str)
            except ValueError:
                return default
            if not isinstance(cur, list) or not (0 <= idx < len(cur)):
                return default
            cur = cur[idx]
        else:
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur.get(part)
    return cur if cur is not None else default


@register("clinicaltrials")
class ClinicalTrialsGovScraper(BaseScraper):
    """
    ClinicalTrials.gov API v2 scraper.
    - Endpoint: /api/v2/studies
    - Pagination: nextPageToken
    - Params used: pageSize (<= 1000), format=json, countTotal=true
    - Query: dict of v2 query params, default {"query.term": "*"}
    """

    name = "clinicaltrials"

    def __init__(
        self,
        *args,
        query: Optional[Dict[str, str]] = None,   # e.g., {"query.cond": "Diabetes"} or {"query.term": "*"}
        page_size: int = 200,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.query = query or {"query.term": "*"}
        self.page_size = max(1, min(1000, page_size))

    async def run(self) -> ScrapeResult:
        from ..utils.io import shard_writer

        write, close = shard_writer(self.shards_dir, f"{self.name}", self.shard_max_records)

        fetched = 0
        page = 0
        page_token: Optional[str] = None

        async with await self._client() as client:
            while page < self.max_pages and fetched < self.max_docs:
                params: Dict[str, Any] = {
                    **self.query,
                    "pageSize": self.page_size,
                    "format": "json",
                    "countTotal": "true",
                }
                if page_token:
                    params["pageToken"] = page_token

                # Polite rate limit handled in BaseScraper._get_json via TokenBucket
                payload = await self._get_json(client, API_V2, params)

                studies = payload.get("studies") or []
                if not studies:
                    log.info("[clinicaltrials] No studies returned; stopping.")
                    break

                for row in studies:
                    doc = self._row_to_rawdoc(row)
                    write(doc.model_dump(mode="json"))
                    fetched += 1
                    if fetched >= self.max_docs:
                        break

                if fetched >= self.max_docs:
                    break

                page += 1
                page_token = payload.get("nextPageToken")
                if not page_token:
                    break

        close()
        return ScrapeResult(fetched, self.shards_dir)

    def _row_to_rawdoc(self, row: Dict[str, Any]) -> RawDoc:
        # Extract commonly used fields safely
        nctid: Optional[str] = _get(row, "protocolSection.identificationModule.nctId")
        title: Optional[str] = (
            _get(row, "protocolSection.identificationModule.officialTitle")
            or _get(row, "protocolSection.identificationModule.briefTitle")
        )

        overall_status: Optional[str] = _get(row, "protocolSection.statusModule.overallStatus")
        phase: Optional[str] = _get(row, "protocolSection.designModule.phase")
        study_type: Optional[str] = _get(row, "protocolSection.designModule.studyType")

        conditions: List[str] = _get(row, "protocolSection.conditionsModule.conditions", default=[]) or []

        # Interventions: flatten names if present
        interventions = _get(row, "protocolSection.armsInterventionsModule.interventions", default=[]) or []
        intervention_names: List[str] = []
        if isinstance(interventions, list):
            for it in interventions:
                name = (it or {}).get("name")
                if isinstance(name, str) and name.strip():
                    intervention_names.append(name.strip())

        # Outcomes: lists of dicts -> take "measure"
        prim_outs = _get(row, "protocolSection.outcomesModule.primaryOutcomes", default=[]) or []
        prim_meas: List[str] = []
        if isinstance(prim_outs, list):
            for o in prim_outs:
                m = (o or {}).get("measure")
                if isinstance(m, str) and m.strip():
                    prim_meas.append(m.strip())

        sec_outs = _get(row, "protocolSection.outcomesModule.secondaryOutcomes", default=[]) or []
        sec_meas: List[str] = []
        if isinstance(sec_outs, list):
            for o in sec_outs:
                m = (o or {}).get("measure")
                if isinstance(m, str) and m.strip():
                    sec_meas.append(m.strip())

        eligibility_txt: Optional[str] = _get(row, "protocolSection.eligibilityModule.eligibilityCriteria")
        if isinstance(eligibility_txt, str):
            eligibility_txt = eligibility_txt.strip() or None
        else:
            eligibility_txt = None

        # Compose a readable body
        parts: List[str] = []
        if overall_status:
            parts.append(f"Status: {overall_status}")
        if phase:
            parts.append(f"Phase: {phase}")
        if study_type:
            parts.append(f"Type: {study_type}")
        if conditions:
            parts.append("Conditions: " + "; ".join(conditions))
        if intervention_names:
            parts.append("Interventions: " + "; ".join(intervention_names))
        if prim_meas:
            parts.append("Primary Outcomes: " + "; ".join(prim_meas))
        if sec_meas:
            parts.append("Secondary Outcomes: " + "; ".join(sec_meas))
        if eligibility_txt:
            parts.append("Eligibility:\n" + eligibility_txt)

        text = "\n\n".join(parts) if parts else None

        prov = Provenance(
            source="clinicaltrials",
            source_url=f"https://clinicaltrials.gov/study/{nctid}" if nctid else "https://clinicaltrials.gov/",
            license="ClinicalTrials.gov API v2 terms",
            retrieved_at=dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            hash=None,
        )

        return RawDoc(
            id=f"ctgov:{nctid}" if nctid else f"ctgov:auto:{abs(hash(str(row)))%10**12}",
            title=title,
            text=text,
            meta={
                "type": "trial",
                "nctid": nctid,
                "status": overall_status,
                "conditions": conditions,
                "intervention_names": intervention_names,
                "phase": phase,
                "study_type": study_type,
                "eligibility": eligibility_txt,
                "primary_outcomes": prim_meas,
                "secondary_outcomes": sec_meas,
                "lang": "en",
                "raw": row,  # keep full raw for downstream enrichment
            },
            prov=prov,
        )
