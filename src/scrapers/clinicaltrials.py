from __future__ import annotations
import datetime as dt
from typing import Any, Dict, Iterable, List, Tuple, Optional
from .base_scraper import BaseScraper, ScrapeResult
from .registry import register
from ..utils.logger import get_logger
from ..utils.schemas import RawDoc, Provenance

log = get_logger("ctgov")

API_V2 = "https://clinicaltrials.gov/api/v2/studies"

# v2 uses dotted field paths. We select a tight set analogous to v1 Study Fields.
FIELDS = [
    "protocolSection.identificationModule.nctId",
    "protocolSection.identificationModule.briefTitle",
    "protocolSection.identificationModule.officialTitle",
    "protocolSection.statusModule.overallStatus",
    "protocolSection.conditionsModule.conditions",
    "protocolSection.designModule.phase",
    "protocolSection.designModule.studyType",
    "protocolSection.eligibilityModule.eligibilityCriteria",
    "protocolSection.armsInterventionsModule.interventions",
    "protocolSection.outcomesModule.primaryOutcomes",
    "protocolSection.outcomesModule.secondaryOutcomes",
    "protocolSection.contactsLocationsModule.locations",
    "protocolSection.sponsorCollaboratorsModule.leadSponsor.name",
    "protocolSection.sponsorCollaboratorsModule.collaborators.name",
    "protocolSection.statusModule.studyFirstPostDateStruct.date",
    "protocolSection.statusModule.lastUpdateSubmitDateStruct.date",
    "resultsSection.publicationsModule.publications",
]

@register("clinicaltrials")
class ClinicalTrialsGovScraper(BaseScraper):
    """
    ClinicalTrials.gov API v2:
      - Endpoint: /api/v2/studies
      - Pagination: pageToken
      - Params: pageSize (<= 1000), format=json, countTotal=true
      - Query: e.g., query.term="*" or query.cond="Diabetes"
    """

    name = "clinicaltrials"

    def __init__(
        self,
        *args,
        query: Optional[Dict[str, str]] = None,   # e.g., {"query.term":"*"} or {"query.cond":"Diabetes"}
        page_size: int = 200,
        **kwargs
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
                    "fields": ",".join(FIELDS),
                }
                if page_token:
                    params["pageToken"] = page_token

                self.bucket.take(1)
                r = await client.get(API_V2, params=params)
                if r.status_code == 404:
                    # Helpful message if site is in transition or fields invalid
                    msg = f"[clinicaltrials] 404 from v2 endpoint. Params={params}"
                    log.error(msg)
                    r.raise_for_status()
                r.raise_for_status()
                payload = r.json()

                studies = payload.get("studies") or []
                for row in studies:
                    doc = self._row_to_rawdoc(row)
                    write(doc.model_dump(mode="json"))
                    fetched += 1
                    if fetched >= self.max_docs:
                        break

                page += 1
                page_token = payload.get("nextPageToken")
                if not page_token or not studies:
                    break

        close()
        return ScrapeResult(fetched, self.shards_dir)

    def _row_to_rawdoc(self, row: Dict[str, Any]) -> RawDoc:
        ps = row.get("protocolSection", {})
        ident = ps.get("identificationModule", {})
        status = ps.get("statusModule", {})
        cond = ps.get("conditionsModule", {})
        design = ps.get("designModule", {})
        elig  = ps.get("eligibilityModule", {})
        arms  = ps.get("armsInterventionsModule", {})
        outs  = ps.get("outcomesModule", {})
        locs  = ps.get("contactsLocationsModule", {})
        spons = ps.get("sponsorCollaboratorsModule", {})

        nctid  = ident.get("nctId")
        title  = ident.get("officialTitle") or ident.get("briefTitle")
        overall_status = status.get("overallStatus")

        # Flatten interventions (names)
        inter_names: List[str] = []
        for it in arms.get("interventions") or []:
            name = it.get("name")
            if isinstance(name, str) and name.strip():
                inter_names.append(name.strip())

        # Conditions (list of strings)
        conditions = cond.get("conditions") or []

        # Outcomes
        prim = outs.get("primaryOutcomes") or []
        prim_meas = [o.get("measure") for o in pr]()_
