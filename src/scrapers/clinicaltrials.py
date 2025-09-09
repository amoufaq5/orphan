from __future__ import annotations
import math, datetime as dt
from typing import Any, Dict, Iterable, List, Tuple
from .base_scraper import BaseScraper
from .registry import register
from ..utils.logger import get_logger
from ..utils.schemas import RawDoc, Provenance

log = get_logger("ctgov")

API = "https://clinicaltrials.gov/api/query/study_fields"

# Fields chosen for downstream usefulness; can extend later.
FIELDS = [
    "NCTId", "BriefTitle", "OfficialTitle", "OverallStatus",
    "Condition", "InterventionType", "InterventionName",
    "EligibilityCriteria", "Phase", "StudyType",
    "StudyFirstPostDate", "LastUpdateSubmitDate",
    "LocationCountry", "LeadSponsorName", "CollaboratorName",
    "PrimaryOutcomeMeasure", "PrimaryOutcomeTimeFrame",
    "SecondaryOutcomeMeasure", "SecondaryOutcomeTimeFrame",
    "ResultsFirstPostDate",
]

@register("clinicaltrials")
class ClinicalTrialsGovScraper(BaseScraper):
    """
    Uses ClinicalTrials.gov Study Fields API with rank paging.
    - Query defaults to broad expression (all studies).
    - Paginates using min_rnk/max_rnk (1-based, inclusive).
    """

    name = "clinicaltrials"

    def __init__(self, *args, expr: str = "AREA[OverallStatus] *", page_size: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self.expr = expr
        self.page_size = max(1, min(1000, page_size))  # API permits up to 1000

    def build_requests(self) -> Iterable[Tuple[str, Dict[str, Any]]]:
        # We don’t know total until first call, so we just yield pages up to max_pages/max_docs.
        # The runner calls _get_json which ratelimits + retries.
        for p in range(self.max_pages):
            start = p * self.page_size + 1
            end   = start + self.page_size - 1
            params = {
                "expr": self.expr,
                "fields": ",".join(FIELDS),
                "min_rnk": start,
                "max_rnk": end,
                "fmt": "json",
            }
            yield API, params

    def parse(self, payload: Dict[str, Any]) -> Iterable[RawDoc]:
        data = (((payload.get("StudyFieldsResponse") or {}).get("StudyFields")) or [])
        for row in data:
            # Row values are arrays per field; unwrap some thoughtfully
            def one(field: str) -> str | None:
                v = row.get(field) or []
                return v[0].strip() if v and isinstance(v[0], str) and v[0].strip() else None

            def many(field: str) -> List[str]:
                v = row.get(field) or []
                return [x.strip() for x in v if isinstance(x, str) and x.strip()]

            nctid  = one("NCTId")
            title  = one("OfficialTitle") or one("BriefTitle")
            status = one("OverallStatus")
            conds  = many("Condition")
            inter_types = many("InterventionType")
            inter_names = many("InterventionName")
            elig   = one("EligibilityCriteria")
            phase  = one("Phase")
            stype  = one("StudyType")
            p_out  = many("PrimaryOutcomeMeasure")
            p_tf   = many("PrimaryOutcomeTimeFrame")
            s_out  = many("SecondaryOutcomeMeasure")
            s_tf   = many("SecondaryOutcomeTimeFrame")
            first  = one("StudyFirstPostDate")
            upd    = one("LastUpdateSubmitDate")
            country= many("LocationCountry")
            sponsor= one("LeadSponsorName")

            # Compose a readable body (we’ll canonicalize later)
            parts: List[str] = []
            if status: parts.append(f"Status: {status}")
            if phase:  parts.append(f"Phase: {phase}")
            if stype:  parts.append(f"Type: {stype}")
            if conds:  parts.append("Conditions: " + "; ".join(conds))
            if inter_names: parts.append("Interventions: " + "; ".join(inter_names))
            if p_out:  parts.append("Primary Outcomes: " + "; ".join(p_out))
            if s_out:  parts.append("Secondary Outcomes: " + "; ".join(s_out))
            if elig:   parts.append("Eligibility:\n" + elig)
            text = "\n\n".join(parts) if parts else None

            prov = Provenance(
                source="clinicaltrials",
                source_url=f"https://clinicaltrials.gov/study/{nctid}" if nctid else "https://clinicaltrials.gov/",
                license="ClinicalTrials.gov terms of use",
                retrieved_at=dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                hash=None,
            )
            yield RawDoc(
                id=f"ctgov:{nctid}" if nctid else f"ctgov:auto:{abs(hash(str(row)))%10**12}",
                title=title,
                text=text,
                meta={
                    "type": "trial",
                    "nctid": nctid,
                    "status": status,
                    "conditions": conds,
                    "intervention_types": inter_types,
                    "intervention_names": inter_names,
                    "primary_outcomes": p_out,
                    "primary_outcomes_timeframe": p_tf,
                    "secondary_outcomes": s_out,
                    "secondary_outcomes_timeframe": s_tf,
                    "phase": phase, "study_type": stype,
                    "first_posted": first, "last_update": upd,
                    "countries": country,
                    "lang": "en",
                    "raw": row,
                },
                prov=prov,
            )
