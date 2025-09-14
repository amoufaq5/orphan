# src/scrapers/clinicaltrials.py
from __future__ import annotations
import datetime as dt
from typing import Any, Dict, Iterable, List, Tuple

from .base_scraper import BaseScraper
from .registry import register
from ..utils.logger import get_logger
from ..utils.schemas import RawDoc, Provenance

log = get_logger("ctgov")

API = "https://clinicaltrials.gov/api/query/study_fields"

# Fields chosen for downstream usefulness; extend if needed.
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

def _one(row: Dict[str, Any], field: str) -> str | None:
    v = row.get(field) or []
    if v and isinstance(v, list) and isinstance(v[0], str):
        s = v[0].strip()
        return s if s else None
    return None

def _many(row: Dict[str, Any], field: str) -> List[str]:
    v = row.get(field) or []
    if isinstance(v, list):
        return [x.strip() for x in v if isinstance(x, str) and x.strip()]
    return []

def _expr_for(term: str) -> str:
    """
    Build a broad ClinicalTrials.gov Study Fields query expression for a disease/drug term.
    Targets Condition + Title fields for recall.
    """
    t = (term or "").strip()
    if not t:
        # Fallback: everything with any status
        return "AREA[OverallStatus] *"
    # OR across several fields for the same term
    return f'(AREA[Condition] "{t}") OR (AREA[BriefTitle] "{t}") OR (AREA[OfficialTitle] "{t}")'

@register("clinicaltrials")
class ClinicalTrialsGovScraper(BaseScraper):
    """
    ClinicalTrials.gov Study Fields API scraper.
    - Supports a single `expr` OR a list of terms via `expr_list` (converted with _expr_for).
    - Paginates using min_rnk/max_rnk (1-based, inclusive).
    - Respects BaseScraper rate limits/retries.
    """
    name = "clinicaltrials"

    def __init__(
        self,
        *args,
        expr: str | None = None,
        expr_list: List[str] | None = None,
        page_size: int = 100,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.expr = expr
        self.expr_list = expr_list or []
        self.page_size = max(1, min(1000, int(page_size)))

    # -------- BaseScraper interface --------

    def build_requests(self) -> Iterable[Tuple[str, Dict[str, Any]]]:
        """
        Yield (url, params) tuples for the runner. We don’t know totals upfront,
        so we page up to `self.max_pages` per expression.
        """
        queries: List[str] = self.expr_list if self.expr_list else [self.expr or "AREA[OverallStatus] *"]
        for q in queries:
            # Convert terms → expressions when expr_list is used
            q_expr = _expr_for(q) if self.expr_list else q
            for p in range(self.max_pages):
                start = p * self.page_size + 1
                end = start + self.page_size - 1
                params = {
                    "expr": q_expr,
                    "fields": ",".join(FIELDS),
                    "min_rnk": start,
                    "max_rnk": end,
                    "fmt": "json",
                }
                yield API, params

    def parse(self, payload: Dict[str, Any]) -> Iterable[RawDoc]:
        """
        Transform StudyFields JSON payload into RawDoc rows with provenance.
        """
        data = (((payload.get("StudyFieldsResponse") or {}).get("StudyFields")) or [])
        retrieved = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

        for row in data:
            nctid  = _one(row, "NCTId")
            title  = _one(row, "OfficialTitle") or _one(row, "BriefTitle")
            status = _one(row, "OverallStatus")
            conds  = _many(row, "Condition")
            inter_types = _many(row, "InterventionType")
            inter_names = _many(row, "InterventionName")
            elig   = _one(row, "EligibilityCriteria")
            phase  = _one(row, "Phase")
            stype  = _one(row, "StudyType")
            p_out  = _many(row, "PrimaryOutcomeMeasure")
            p_tf   = _many(row, "PrimaryOutcomeTimeFrame")
            s_out  = _many(row, "SecondaryOutcomeMeasure")
            s_tf   = _many(row, "SecondaryOutcomeTimeFrame")
            first  = _one(row, "StudyFirstPostDate")
            upd    = _one(row, "LastUpdateSubmitDate")
            country= _many(row, "LocationCountry")
            sponsor= _one(row, "LeadSponsorName")
            collab = _many(row, "CollaboratorName")

            parts: List[str] = []
            if status: parts.append(f"Status: {status}")
            if phase:  parts.append(f"Phase: {phase}")
            if stype:  parts.append(f"Type: {stype}")
            if sponsor: parts.append(f"Sponsor: {sponsor}")
            if collab: parts.append("Collaborators: " + "; ".join(collab))
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
                retrieved_at=retrieved,
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
                    "phase": phase,
                    "study_type": stype,
                    "first_posted": first,
                    "last_update": upd,
                    "countries": country,
                    "sponsor": sponsor,
                    "collaborators": collab,
                    "lang": "en",
                    "raw": row,
                    "query_expr": (self.expr or None),
                },
                prov=prov,
            )
