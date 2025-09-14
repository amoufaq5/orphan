from __future__ import annotations
import datetime as dt
from typing import Any, Dict, Iterable, List, Tuple, Optional
from .base_scraper import BaseScraper
from .registry import register
from ..utils.logger import get_logger
from ..utils.schemas import RawDoc, Provenance

log = get_logger("ctgov")

API = "https://clinicaltrials.gov/api/query/study_fields"

# ====== Built-in disease term list (edit/extend freely) ======
DISEASE_TERMS: List[str] = [
    # Common chronic
    "asthma", "chronic obstructive pulmonary disease", "copd",
    "diabetes mellitus", "hypertension", "hyperlipidemia",
    "ischemic heart disease", "coronary artery disease",
    "heart failure", "atrial fibrillation", "stroke",
    "chronic kidney disease", "ckd", "end stage renal disease",
    "nonalcoholic fatty liver disease", "nafld", "cirrhosis",
    # Infectious
    "covid-19", "influenza", "pneumonia", "tuberculosis",
    "hepatitis b", "hepatitis c", "hiv infection", "aids",
    # Neuro / Psych
    "migraine", "epilepsy", "multiple sclerosis",
    "parkinson disease", "alzheimer disease", "depression", "anxiety",
    "schizophrenia", "bipolar disorder",
    # Rheum / Pain
    "osteoarthritis", "rheumatoid arthritis", "psoriatic arthritis",
    "gout", "fibromyalgia", "lupus",
    # Oncology (high-level terms)
    "breast cancer", "lung cancer", "colorectal cancer", "prostate cancer",
    "pancreatic cancer", "leukemia", "lymphoma", "melanoma",
    "glioblastoma", "ovarian cancer", "gastric cancer", "hepatocellular carcinoma",
    # Endocrine
    "hypothyroidism", "hyperthyroidism", "cushing syndrome", "addison disease",
    "osteoporosis",
    # GI
    "irritable bowel syndrome", "ibs", "inflammatory bowel disease",
    "crohn disease", "ulcerative colitis", "celiac disease",
    # Women’s health
    "endometriosis", "polycystic ovary syndrome", "pcos",
    "postpartum depression", "preeclampsia",
    # Pediatrics
    "respiratory syncytial virus", "rsv", "bronchiolitis",
    # Hem/Immune
    "sickle cell disease", "thalassemia", "hemophilia",
    "myelodysplastic syndromes",
    # Derm
    "psoriasis", "atopic dermatitis", "eczema", "acne vulgaris",
]

# StudyFields we’ll fetch (extend if you need more)
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

def _expr_for(term: str) -> str:
    """
    Compose a broad ct.gov StudyFields expression for a disease term.
    Matches in Condition/BriefTitle/OfficialTitle.
    """
    t = (term or "").strip()
    if not t:
        return "AREA[OverallStatus] *"
    # Quote the term; OR across common fields for recall
    return f'(AREA[Condition] "{t}") OR (AREA[BriefTitle] "{t}") OR (AREA[OfficialTitle] "{t}")'

@register("clinicaltrials")
class ClinicalTrialsGovScraper(BaseScraper):
    """
    ClinicalTrials.gov Study Fields scraper.

    Default behavior:
      • Iterates over a built-in list of DISEASE_TERMS and queries each with a broad expression.
      • Respects global paging/rate limits from conf/scrape.yaml (max_pages_per_source, etc.).
      • Emits RawDoc shards with provenance.

    You can override defaults via YAML by passing kwargs for this scraper (if your
    runner supports per-scraper kwargs), e.g.:
      clinicaltrials:
        page_size: 200
        use_disease_list: true
        disease_terms: ["your", "custom", "terms"]
        expr: 'AREA[OverallStatus] Recruiting'          # overrides the list if set
        expr_list: ['AREA[Condition] "migraine"', ...]  # explicit list of expressions
    """

    name = "clinicaltrials"

    def __init__(
        self,
        *args,
        expr: Optional[str] = None,
        expr_list: Optional[List[str]] = None,
        page_size: int = 100,
        use_disease_list: bool = True,
        disease_terms: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.page_size = max(1, min(1000, page_size))

        # Priority of query sources:
        # 1) explicit expr_list
        # 2) single expr
        # 3) built-in disease list (default)
        if expr_list:
            self.queries: List[str] = expr_list
            self.query_mode = "expr_list"
        elif expr:
            self.queries = [expr]
            self.query_mode = "expr"
        else:
            terms = disease_terms if (use_disease_list and disease_terms) else (DISEASE_TERMS if use_disease_list else [])
            if not terms:
                # If no terms and no expr provided, fall back to a broad catch-all
                self.queries = ["AREA[OverallStatus] *"]
                self.query_mode = "fallback_all"
            else:
                self.queries = terms
                self.query_mode = "disease_terms"

        log.info(f"[ctgov] mode={self.query_mode} queries={len(self.queries)} page_size={self.page_size}")

    # BaseScraper calls this to enumerate requests; we iterate our query list.
    def build_requests(self) -> Iterable[Tuple[str, Dict[str, Any]]]:
        for q in self.queries:
            # If we are using disease terms, transform to a StudyFields expr
            expr = _expr_for(q) if self.query_mode == "disease_terms" else q
            for p in range(self.max_pages):
                start = p * self.page_size + 1
                end = start + self.page_size - 1
                params = {
                    "expr": expr,
                    "fields": ",".join(FIELDS),
                    "min_rnk": start,
                    "max_rnk": end,
                    "fmt": "json",
                }
                yield API, params

    def parse(self, payload: Dict[str, Any]) -> Iterable[RawDoc]:
        data = (((payload.get("StudyFieldsResponse") or {}).get("StudyFields")) or [])
        retrieved_at = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

        for row in data:
            def one(field: str) -> Optional[str]:
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
                retrieved_at=retrieved_at,
                hash=None,
            )

            meta_extra = {
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
                "lead_sponsor": sponsor,
                "lang": "en",
                "raw": row,
                # Mark how this result was fetched to aid debugging/metrics:
                "query_mode": getattr(self, "query_mode", None),
            }

            yield RawDoc(
                id=f"ctgov:{nctid}" if nctid else f"ctgov:auto:{abs(hash(str(row)))%10**12}",
                title=title,
                text=text,
                meta=meta_extra,
                prov=prov,
            )
