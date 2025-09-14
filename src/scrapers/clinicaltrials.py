from __future__ import annotations
import datetime as dt, urllib.parse
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

# Keep fields *very* compact to reduce payload size / errors
FIELDS = [
    "NCTId", "BriefTitle", "OfficialTitle", "OverallStatus",
    "Condition", "InterventionType", "InterventionName",
    "Phase", "StudyType",
    "PrimaryOutcomeMeasure",
    "StudyFirstPostDate", "LastUpdateSubmitDate",
]

def _expr_for(term: str, status_filter: Optional[str]) -> str:
    """
    Build a StudyFields expression. Optionally AND with status to shrink results.
    """
    t = (term or "").strip()
    if not t:
        base = "AREA[OverallStatus] *"
    else:
        base = f'(AREA[Condition] "{t}") OR (AREA[BriefTitle] "{t}") OR (AREA[OfficialTitle] "{t}")'
    if status_filter:
        return f"({base}) AND (AREA[OverallStatus] {status_filter})"
    return base

@register("clinicaltrials")
class ClinicalTrialsGovScraper(BaseScraper):
    """
    ClinicalTrials.gov Study Fields scraper with built-in disease sweep,
    gentler defaults, explicit JSON headers, and status filtering.
    """

    name = "clinicaltrials"

    def __init__(
        self,
        *args,
        expr: Optional[str] = None,
        expr_list: Optional[List[str]] = None,
        page_size: int = 25,                 # ↓ even gentler (25 rows)
        use_disease_list: bool = True,
        disease_terms: Optional[List[str]] = None,
        status_filter: Optional[str] = "Recruiting",  # ← try "Recruiting" first; set to None to disable
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.page_size = max(1, min(250, page_size))
        self.status_filter = status_filter

        # Priority: expr_list > expr > disease list > catch-all
        if expr_list:
            self.queries: List[str] = expr_list
            self.query_mode = "expr_list"
        elif expr:
            self.queries = [expr]
            self.query_mode = "expr"
        else:
            terms = disease_terms if (use_disease_list and disease_terms) else (DISEASE_TERMS if use_disease_list else [])
            if not terms:
                self.queries = ["AREA[OverallStatus] *"]
                self.query_mode = "fallback_all"
            else:
                self.queries = terms
                self.query_mode = "disease_terms"

        # Per-request headers (BaseScraper should merge these with global UA)
        self.request_headers = {
            "Accept": "application/json",
        }

        log.info(f"[ctgov] mode={self.query_mode} queries={len(self.queries)} page_size={self.page_size} status_filter={self.status_filter}")

    def build_requests(self) -> Iterable[Tuple[str, Dict[str, Any]]]:
        first_url_logged = False
        for q in self.queries:
            expr = _expr_for(q, self.status_filter) if self.query_mode == "disease_terms" else q
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
                # Hint to runner (if supported) to attach headers & extended timeout
                params["__headers__"] = self.request_headers
                params["__timeout__"] = 45  # seconds

                if not first_url_logged:
                    # Log the first full URL for debugging
                    debug_qs = urllib.parse.urlencode({k: v for k, v in params.items() if not str(k).startswith("__")})
                    log.info(f"[ctgov] debug first url: {API}?{debug_qs}")
                    first_url_logged = True

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
            phase  = one("Phase")
            stype  = one("StudyType")
            p_out  = many("PrimaryOutcomeMeasure")
            first  = one("StudyFirstPostDate")
            upd    = one("LastUpdateSubmitDate")

            parts: List[str] = []
            if status: parts.append(f"Status: {status}")
            if phase:  parts.append(f"Phase: {phase}")
            if stype:  parts.append(f"Type: {stype}")
            if conds:  parts.append("Conditions: " + "; ".join(conds))
            if inter_names: parts.append("Interventions: " + "; ".join(inter_names))
            if p_out:  parts.append("Primary Outcomes: " + "; ".join(p_out))
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
                "phase": phase,
                "study_type": stype,
                "primary_outcomes": p_out,
                "first_posted": first,
                "last_update": upd,
                "lang": "en",
                "raw": row,
                "query_mode": getattr(self, "query_mode", None),
            }

            yield RawDoc(
                id=f"ctgov:{nctid}" if nctid else f"ctgov:auto:{abs(hash(str(row)))%10**12}",
                title=title,
                text=text,
                meta=meta_extra,
                prov=prov,
            )
