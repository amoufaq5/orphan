# src/scrapers/ctgov.py
import os, math, json
from typing import Iterator, Dict, List, Optional
from urllib.parse import urlencode
from .http import make_session, get_json
from src.utils.logger import get_logger

log = get_logger("ctgov")

API = "https://clinicaltrials.gov/api/query/study_fields"
DEFAULT_FIELDS = [
    "NCTId","BriefTitle","OfficialTitle","OverallStatus","Condition",
    "InterventionType","InterventionName","Phase","StudyType",
    "PrimaryOutcomeMeasure","StudyFirstPostDate","LastUpdateSubmitDate"
]

def _build_expr(term: str, status_filter: Optional[str]) -> str:
    # Keep expression simple & valid; status filter is optional
    base = f'(AREA[Condition] "{term}") OR (AREA[BriefTitle] "{term}") OR (AREA[OfficialTitle] "{term}")'
    if status_filter:
        # ClinicalTrials supports OverallStatus as an AREA
        base = f"(({base})) AND (AREA[OverallStatus] {status_filter})"
    return base

def fetch_term(
    term: str,
    out_dir: str,
    page_size: int = 25,
    max_pages: int = 40,
    status_filter: Optional[str] = None,
    fields: Optional[List[str]] = None,
    session=None
) -> int:
    os.makedirs(out_dir, exist_ok=True)
    fields = fields or DEFAULT_FIELDS
    session = session or make_session()

    expr = _build_expr(term, status_filter)
    # First probe to get total count
    params_probe = dict(
        expr=expr,
        fields="NCTId",  # light probe
        min_rnk=1,
        max_rnk=1,
        fmt="json",
    )
    try:
        probe = get_json(session, API, params_probe)
    except Exception as e:
        log.error(f"[ctgov] probe failed for term='{term}': {e}")
        return 0

    try:
        n_found = int(probe["StudyFieldsResponse"]["NStudiesFound"])
    except Exception:
        log.warning(f"[ctgov] probe parse issue for term='{term}', skipping.")
        return 0

    if n_found == 0:
        log.info(f"[ctgov] 0 studies for term='{term}' (status={status_filter or 'ANY'}).")
        return 0

    # Page through results
    pages = int(math.ceil(min(n_found, page_size * max_pages) / page_size))
    saved = 0
    for p in range(pages):
        start = p * page_size + 1
        end = min(start + page_size - 1, page_size * max_pages)
        params = dict(
            expr=expr,
            fields=",".join(fields),
            min_rnk=start,
            max_rnk=end,
            fmt="json",
        )
        try:
            data = get_json(session, API, params)
        except Exception as e:
            log.warning(f"[ctgov] page fetch failed term='{term}' p={p+1}/{pages}: {e}")
            continue

        items = data.get("StudyFieldsResponse", {}).get("StudyFields", []) or []
        if not items:
            # Nothing returned: stop early
            log.info(f"[ctgov] empty page for term='{term}' at p={p+1}, stopping.")
            break

        # Shard output per page to avoid large files
        shard_path = os.path.join(out_dir, f"{term.replace(' ','_').lower()}_{start:06d}-{end:06d}.jsonl")
        with open(shard_path, "w", encoding="utf-8") as f:
            for row in items:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        saved += len(items)
        log.info(f"[ctgov] {term}: saved {len(items)} → {shard_path}")

        # Defensive early stop if server returned fewer than requested
        if len(items) < (end - start + 1):
            break

    return saved
