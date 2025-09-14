import os, math, json
from typing import List, Optional
from .http import make_session, get_json
from src.utils.logger import get_logger

log = get_logger("ctgov")

API = "https://clinicaltrials.gov/api/query/study_fields"
DEFAULT_FIELDS = [
    "NCTId","BriefTitle","OfficialTitle","OverallStatus","Condition",
    "InterventionType","InterventionName","Phase","StudyType",
    "PrimaryOutcomeMeasure","StudyFirstPostDate","LastUpdateSubmitDate"
]

def _expr(term: str, status_filter: Optional[str]) -> str:
    base = f'(AREA[Condition] "{term}") OR (AREA[BriefTitle] "{term}") OR (AREA[OfficialTitle] "{term}")'
    if status_filter:
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

    expr = _expr(term, status_filter)

    # Probe for total
    try:
        probe = get_json(session, API, dict(expr=expr, fields="NCTId", min_rnk=1, max_rnk=1, fmt="json"))
        n_found = int(probe["StudyFieldsResponse"]["NStudiesFound"])
    except Exception as e:
        log.error(f"[ctgov] probe failed term='{term}': {e}")
        return 0

    if n_found == 0:
        log.info(f"[ctgov] 0 studies for '{term}' (status={status_filter or 'ANY'}).")
        return 0

    total_cap = page_size * max_pages
    pages = int(math.ceil(min(n_found, total_cap) / page_size))
    saved = 0

    for p in range(pages):
        start = p * page_size + 1
        end = min(start + page_size - 1, total_cap)
        try:
            data = get_json(session, API, dict(
                expr=expr, fields=",".join(fields),
                min_rnk=start, max_rnk=end, fmt="json"
            ))
        except Exception as e:
            log.warning(f"[ctgov] page fetch failed '{term}' p={p+1}/{pages}: {e}")
            continue

        items = data.get("StudyFieldsResponse", {}).get("StudyFields", []) or []
        if not items:
            log.info(f"[ctgov] empty page for '{term}' at p={p+1}, stopping.")
            break

        shard = os.path.join(out_dir, f"{term.replace(' ','_').lower()}_{start:06d}-{end:06d}.jsonl")
        with open(shard, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

        saved += len(items)
        log.info(f"[ctgov] {term}: saved {len(items)} → {shard}")

        # Early stop if short page
        if len(items) < (end - start + 1):
            break

    return saved
