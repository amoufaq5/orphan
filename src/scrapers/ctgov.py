# src/scrapers/ctgov.py
import os, json, math
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode

from .http import make_session, get_json
from src.utils.logger import get_logger

log = get_logger("ctgov")

API_V2 = "https://clinicaltrials.gov/api/v2/studies"

# Map human-friendly status to v2 enum (case-insensitive)
_STATUS_MAP = {
    "recruiting": "RECRUITING",
    "not yet recruiting": "NOT_YET_RECRUITING",
    "active, not recruiting": "ACTIVE_NOT_RECRUITING",
    "enrolling by invitation": "ENROLLING_BY_INVITATION",
    "completed": "COMPLETED",
    "terminated": "TERMINATED",
    "suspended": "SUSPENDED",
    "withdrawn": "WITHDRAWN",
    "unknown status": "UNKNOWN",
    # add others as you need
}

DEFAULT_FIELDS = [
    # v2 returns full study; we keep shards small by writing raw slices per page.
    # (No field projection param in v2; you can post-filter locally later.)
]

def _normalize_status(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    key = s.strip().lower()
    return _STATUS_MAP.get(key, s)  # accept direct enum if user already passed RECRUITING

def _build_params(term: str,
                  page_size: int,
                  page_token: Optional[str],
                  status_filter: Optional[str],
                  count_total: bool = True) -> Dict[str, str]:
    """
    v2 uses flat query.* and filter.* params.
      - query.cond, query.term, query.loc, ... (we use condition search here)
      - filter.overallStatus expects enums like RECRUITING
      - pageSize (1..100? docs show typical 20)
      - pageToken for pagination
      - format=json (default), countTotal=true for totalCount
    """
    params: Dict[str, str] = {
        "query.cond": term,
        "pageSize": str(page_size),
        "format": "json",
    }
    if count_total:
        params["countTotal"] = "true"
    if page_token:
        params["pageToken"] = page_token
    status_enum = _normalize_status(status_filter)
    if status_enum:
        params["filter.overallStatus"] = status_enum
    return params

def _page_fetch(session, term: str, page_size: int, page_token: Optional[str], status_filter: Optional[str]) -> Tuple[List[dict], Optional[str], int]:
    params = _build_params(term, page_size, page_token, status_filter)
    data = get_json(session, API_V2, params)
    studies = data.get("studies", []) or []
    next_token = data.get("nextPageToken")
    total = int(data.get("totalCount", 0))
    return studies, next_token, total

def fetch_term(
    term: str,
    out_dir: str,
    page_size: int = 25,
    max_pages: int = 40,
    status_filter: Optional[str] = None,
    fields: Optional[List[str]] = None,  # kept for signature compatibility; unused in v2
    session=None
) -> int:
    """
    Fetches studies for a single term using API v2, paginating with pageToken.
    Shards each page into a JSONL file.
    Returns the number of study objects saved.
    """
    os.makedirs(out_dir, exist_ok=True)
    session = session or make_session()

    # First page (probe) to get totalCount
    try:
        studies, next_token, total = _page_fetch(session, term, page_size, None, status_filter)
    except Exception as e:
        log.error(f"[ctgov] probe failed term='{term}': {e}")
        return 0

    if total == 0 and not studies:
        log.info(f"[ctgov] 0 studies for '{term}' (status={status_filter or 'ANY'}).")
        return 0

    saved = 0
    page_idx = 1

    def _write_shard(idx: int, items: List[dict]):
        shard = os.path.join(out_dir, f"{term.replace(' ','_').lower()}_{idx:04d}.jsonl")
        with open(shard, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        log.info(f"[ctgov] {term}: saved {len(items)} → {shard}")

    if studies:
        _write_shard(page_idx, studies)
        saved += len(studies)

    # Continue pages up to max_pages
    while next_token and page_idx < max_pages:
        page_idx += 1
        try:
            studies, next_token, _ = _page_fetch(session, term, page_size, next_token, status_filter)
        except Exception as e:
            log.warning(f"[ctgov] page fetch failed '{term}' p={page_idx}: {e}")
            break
        if not studies:
            break
        _write_shard(page_idx, studies)
        saved += len(studies)

    return saved
