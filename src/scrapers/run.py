# src/scrapers/run.py
import os
import sys
import argparse
from typing import List, Dict, Any, Optional

try:
    import yaml
except Exception as e:
    print("YAML support required. Install with: pip install pyyaml", file=sys.stderr)
    raise

from src.utils.logger import get_logger
from src.scrapers.http import make_session
from src.scrapers.ctgov import fetch_term as ctgov_fetch_term, _expr as ctgov_expr

log = get_logger("scrape-runner")


# ----------------------------
# Config & CLI
# ----------------------------
def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Orphan scrapers runner")
    p.add_argument("--config", "-c", required=True, help="Path to scrape YAML config")
    p.add_argument("--sources", nargs="+", required=True, help="Sources to run (e.g. clinicaltrials openfda)")
    p.add_argument("--only", nargs="+", help="Limit to these terms (override config list)")
    p.add_argument("--max-terms", type=int, default=None, help="Cap number of terms to process")
    p.add_argument("--page-size", type=int, default=None, help="Override page_size")
    p.add_argument("--max-pages", type=int, default=None, help="Override max_pages")
    p.add_argument("--status-filter", type=str, default=None, help="Override status filter")
    p.add_argument("--dry-run", action="store_true", help="Show plan without scraping")
    return p.parse_args(argv)


# ----------------------------
# ClinicalTrials runner
# ----------------------------
def run_clinicaltrials(conf: Dict[str, Any], overrides: argparse.Namespace) -> int:
    src_conf = conf.get("clinicaltrials", {})
    out_dir = src_conf.get("out", "./data/raw/clinicaltrials")
    page_size = overrides.page_size if overrides.page_size else src_conf.get("page_size", 25)
    max_pages = overrides.max_pages if overrides.max_pages else src_conf.get("max_pages", 40)
    status_filter = overrides.status_filter if overrides.status_filter is not None else src_conf.get("status_filter")

    # Terms from YAML
    terms: List[str] = src_conf.get("disease_terms", []) or []

    # Optionally load from file
    terms_file = src_conf.get("disease_terms_file")
    if terms_file and os.path.exists(terms_file):
        with open(terms_file, "r", encoding="utf-8") as f:
            file_terms = [ln.strip() for ln in f if ln.strip()]
        if not terms:
            terms = file_terms
        else:
            terms.extend(file_terms)

    # CLI overrides
    if overrides.only:
        terms = overrides.only
    if overrides.max_terms:
        terms = terms[:overrides.max_terms]

    # Guard against empty
    if not terms:
        log.info("[ctgov] No terms provided. Check conf or use --only.")
        log.info(f"[clinicaltrials] fetched=0, shards={out_dir}")
        return 0

    log.info(f"[ctgov] mode=disease_terms queries={len(terms)} page_size={page_size} status_filter={status_filter or 'ANY'}")

    # Debug first URL
    expr_preview = ctgov_expr(terms[0], status_filter)
    debug_url = (
        "https://clinicaltrials.gov/api/query/study_fields"
        f"?expr={expr_preview.replace(' ', '+')}"
        f"&fields=NCTId,BriefTitle,OfficialTitle,OverallStatus,Condition,InterventionType,InterventionName,"
        f"Phase,StudyType,PrimaryOutcomeMeasure,StudyFirstPostDate,LastUpdateSubmitDate"
        f"&min_rnk=1&max_rnk={page_size}&fmt=json"
    )
    log.info(f"[ctgov] debug first url: {debug_url}")

    if overrides.dry_run:
        log.info("[ctgov] dry-run: skipping network calls.")
        return 0

    os.makedirs(out_dir, exist_ok=True)
    session = make_session()
    total_saved = 0

    for idx, term in enumerate(terms, 1):
        try:
            saved = ctgov_fetch_term(
                term=term,
                out_dir=out_dir,
                page_size=page_size,
                max_pages=max_pages,
                status_filter=status_filter,
                session=session,
            )
            total_saved += saved
        except Exception as e:
            log.error(f"[ctgov] Term '{term}' failed: {e}")

    log.info(f"[clinicaltrials] fetched={total_saved}, shards={out_dir}")
    return total_saved


# ----------------------------
# Dispatch table
# ----------------------------
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    runners = {
        "clinicaltrials": run_clinicaltrials,
        # "openfda": run_openfda,
        # "pubmed": run_pubmed,
    }

    requested_sources = [s.lower() for s in args.sources]
    log.info(f"Sources to run: {requested_sources}")

    overall = 0
    for src in requested_sources:
        fn = runners.get(src)
        if not fn:
            log.error(f"Unknown source: {src}")
            continue
        overall += fn(cfg, args)

    log.info(f"All done. Total items fetched: {overall}")


if __name__ == "__main__":
    main()
