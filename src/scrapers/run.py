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
from src.scrapers.ctgov import fetch_term as ctgov_fetch_term

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
    p = argparse.ArgumentParser(
        description="Orphan scrapers runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", "-c", required=True, help="Path to scrape YAML config")
    p.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="Which sources to run (e.g., clinicaltrials openfda pubmed)",
    )
    # Optional global overrides
    p.add_argument("--only", nargs="+", help="Limit to these terms (override config list)")
    p.add_argument("--max-terms", type=int, default=None, help="Cap number of terms to process")
    p.add_argument("--page-size", type=int, default=None, help="Override page_size for sources that support it")
    p.add_argument("--max-pages", type=int, default=None, help="Override max_pages for sources that support it")
    p.add_argument("--status-filter", type=str, default=None, help="Override status filter (e.g., Recruiting)")
    p.add_argument("--dry-run", action="store_true", help="Parse and show plan without making network calls")
    return p.parse_args(argv)


# ----------------------------
# Source runners
# ----------------------------
def run_clinicaltrials(conf: Dict[str, Any], overrides: argparse.Namespace) -> int:
    """
    Runs the ClinicalTrials.gov study_fields fetcher across terms from config,
    honoring CLI overrides. Returns the number of items (studies) saved.
    """
    src_conf = conf.get("clinicaltrials", {})
    out_dir = src_conf.get("out", "./data/raw/clinicaltrials")
    page_size = overrides.page_size if overrides.page_size else src_conf.get("page_size", 25)
    max_pages = overrides.max_pages if overrides.max_pages else src_conf.get("max_pages", 40)
    status_filter = overrides.status_filter if overrides.status_filter is not None else src_conf.get("status_filter")

    # Determine term list
    terms = src_conf.get("disease_terms", [])
    if overrides.only:
        terms = overrides.only
    if overrides.max_terms is not None:
        terms = terms[:overrides.max_terms]

    mode = "disease_terms"
    log.info(f"[ctgov] mode={mode} queries={len(terms)} page_size={page_size} status_filter={status_filter or 'ANY'}")
    if terms:
        # For visibility, show the first URL that would be requested
        from src.scrapers.ctgov import _expr as _ctgov_expr  # local helper used only for debug preview
        expr_preview = _ctgov_expr(terms[0], status_filter)
        debug_url = (
            "https://clinicaltrials.gov/api/query/study_fields"
            f"?expr={expr_preview.replace(' ', '+')}"
            f"&fields=NCTId%2CBriefTitle%2COfficialTitle%2COverallStatus%2CCondition%2CInterventionType%2CInterventionName"
            f"%2CPhase%2CStudyType%2CPrimaryOutcomeMeasure%2CStudyFirstPostDate%2CLastUpdateSubmitDate"
            f"&min_rnk=1&max_rnk={page_size}&fmt=json"
        )
        log.info(f"[ctgov] debug first url: {debug_url}")

    if overrides.dry_run:
        log.info("[ctgov] dry-run enabled, skipping network calls.")
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


# Stubs for future sources (keep interface uniform)
def run_openfda(conf: Dict[str, Any], overrides: argparse.Namespace) -> int:
    log.warning("[openfda] runner not wired yet in this file.")
    return 0


def run_pubmed(conf: Dict[str, Any], overrides: argparse.Namespace) -> int:
    log.warning("[pubmed] runner not wired yet in this file.")
    return 0


# ----------------------------
# Main orchestrator
# ----------------------------
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Some configs use 'sources: [...]' at the root; we honor CLI regardless.
    requested_sources = [s.lower() for s in args.sources]
    log.info(f"Sources to run: {requested_sources}")

    # Dispatch table
    runners = {
        "clinicaltrials": run_clinicaltrials,
        "openfda": run_openfda,
        "pubmed": run_pubmed,
    }

    overall_count = 0
    had_errors = False

    for src in requested_sources:
        fn = runners.get(src)
        if not fn:
            log.error(f"Unknown source: {src}")
            had_errors = True
            continue
        try:
            count = fn(cfg, args)
            overall_count += int(count or 0)
        except KeyboardInterrupt:
            log.error(f"[{src}] interrupted by user.")
            had_errors = True
            break
        except Exception as e:
            log.error(f"[{src}] runner error: {e}")
            had_errors = True

    # Exit code reflects whether anything failed
    if had_errors:
        # Non-zero code to help CI or scripts detect failure
        sys.exit(2)
    else:
        log.info(f"All done. Total items fetched: {overall_count}")
        sys.exit(0)


if __name__ == "__main__":
    main()
