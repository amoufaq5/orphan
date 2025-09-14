# src/scrapers/run.py
import os
import sys
import argparse
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode

try:
    import yaml
except Exception:
    print("YAML support required. Install with: pip install pyyaml", file=sys.stderr)
    raise

from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.logger import get_logger
from src.scrapers.http import make_session
from src.scrapers.ctgov import fetch_term as ctgov_fetch_term

log = get_logger("scrape-runner")


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
    p.add_argument("--sources", nargs="+", required=True, help="Sources to run (e.g., clinicaltrials)")
    # Global overrides
    p.add_argument("--only", nargs="+", help="Limit to these terms")
    p.add_argument("--max-terms", type=int, default=None, help="Cap number of terms")
    p.add_argument("--page-size", type=int, default=None, help="Override page_size")
    p.add_argument("--max-pages", type=int, default=None, help="Override max_pages")
    p.add_argument("--status-filter", type=str, default=None, help="Override status filter (e.g., RECRUITING)")
    p.add_argument("--dry-run", action="store_true", help="Plan only; no network calls")
    p.add_argument("--workers", type=int, default=2, help="Parallel workers (by term). Use 2–4 to stay polite.")
    return p.parse_args(argv)


def _load_terms(src_conf: Dict[str, Any], overrides: argparse.Namespace) -> List[str]:
    terms: List[str] = src_conf.get("disease_terms", []) or []
    terms_file = src_conf.get("disease_terms_file")
    if terms_file and os.path.exists(terms_file):
        with open(terms_file, "r", encoding="utf-8") as f:
            file_terms = [ln.strip() for ln in f if ln.strip()]
        terms = terms + file_terms if terms else file_terms
    if overrides.only:
        terms = overrides.only
    if overrides.max_terms:
        terms = terms[:overrides.max_terms]
    return terms


def run_clinicaltrials(conf: Dict[str, Any], overrides: argparse.Namespace) -> int:
    src_conf = conf.get("clinicaltrials", {})
    out_dir = src_conf.get("out", "./data/raw/clinicaltrials")
    page_size = overrides.page_size if overrides.page_size else src_conf.get("page_size", 25)
    max_pages = overrides.max_pages if overrides.max_pages else src_conf.get("max_pages", 40)
    status_filter = overrides.status_filter if overrides.status_filter is not None else src_conf.get("status_filter")
    workers = max(1, overrides.workers)

    terms = _load_terms(src_conf, overrides)
    if not terms:
        log.info("[ctgov] No terms found. Check config or use --only.")
        log.info(f"[clinicaltrials] fetched=0, shards={out_dir}")
        return 0

    log.info(f"[ctgov] mode=disease_terms queries={len(terms)} page_size={page_size} status_filter={status_filter or 'ANY'} workers={workers}")

    # v2 debug URL preview
    preview_params = {"query.cond": terms[0], "pageSize": str(page_size), "format": "json"}
    if status_filter:
        preview_params["filter.overallStatus"] = status_filter
    debug_url = f"https://clinicaltrials.gov/api/v2/studies?{urlencode(preview_params)}"
    log.info(f"[ctgov] debug first url: {debug_url}")

    if overrides.dry_run:
        log.info("[ctgov] dry-run enabled, skipping network calls.")
        return 0

    os.makedirs(out_dir, exist_ok=True)

    total_saved = 0
    if workers == 1:
        # sequential (safest)
        session = make_session()
        for term in terms:
            try:
                total_saved += ctgov_fetch_term(
                    term=term, out_dir=out_dir, page_size=page_size,
                    max_pages=max_pages, status_filter=status_filter, session=session
                )
            except Exception as e:
                log.error(f"[ctgov] Term '{term}' failed: {e}")
    else:
        # concurrent by term; each worker uses its own session (simpler & thread-safe)
        def _do(term: str) -> int:
            return ctgov_fetch_term(term=term, out_dir=out_dir, page_size=page_size,
                                    max_pages=max_pages, status_filter=status_filter, session=None)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_map = {ex.submit(_do, t): t for t in terms}
            for fut in as_completed(fut_map):
                term = fut_map[fut]
                try:
                    saved = fut.result()
                    total_saved += int(saved or 0)
                    log.info(f"[ctgov] DONE term='{term}' saved={saved}")
                except Exception as e:
                    log.error(f"[ctgov] Term '{term}' failed: {e}")

    log.info(f"[clinicaltrials] fetched={total_saved}, shards={out_dir}")
    return total_saved


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    requested_sources = [s.lower() for s in args.sources]
    log.info(f"Sources to run: {requested_sources}")

    runners = {"clinicaltrials": run_clinicaltrials}

    overall = 0
    had_errors = False
    for src in requested_sources:
        fn = runners.get(src)
        if not fn:
            log.error(f"Unknown source: {src}")
            had_errors = True
            continue
        try:
            overall += int(fn(cfg, args) or 0)
        except KeyboardInterrupt:
            log.error(f"[{src}] interrupted by user.")
            had_errors = True
            break
        except Exception as e:
            log.error(f"[{src}] runner error: {e}")
            had_errors = True

    if had_errors:
        sys.exit(2)
    log.info(f"All done. Total items fetched: {overall}")
    sys.exit(0)


if __name__ == "__main__":
    main()
