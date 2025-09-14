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
from src.scrapers.ctgov import fetch_term as ctgov_fetch_term  # ClinicalTrials.gov v2
from src.scrapers.pubmed import fetch_term as pubmed_fetch_term

log = get_logger("scrape-runner")


# ----------------------------
# CLI & config helpers
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
    p.add_argument("--sources", nargs="+", required=True,
                   help="Sources to run (e.g., clinicaltrials pubmed)")

    # Optional global overrides
    p.add_argument("--only", nargs="+", help="Limit to these terms (override config lists)")
    p.add_argument("--max-terms", type=int, default=None, help="Cap number of terms")
    p.add_argument("--page-size", type=int, default=None, help="Override page_size (per-source if supported)")
    p.add_argument("--max-pages", type=int, default=None, help="Override max_pages (per-source if supported)")
    p.add_argument("--status-filter", type=str, default=None, help="ClinicalTrials: override status filter (e.g., RECRUITING or '')")
    p.add_argument("--dry-run", action="store_true", help="Show plan without making network calls")
    p.add_argument("--workers", type=int, default=2, help="Parallel workers by term (2–4 recommended)")
    return p.parse_args(argv)


def _load_terms(src_conf: Dict[str, Any], overrides: argparse.Namespace, tag: str) -> List[str]:
    """
    Load disease terms from YAML `disease_terms` and/or `disease_terms_file`.
    Apply CLI overrides. Logs a clear error if the file path is missing.
    """
    terms: List[str] = src_conf.get("disease_terms", []) or []

    terms_file = src_conf.get("disease_terms_file")
    if terms_file:
        if not os.path.exists(terms_file):
            log.error(f"[{tag}] disease_terms_file not found: {os.path.abspath(terms_file)}")
        else:
            with open(terms_file, "r", encoding="utf-8") as f:
                file_terms = [ln.strip() for ln in f if ln.strip()]
            terms = terms + file_terms if terms else file_terms

    if overrides.only:
        terms = overrides.only
    if overrides.max_terms:
        terms = terms[:overrides.max_terms]
    return terms


# ----------------------------
# ClinicalTrials.gov (API v2)
# ----------------------------
def _ctgov_debug_url(term: str, page_size: int, status_filter: Optional[str]) -> str:
    params = {"query.cond": term, "pageSize": str(page_size), "format": "json"}
    if status_filter is not None and status_filter != "":
        params["filter.overallStatus"] = status_filter
    return f"https://clinicaltrials.gov/api/v2/studies?{urlencode(params)}"


def run_clinicaltrials(conf: Dict[str, Any], overrides: argparse.Namespace) -> int:
    src_conf = conf.get("clinicaltrials", {})
    out_dir = src_conf.get("out", "./data/raw/clinicaltrials")
    page_size = overrides.page_size if overrides.page_size else src_conf.get("page_size", 25)
    max_pages = overrides.max_pages if overrides.max_pages else src_conf.get("max_pages", 40)
    status_filter = overrides.status_filter if overrides.status_filter is not None else src_conf.get("status_filter", "")

    workers = max(1, overrides.workers)
    terms = _load_terms(src_conf, overrides, tag="ctgov")

    if not terms:
        log.info("[ctgov] No terms found. Check config or use --only.")
        log.info(f"[clinicaltrials] fetched=0, shards={out_dir}")
        return 0

    log.info(f"[ctgov] mode=disease_terms queries={len(terms)} page_size={page_size} "
             f"status_filter={status_filter if status_filter != '' else 'ALL'} workers={workers}")

    # Debug preview
    log.info(f"[ctgov] debug first url: {_ctgov_debug_url(terms[0], page_size, status_filter)}")

    if overrides.dry_run:
        log.info("[ctgov] dry-run enabled, skipping network calls.")
        return 0

    os.makedirs(out_dir, exist_ok=True)

    total_saved = 0
    if workers == 1:
        session = make_session()
        for term in terms:
            try:
                total_saved += ctgov_fetch_term(
                    term=term,
                    out_dir=out_dir,
                    page_size=page_size,
                    max_pages=max_pages,
                    status_filter=status_filter,
                    session=session,
                )
            except Exception as e:
                log.error(f"[ctgov] Term '{term}' failed: {e}")
    else:
        # Each worker creates its own session inside fetch_term (session=None)
        def _do(t: str) -> int:
            return ctgov_fetch_term(
                term=t,
                out_dir=out_dir,
                page_size=page_size,
                max_pages=max_pages,
                status_filter=status_filter,
                session=None,
            )

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_do, t): t for t in terms}
            for fut in as_completed(futs):
                t = futs[fut]
                try:
                    s = fut.result()
                    total_saved += int(s or 0)
                    log.info(f"[ctgov] DONE term='{t}' saved={s}")
                except Exception as e:
                    log.error(f"[ctgov] Term '{t}' failed: {e}")

    log.info(f"[clinicaltrials] fetched={total_saved}, shards={out_dir}")
    return total_saved


# ----------------------------
# PubMed (E-utilities)
# ----------------------------
def _pubmed_query_preview(tmpl: str, term: str, since_year: str, until_year: str) -> str:
    q = tmpl.replace("%TERM%", term)
    if since_year or until_year:
        sy = since_year or "1800"
        uy = until_year or "3000"
        q = f"({q}) AND ({sy}[pdat] : {uy}[pdat])"
    # Show an ESearch-style preview URL
    params = {"db": "pubmed", "term": q, "retmode": "json", "retmax": "5", "retstart": "0"}
    return f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{urlencode(params)}"


def run_pubmed(conf: Dict[str, Any], overrides: argparse.Namespace) -> int:
    src_conf = conf.get("pubmed", {})
    out_dir = src_conf.get("out", "./data/raw/pubmed")

    # Tuning knobs
    retmax = int(src_conf.get("retmax", 1000))  # ESearch page size
    max_docs = int(src_conf.get("max_docs_per_term", 5000))
    batch = int(src_conf.get("efetch_batch_size", 200))  # EFetch ids per call (<=200 recommended)
    tmpl = src_conf.get("search_query_template", '("%TERM%"[Title/Abstract]) OR ("%TERM%"[MeSH Terms])')
    since_year = str(src_conf.get("since_year", "")) or ""
    until_year = str(src_conf.get("until_year", "")) or ""

    workers = max(1, overrides.workers)
    terms = _load_terms(src_conf, overrides, tag="pubmed")

    if not terms:
        log.info("[pubmed] No terms found. Check config or use --only.")
        log.info(f"[pubmed] fetched=0, shards={out_dir}")
        return 0

    log.info(f"[pubmed] queries={len(terms)} retmax={retmax} max_docs/term={max_docs} "
             f"batch={batch} workers={workers}")
    log.info(f"[pubmed] debug first url: {_pubmed_query_preview(tmpl, terms[0], since_year, until_year)}")

    if overrides.dry_run:
        log.info("[pubmed] dry-run enabled, skipping network calls.")
        return 0

    os.makedirs(out_dir, exist_ok=True)

    total_saved = 0
    if workers == 1:
        session = make_session()
        for term in terms:
            try:
                total_saved += pubmed_fetch_term(
                    term=term,
                    out_dir=out_dir,
                    retmax=retmax,
                    max_docs_per_term=max_docs,
                    efetch_batch_size=batch,
                    search_query_template=tmpl,
                    since_year=since_year,
                    until_year=until_year,
                    session=session,
                )
            except Exception as e:
                log.error(f"[pubmed] Term '{term}' failed: {e}")
    else:
        def _do(t: str) -> int:
            return pubmed_fetch_term(
                term=t,
                out_dir=out_dir,
                retmax=retmax,
                max_docs_per_term=max_docs,
                efetch_batch_size=batch,
                search_query_template=tmpl,
                since_year=since_year,
                until_year=until_year,
                session=None,
            )

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_do, t): t for t in terms}
            for fut in as_completed(futs):
                t = futs[fut]
                try:
                    s = fut.result()
                    total_saved += int(s or 0)
                    log.info(f"[pubmed] DONE term='{t}' saved={s}")
                except Exception as e:
                    log.error(f"[pubmed] Term '{t}' failed: {e}")

    log.info(f"[pubmed] fetched={total_saved}, shards={out_dir}")
    return total_saved


# ----------------------------
# Dispatch & entrypoint
# ----------------------------
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    requested_sources = [s.lower() for s in args.sources]
    log.info(f"Sources to run: {requested_sources}")

    runners = {
        "clinicaltrials": run_clinicaltrials,
        "pubmed": run_pubmed,
        # "openfda": run_openfda,  # add when ready
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

    if had_errors:
        sys.exit(2)
    else:
        log.info(f"All done. Total items fetched: {overall_count}")
        sys.exit(0)


if __name__ == "__main__":
    main()
