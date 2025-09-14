# src/scrapers/run_ctgov_by_list.py
from __future__ import annotations
import argparse
import inspect
from typing import Any, Dict, List

from .clinicaltrials import ClinicalTrialsGovScraper
from ..utils.config import load_yaml
from ..utils.logger import get_logger

log = get_logger("ctgov-batch")

def _filter_kwargs_for_ctor(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the kwargs accepted by cls.__init__."""
    try:
        sig = inspect.signature(cls.__init__)
        return {k: v for k, v in kwargs.items() if k in sig.parameters}
    except Exception:
        # Fallback if signature fails
        allowed = {"user_agent", "rate_limit_per_sec", "request_timeout_sec",
                   "total_retries", "backoff_base_sec", "backoff_max_sec", "max_pages"}
        return {k: v for k, v in kwargs.items() if k in allowed}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/scrape.yaml")
    ap.add_argument("--diseases", default="conf/diseases.txt")
    ap.add_argument("--page_size", type=int, default=None, help="Override page size for ct.gov (optional)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    scrape = cfg.get("scrape", {})

    # ---- read disease terms ----
    with open(args.diseases, "r", encoding="utf-8") as f:
        terms: List[str] = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if not terms:
        log.error("No disease terms found in %s", args.diseases)
        return

    # ---- prepare ctor kwargs (only pass what the scraper supports) ----
    ctor_kwargs = dict(
        user_agent=scrape.get("user_agent", "orphan-studio/0.1 (contact: ops@orphan.local)"),
        rate_limit_per_sec=scrape.get("rate", {}).get("requests_per_second", 3),
        request_timeout_sec=scrape.get("request_timeout_sec", 30),
        total_retries=scrape.get("total_retries", 5),
        backoff_base_sec=scrape.get("backoff_base_sec", 0.5),
        backoff_max_sec=scrape.get("backoff_max_sec", 8),
        max_pages=scrape.get("paging", {}).get("max_pages_per_source", 2),
        page_size=args.page_size if args.page_size is not None else None,  # only if the scraper supports it
    )
    ctor_kwargs = {k: v for k, v in ctor_kwargs.items() if v is not None}
    ctor_kwargs = _filter_kwargs_for_ctor(ClinicalTrialsGovScraper, ctor_kwargs)

    # Instantiate with expr_list + filtered ctor kwargs
    sc = ClinicalTrialsGovScraper(expr_list=terms, **ctor_kwargs)

    # ---- set optional attributes post-init if present ----
    # out_dir: where shards will be written (default if attribute exists)
    out_dir = scrape.get("out_dir", "data/shards")
    if hasattr(sc, "out_dir"):
        setattr(sc, "out_dir", out_dir)

    # max_pages override from YAML if attribute exists (safeguard)
    mp = scrape.get("paging", {}).get("max_pages_per_source", None)
    if mp is not None and hasattr(sc, "max_pages"):
        setattr(sc, "max_pages", int(mp))

    # page_size optional override if attribute exists and user passed CLI flag
    if args.page_size is not None and hasattr(sc, "page_size"):
        setattr(sc, "page_size", int(args.page_size))

    log.info("Running ClinicalTrialsGovScraper with %d disease terms; shards -> %s", len(terms), out_dir)
    sc.run()  # BaseScraper should handle request loop + writing shards

if __name__ == "__main__":
    main()
