from __future__ import annotations
import argparse
from .clinicaltrials import ClinicalTrialsGovScraper
from ..utils.config import load_yaml
from ..utils.logger import get_logger

log = get_logger("ctgov-batch")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/scrape.yaml")
    ap.add_argument("--diseases", default="conf/diseases.txt")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    scrape_cfg = cfg.get("scrape", {})

    # read disease terms
    with open(args.diseases, "r", encoding="utf-8") as f:
        terms = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if not terms:
        log.error("no disease terms found.")
        return

    # instantiate with YAML params
    sc = ClinicalTrialsGovScraper(
        expr_list=terms,
        user_agent=scrape_cfg.get("user_agent", "orphan-studio/0.1"),
        max_pages=scrape_cfg.get("paging", {}).get("max_pages_per_source", 2),
        out_dir=scrape_cfg.get("out_dir", "data/shards"),
        request_timeout_sec=scrape_cfg.get("request_timeout_sec", 30),
        total_retries=scrape_cfg.get("total_retries", 5),
        backoff_base_sec=scrape_cfg.get("backoff_base_sec", 0.5),
        backoff_max_sec=scrape_cfg.get("backoff_max_sec", 8),
        rate_limit_per_sec=scrape_cfg.get("rate", {}).get("requests_per_second", 3),
    )
    sc.run()  # writes shards to out_dir

if __name__ == "__main__":
    main()
