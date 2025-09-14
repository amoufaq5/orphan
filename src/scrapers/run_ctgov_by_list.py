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
    user_agent = scrape_cfg.get("user_agent", "orphan-studio/0.1 (contact: ops@orphan.local)")

    # read disease terms
    with open(args.diseases, "r", encoding="utf-8") as f:
        terms = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if not terms:
        log.error("no disease terms found.")
        return

    # Instantiate with the ONLY required arg for BaseScraper, then load the rest from YAML
    sc = ClinicalTrialsGovScraper(expr_list=terms, user_agent=user_agent)

    # Load rate limits, retries, output dirs, etc. from YAML
    sc.configure_from_yaml(args.config)

    # Optional: sync max_pages/page_size from config if you want explicit control
    paging = scrape_cfg.get("paging", {})
    if "max_pages_per_source" in paging:
        sc.max_pages = int(paging["max_pages_per_source"])
    # If you added page_size to your clinicaltrials scraper, you can read a custom value here:
    # sc.page_size = int(scrape_cfg.get("ctgov_page_size", sc.page_size))

    sc.run()  # writes shards using configured out_dir

if __name__ == "__main__":
    main()
