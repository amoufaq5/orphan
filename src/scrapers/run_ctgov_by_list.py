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

    # read disease terms
    with open(args.diseases, "r", encoding="utf-8") as f:
        terms = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if not terms:
        log.error("no disease terms found.")
        return

    # use BaseScraper runner API
    sc = ClinicalTrialsGovScraper(expr_list=terms)
    sc.configure_from_yaml(args.config)  # inherits rate limits, retries, etc.
    sc.run()  # writes shards to data/shards/

if __name__ == "__main__":
    main()
