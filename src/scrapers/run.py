from __future__ import annotations
import asyncio, os
from typing import List
from . import get_scraper, list_scrapers  # ensures concrete scrapers are registered
from ..utils.config import load_yaml
from ..utils.logger import get_logger

log = get_logger("scrape-runner")

async def _run_one(name: str, cfg: dict):
    try:
        sconf = cfg.get("scrape", {})
        dconf = cfg.get("data", {})
        scraper_cls = get_scraper(name)
        scraper = scraper_cls(
            user_agent=sconf.get("user_agent", "orphan-studio/0.1"),
            rps=float((sconf.get("rate") or {}).get("requests_per_second", 3)),
            burst=int((sconf.get("rate") or {}).get("burst", 3)),
            timeout=int(sconf.get("request_timeout_sec", 30)),
            total_retries=int(sconf.get("total_retries", 5)),
            max_pages=int((sconf.get("paging") or {}).get("max_pages_per_source", 50)),
            max_docs=int((sconf.get("caps") or {}).get("max_docs_per_source", 200000)),
            shards_dir=dconf.get("shards_dir", "data/shards"),
            shard_max_records=int((sconf.get("shards") or {}).get("max_records_per_shard", 5000)),
        )
        res = await scraper.run()
        log.info(f"[{name}] fetched={res.total_fetched}, shards={res.shards_path}")
    except Exception as e:
        log.error(f"[{name}] failed with error: {e}")

async def _runner(sources: List[str], cfg: dict):
    await asyncio.gather(*[_run_one(s, cfg) for s in sources])

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="conf/scrape.yaml")
    p.add_argument("--sources", nargs="*", default=["dailymed_spls", "openfda_labels"])
    args = p.parse_args()
    cfg = load_yaml(args.config)
    log.info(f"Sources to run: {args.sources}")
    asyncio.run(_runner(args.sources, cfg))

if __name__ == "__main__":
    main()

# src/scrapers/run.py (snippet for clinicaltrials)
from src.scrapers.ctgov import fetch_term
from src.scrapers.http import make_session
from src.utils.logger import get_logger
import yaml, os

log = get_logger("scrape-runner")

def run_ctgov(conf):
    out = conf.get("out", "./data/raw/clinicaltrials")
    page_size = conf.get("page_size", 25)
    max_pages = conf.get("max_pages", 40)
    status_filter = conf.get("status_filter")  # e.g., "Recruiting"
    terms = conf["disease_terms"]              # list of strings

    session = make_session()
    total = 0
    for t in terms:
        saved = fetch_term(
            t, out_dir=out, page_size=page_size,
            max_pages=max_pages, status_filter=status_filter,
            session=session
        )
        total += saved
    log.info(f"[ctgov] DONE. Total studies saved: {total}")
