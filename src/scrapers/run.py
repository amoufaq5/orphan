from __future__ import annotations
import asyncio, os
from typing import List
from . import get_scraper, list_scrapers  # ensures concrete scrapers are registered
from ..utils.config import load_yaml
from ..utils.logger import get_logger

log = get_logger("scrape-runner")

async def _run_one(name: str, cfg: dict):
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

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="conf/scrape.yaml")
    p.add_argument("--sources", nargs="*", default=["dailymed_spls", "openfda_labels"])
    args = p.parse_args()

    cfg = load_yaml(args.config)
    log.info(f"Sources to run: {args.sources}")
    asyncio.run(asyncio.gather(*[_run_one(s, cfg) for s in args.sources]))

if __name__ == "__main__":
    main()
