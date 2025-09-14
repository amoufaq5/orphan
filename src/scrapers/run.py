# in src/scrapers/run.py (add near imports)
from src.scrapers.pubmed import fetch_term as pubmed_fetch_term
# ... existing code ...

def _load_terms(src_conf, overrides):
    terms = src_conf.get("disease_terms", []) or []
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

def run_pubmed(conf: Dict[str, Any], overrides: argparse.Namespace) -> int:
    src_conf = conf.get("pubmed", {})
    out_dir = src_conf.get("out", "./data/raw/pubmed")
    retmax = src_conf.get("retmax", 1000)
    max_docs = src_conf.get("max_docs_per_term", 5000)
    batch = src_conf.get("efetch_batch_size", 200)
    tmpl = src_conf.get("search_query_template", '("%TERM%"[Title/Abstract]) OR ("%TERM%"[MeSH Terms])')
    since_year = src_conf.get("since_year", "")
    until_year = src_conf.get("until_year", "")
    workers = max(1, overrides.workers)

    terms = _load_terms(src_conf, overrides)
    if not terms:
        log.info("[pubmed] No terms found. Check config or use --only.")
        log.info(f"[pubmed] fetched=0, shards={out_dir}")
        return 0

    log.info(f"[pubmed] queries={len(terms)} retmax={retmax} max_docs/term={max_docs} batch={batch} workers={workers}")

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
                    term=term, out_dir=out_dir,
                    retmax=retmax, max_docs_per_term=max_docs, efetch_batch_size=batch,
                    search_query_template=tmpl, since_year=since_year, until_year=until_year,
                    session=session
                )
            except Exception as e:
                log.error(f"[pubmed] Term '{term}' failed: {e}")
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        def _do(t: str) -> int:
            return pubmed_fetch_term(
                term=t, out_dir=out_dir, retmax=retmax, max_docs_per_term=max_docs,
                efetch_batch_size=batch, search_query_template=tmpl,
                since_year=since_year, until_year=until_year, session=None
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

# ... in main() dispatch table:
runners = {
    "clinicaltrials": run_clinicaltrials,
    "pubmed": run_pubmed,
}
