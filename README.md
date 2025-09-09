
# orphan-studio (private)

> Medical LLM platform scaffold — text-first pipeline. Imaging to follow after first batch.

## What’s included
- Config-driven pipeline (`conf/*.yaml`)
- Logging, IO, PII scrub stubs, hashing utils
- Production-ready **scraper base** (sessions, retries, polite rate limits, paging, sharded outputs, provenance)
- Cleaning/normalization stubs
- Ontology adapters (ICD-10, ATC, RxNorm, MeSH) stubs
- RAG chunk/index placeholders
- FastAPI skeleton
- Windows-friendly scripts

## Quick start (local)
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -U pip && pip install -r requirements.txt

# Prepare data folders (already in repo)
mkdir data\raw, data\cleaned, data\canonical, data\shards

# Configure
# conf/app.yaml, conf/scrape.yaml etc.

# Run (no external scrapers yet)
.\scripts\run_data.ps1
.\scripts\run_api.ps1

