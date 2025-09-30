# Phase 1 Launch Roadmap & Checklist

## Overview
Phase 1 focuses on releasing the NHS-oriented medical education and guidance platform with the following launch features:

- Medical student education platform
- USMLE practice questions
- Medical literature search
- General health information
- Drug information database

## Roadmap Timeline

| Milestone | Description | Owners | Target Date |
| --- | --- | --- | --- |
| M1 | Finalize data ingestion across legacy + extended scrapers | Data Engineering | Week 1 |
| M2 | Complete cleaning, canonicalization, and indexing refresh | Data Engineering | Week 2 |
| M3 | Integrate WWHAM/AS METHOD flows into chat & API stack | Applied AI | Week 2 |
| M4 | Deliver technical IP, risk governance, and compliance docs | Governance | Week 2 |
| M5 | Run closed beta with 20-30 UK clinicians, collect feedback | Clinical Ops | Week 3 |
| M6 | Production readiness review & greenlight for Phase 1 launch | Program Leads | Week 4 |

## Execution Workstreams

- **[Data & Ingestion]** Validate new scrapers in `src/scrapers/extended_sources.py`, schedule nightly runs via `run_scrapers.py`, and backfill historical corpora.
- **[Cleaning & Indexing]** Execute `python -m src.clean.make_corpus` and rebuild retrieval assets using `src/clean/build_index.py` after the new ingestion cycle.
- **[Application Layer]** Ensure `src/chat/triage.py`, `src/chat/patient_interface.py`, and API endpoints in `src/api/` surface updated WWHAM outputs.
- **[Safety & Compliance]** Implement recommendations from `docs/phase1/risk_control_measures.md` and update patient guidance content in `gradio_app.py`.
- **[Clinical Pilot]** Coordinate clinician onboarding, distribute briefing from `docs/phase1/briefing_document.md`, and capture feedback per `docs/phase1/user_testing_plan.md`.

## Launch Checklist

- **[ ] Data ingestion run completed** (`run_scrapers.py` with refreshed `scrape_config.yaml`).
- **[ ] Cleaned corpus regenerated** (`src/clean/to_canonical.py`, `src/clean/normalize.py`).
- **[ ] Retrieval index rebuilt** (`src/clean/build_index.py`).
- **[ ] Chat flows validated** (WHO/WHAT/HOW/ACTION/MEDICATION/MONITORING prompts).
- **[ ] API regression tests executed** (refer to automation scripts in `scripts/`).
- **[ ] Risk documentation approved** (hazards, matrix, controls, residual risk).
- **[ ] Clinical pilot scheduled** (participant roster + kickoff briefing).
- **[ ] Launch go/no-go review held** (minutes stored in `docs/release/`).

## Dependencies & Notes

- Use rate limits defined in `scrape_config.yaml` to avoid API throttling.
- Ensure environment variables for `PUBMED_API_KEY`, `OPENFDA_API_KEY`, and `RXNAV` endpoints are configured in deployment pipelines.
- Coordinate with infrastructure for H100 training slots described in `conf/h100_config.yaml`.
