# Phase 1 Briefing Document

## Intended Use & Target Users
- **Primary audience:** UK-based medical students, foundation doctors, and clinical pharmacists seeking decision support and education.
- **Secondary audience:** NHS clinicians using the platform for rapid literature search, guideline retrieval, and OTC triage support.
- **Use cases:** Symptom triage with WWHAM/AS METHOD protocols, USMLE-style question practice, drug information lookup, CME discovery, and evidence sourcing from PubMed/Europe PMC.

## Technical Architecture Overview
- **Application stack:** FastAPI service (`src/api/main.py`) with background ingestion orchestrated via `run_scrapers.py` and Celery tasks.
- **Model layer:** Custom 7B transformer (`src/models/textlm/`), multimodal encoder for imaging (`src/models/multimodal/vision/`), and safety classifier ensemble (`src/models/clinical/`).
- **Data pipeline:** Scrapers under `src/scrapers/` feed canonicalization (`src/clean/`) and indexing (`src/clean/build_index.py`) stored in `data/` sharded JSONL and TF-IDF artifacts.
- **Protocols & UX:** Conversational interface `src/chat/patient_interface.py` invoking WWHAM/AS METHOD logic with compliance filters from `src/protocols/safety/`.
- **Deployment:** H100-optimized containers defined in `conf/h100_config.yaml`, with Kubernetes manifests under `deploy/` (if applicable) and monitoring hooks integrated via Prometheus exporters.

## Medical Protocols Implemented
- **WWHAM:** Expanded Who, What, How, Action, Medication, Monitoring stages in `src/protocols/wham/protocol.py` with red flag escalation.
- **AS METHOD:** Age, Self-care, Medication, Extra, Time, History, Other, Danger captured through `src/protocols/as_method/protocol.py` and surfaced in triage decisions.
- **Safety layers:** Red flag detector (`src/protocols/safety/red_flags.py`), OTC guardrails (`src/chat/triage.py`), and risk alerts embedded in patient messaging (`src/chat/patient_interface.py`).

## Risk Management Approach
- **Hazard catalog:** Detailed in `docs/phase1/hazard_identification_log.md` covering clinical, technical, and human factors.
- **Risk matrix:** Severity-probability assessment (`docs/phase1/risk_assessment_matrix.md`) guiding mitigation priorities.
- **Controls:** Technical and procedural safeguards documented in `docs/phase1/risk_control_measures.md`, aligned with NHS SAMD guidance.
- **Monitoring:** Logging of triage decisions, anomaly alerts for out-of-policy responses, and manual review workflows.

## Clinical Validation Plan
1. **Pilot cohort:** Recruit 20-30 UK clinicians (GPs, pharmacists, trainees) for structured evaluation sessions.
2. **Test protocol:** Provide standardized cases covering red flags, OTC scenarios, and guideline retrieval tasks.
3. **Data capture:** Collect quantitative metrics (accuracy vs clinician judgment, response time) and qualitative feedback via structured surveys.
4. **Iteration loop:** Log issues in `docs/phase1/user_testing_plan.md`, prioritize fixes, and re-test until acceptance thresholds met.
5. **Approval:** Produce validation summary report for governance committee prior to broader rollout.
