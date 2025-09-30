# Risk Control Measures

## Technical Safeguards
- **[Red flag detection]** `src/protocols/safety/red_flags.py` triggers emergency messaging and halts OTC advice when high-risk symptoms are detected.
- **[Guardrails & policies]** `src/chat/triage.py` and `src/chat/patient_interface.py` enforce WWHAM/AS METHOD flows, refuse unsafe recommendations, and provide safety netting prompts.
- **[Model moderation]** Safety classifiers in `src/models/clinical/` filter hallucinations; logs stored for audit in accordance with NHS SAMD expectations.
- **[Data freshness checks]** Scraper orchestration (`run_scrapers.py`, `src/scrapers/run.py`) retries on failure and alerts operators when critical feeds (e.g., CDC, NICE-derived content) fail.
- **[PII scrubbing]** `src/utils/pii.py` removes personal identifiers before logging or analytics.

## Procedural Controls
- **[Clinical oversight]** Weekly governance meeting with clinical leads to review triage outputs and high-risk incidents logged in `docs/phase1/hazard_identification_log.md`.
- **[Content verification]** Pharmacist-led review of drug content sourced from DailyMed/OpenFDA/DrugBank, with sign-off recorded in `docs/phase1/review_logs.md`.
- **[Emergency disclaimers]** UI copy in `gradio_app.py` and patient handouts reminds users to dial 999/911 for emergencies.
- **[Pilot supervision]** On-call clinician available during user testing to intervene if unsafe guidance observed.
- **[Documentation audits]** Quarterly review of technical IP and risk documents to ensure accuracy.

## Information for Safety
- **Patient handouts:** Provide plain-language instructions on when to seek urgent or routine care; stored in `docs/patient_materials/`.
- **User manual:** Phase 1 operations handbook (to be finalized) detailing system capabilities, limitations, and escalation pathways.
- **FAQ & support:** In-app help links to NHS resources and contact details for support team.
- **Logging & feedback:** Capture user feedback through interface prompts; issues triaged via ticketing system with SLAs defined in `docs/operations/sla.md`.
