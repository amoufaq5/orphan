# Hazard Identification Log

| ID | Hazard Type | Description | Potential Harm | Source References |
| --- | --- | --- | --- | --- |
| H-001 | Clinical | Incorrect triage advice for red-flag symptoms | Delay in emergency care leading to morbidity/mortality | `src/chat/triage.py`, `src/protocols/safety/red_flags.py` |
| H-002 | Clinical | Inaccurate medication guidance or contraindication omission | Adverse drug events, allergic reactions | `src/protocols/wham/protocol.py`, `src/scrapers/dailymed.py` |
| H-003 | Clinical | Outdated guideline information from scraped sources | Suboptimal treatment decisions | `src/scrapers/extended_sources.py`, `scrape_config.yaml` |
| H-004 | Technical | Model hallucination or unsafe responses | Misinformation, unsafe self-care instructions | `src/models/textlm/`, `src/eval/eval_gold.py` |
| H-005 | Technical | Data ingestion failure leading to partial datasets | Missing clinical evidence, biased recommendations | `run_scrapers.py`, `src/scrapers/run.py` |
| H-006 | Technical | Index corruption or stale embeddings | Irrelevant search results, misinformed decisions | `src/clean/build_index.py`, `data/` indices |
| H-007 | User | Misinterpretation of AI guidance by patients | Unsafe self-treatment, delayed clinician contact | `src/chat/patient_interface.py`, `gradio_app.py` |
| H-008 | User | Accessibility gaps or confusing UI | Improper data entry, incomplete protocols | `gradio_app.py`, UX components |
| H-009 | Compliance | Unauthorized data disclosure or logging of PII | Privacy breach, GDPR violations | `src/utils/pII.py`, logging configs |
| H-010 | Operational | Insufficient clinician oversight during pilot | Undetected errors, reputational risk | `docs/phase1/user_testing_plan.md` |

## Notes
- Hazard register to be reviewed weekly during Phase 1 execution.
- Mitigation and residual risk documented in companion risk control files.
- Updates logged in `docs/phase1/risk_changes.md` for traceability.
