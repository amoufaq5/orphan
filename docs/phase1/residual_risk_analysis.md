# Residual Risk Analysis

| Risk ID | Description | Controls Applied | Residual Risk | Acceptance Rationale | Owner |
| --- | --- | --- | --- | --- | --- |
| H-001 | Incorrect red-flag triage | Red flag detector, clinician escalation workflow | Low-Medium | Residual risk acceptable with clinician oversight and emergency messaging | Clinical Lead |
| H-002 | Medication guidance error | Pharmacist review, multi-source drug data, guardrails | Medium | Human review for drug cards ensures accuracy before release | Pharmacy Liaison |
| H-003 | Outdated guidelines | Scheduled scraper runs, freshness monitoring | Low | Automation catches stale feeds; manual spot checks weekly | Data Engineering |
| H-004 | Model hallucination | Safety classifier, refusal templates, logging | Medium | Some residual risk remains; mitigated by manual review during pilot | Applied AI |
| H-005 | Data ingestion failure | Retry logic, alerting, manual reruns | Low | Operations on-call handles failures promptly | Data Operations |
| H-006 | Corrupted index | Regular rebuild scripts, checksum validation | Low | Index rebuild automated with validation steps | Data Operations |
| H-007 | User misinterpretation | Safety messaging, user education, clinician support | Medium | Residual risk acceptable with clear disclaimers and clinician hotline | Clinical Ops |
| H-008 | Accessibility gaps | UX testing, inclusive design checklist | Low | Continuous UX improvements planned during pilot | Product Design |
| H-009 | PII disclosure | PII scrubbing, access controls, logging policies | Low | Strict controls reduce exposure to negligible levels | Security Officer |
| H-010 | Lack of clinician oversight | Pilot governance, weekly review meetings | Low | Structured pilot with assigned oversight ensures rapid intervention | Program Manager |

Residual risks reviewed on 2025-09-30. Next review scheduled prior to Phase 1 go/no-go meeting.
