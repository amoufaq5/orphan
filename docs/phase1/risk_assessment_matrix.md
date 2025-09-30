# Risk Assessment Matrix

| ID | Hazard | Severity (1-5) | Probability (1-5) | Risk Priority Number (S×P) | Rationale |
| --- | --- | --- | --- | --- | --- |
| H-001 | Incorrect red-flag triage | 5 | 2 | 10 | Life-threatening if missed; mitigated by red flag detector and clinician review |
| H-002 | Medication guidance error | 4 | 3 | 12 | Potential serious adverse events; multiple data sources but human validation required |
| H-003 | Outdated guideline content | 3 | 2 | 6 | Causes suboptimal care; nightly scraper schedule reduces probability |
| H-004 | Model hallucination | 4 | 2 | 8 | Safety classifier + guardrails reduce probability; severity moderate-high |
| H-005 | Data ingestion failure | 3 | 3 | 9 | Leads to missing evidence; monitoring + retry logic reduce probability |
| H-006 | Corrupted index | 3 | 2 | 6 | Retrieval degraded but not critical; regular rebuilds minimize impact |
| H-007 | User misinterpretation | 4 | 3 | 12 | Clear disclaimers and monitoring prompts to seek professional care |
| H-008 | Accessibility gaps | 2 | 2 | 4 | Limits usability; addressed via UX testing and iterative design |
| H-009 | PII disclosure | 5 | 1 | 5 | Strict logging scrubbing + access controls keep probability low |
| H-010 | Lack of clinician oversight | 4 | 2 | 8 | Pilot governance ensures escalation pathways |

**Severity scale:** 1 (Negligible) → 5 (Catastrophic)

**Probability scale:** 1 (Rare) → 5 (Frequent)

Review the matrix during weekly risk meetings; update ratings after mitigation effectiveness is validated.
