# User Testing Plan (Healthcare Professionals)

## Objectives
- Validate clinical accuracy and safety of WWHAM/AS METHOD triage flows.
- Measure usability and satisfaction across medical education and literature search features.
- Capture actionable feedback for Phase 1 production readiness.

## Participant Criteria
- 20-30 UK-based clinicians (GPs, clinical pharmacists, junior doctors).
- Mix of digital health experience levels.
- Signed participation agreement and confidentiality acknowledgment.

## Test Structure
1. **Orientation (30 min)**
   - Overview of platform capabilities (`docs/phase1/briefing_document.md`).
   - Review safety protocols and escalation contacts.
2. **Guided Scenarios (60 min)**
   - Red-flag triage cases (e.g., chest pain, sepsis symptoms).
   - OTC-appropriate cases (e.g., self-limiting conditions).
   - Drug information lookup via new scrapers (DrugBank, RxNorm, DailyMed).
3. **Self-directed Tasks (45 min)**
   - Literature search with Semantic Scholar/Europe PMC integration.
   - CME discovery using `cme_list` feeds.
4. **Feedback & Debrief (30 min)**
   - Structured survey (Likert scale) and open comments.
   - Capture improvement requests and safety concerns.

## Metrics
- Clinical concordance (% agreement with clinician judgment).
- Time-to-decision per scenario.
- SUS (System Usability Scale) score.
- Net Promoter Score (likelihood to recommend for peers).

## Data Collection
- Session recordings (with consent) stored per GDPR guidelines.
- Survey responses logged in secure analytics workspace.
- Issue tracker entries in `docs/phase1/testing_findings.md` with severity tags.

## Follow-up Actions
- Weekly review of findings with Applied AI and Clinical leads.
- Prioritize high-severity issues for immediate fix before release.
- Publish validation summary in `docs/phase1/clinical_validation_report.md`.
