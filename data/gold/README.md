Gold Set (v0)
- 60 triage (danger vs safe)
- 80 OTC counselling
- 40 contraindication/dose suitability
- 40 interaction checks
Format: JSONL with fields:
{ "input": "<patient text>", "age": "35", "tags": ["OTC","cough"],
  "expected": "<short target answer or key bullets>",
  "refer_level": "OTC|Doctor|Emergency",
  "notes": "why" }
