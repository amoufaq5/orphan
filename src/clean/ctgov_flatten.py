# src/clean/ctgov_flatten.py
import os, json, csv, argparse, glob
from typing import Any, Dict, List, Optional

def get_in(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def to_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

# Try multiple possible v2 paths for robustness
CANDIDATE_PATHS = {
    "nct_id": [
        ["protocolSection","identificationModule","nctId"],
        ["identificationModule","nctId"],
        ["nctId"],
    ],
    "brief_title": [
        ["protocolSection","identificationModule","briefTitle"],
        ["identificationModule","briefTitle"],
        ["briefTitle"],
    ],
    "official_title": [
        ["protocolSection","identificationModule","officialTitle"],
        ["officialTitle"],
    ],
    "overall_status": [
        ["protocolSection","statusModule","overallStatus"],
        ["statusModule","overallStatus"],
        ["overallStatus"],
    ],
    "conditions": [
        ["protocolSection","conditionsModule","conditions"],
        ["conditionsModule","conditions"],
        ["conditions"],
    ],
    "interventions": [
        ["protocolSection","armsInterventionsModule","interventions"],
        ["armsInterventionsModule","interventions"],
        ["interventions"],
    ],
    "study_type": [
        ["protocolSection","designModule","studyType"],
        ["designModule","studyType"],
        ["studyType"],
    ],
    "phases": [
        ["protocolSection","designModule","phases"],
        ["designModule","phases"],
        ["phases"],
    ],
    "primary_outcomes": [
        ["protocolSection","outcomesModule","primaryOutcomes"],
        ["outcomesModule","primaryOutcomes"],
        ["primaryOutcomes"],
    ],
    "first_posted_date": [
        ["protocolSection","statusModule","studyFirstPostDateStruct","date"],
        ["statusModule","studyFirstPostDateStruct","date"],
        ["studyFirstPostDate"],
    ],
    "last_update_posted_date": [
        ["protocolSection","statusModule","lastUpdatePostDateStruct","date"],
        ["statusModule","lastUpdatePostDateStruct","date"],
        ["lastUpdatePostDate"],
    ],
}

def extract_field(study: Dict[str, Any], key: str):
    for path in CANDIDATE_PATHS.get(key, []):
        val = get_in(study, path, None)
        if val is not None:
            return val
    return None

def flatten_study(study: Dict[str, Any]) -> Dict[str, Any]:
    nct_id = extract_field(study, "nct_id")
    brief_title = extract_field(study, "brief_title")
    official_title = extract_field(study, "official_title")
    status = extract_field(study, "overall_status")

    conditions = to_list(extract_field(study, "conditions"))
    # Interventions can be complex dicts in v2; grab names if present
    raw_interv = to_list(extract_field(study, "interventions"))
    interventions = []
    for it in raw_interv:
        if isinstance(it, dict):
            name = it.get("name") or it.get("interventionName") or it.get("label")
            if name: interventions.append(name)
        elif isinstance(it, str):
            interventions.append(it)

    study_type = extract_field(study, "study_type")
    phases = to_list(extract_field(study, "phases"))

    raw_pos = to_list(extract_field(study, "primary_outcomes"))
    primary_outcomes = []
    for po in raw_pos:
        if isinstance(po, dict):
            m = po.get("measure") or po.get("description") or po.get("name")
            if m: primary_outcomes.append(m)
        elif isinstance(po, str):
            primary_outcomes.append(po)

    first_posted = extract_field(study, "first_posted_date")
    last_update_posted = extract_field(study, "last_update_posted_date")

    return {
        "nct_id": nct_id,
        "brief_title": brief_title,
        "official_title": official_title,
        "overall_status": status,
        "conditions": "; ".join(conditions) if conditions else None,
        "interventions": "; ".join(interventions) if interventions else None,
        "study_type": study_type,
        "phases": "; ".join(phases) if phases else None,
        "primary_outcomes": "; ".join(primary_outcomes) if primary_outcomes else None,
        "first_posted_date": first_posted,
        "last_update_posted_date": last_update_posted,
    }

def main():
    ap = argparse.ArgumentParser(description="Flatten ClinicalTrials v2 shards to JSONL/CSV")
    ap.add_argument("--in_dir", default="./data/raw/clinicaltrials", help="Directory with *.jsonl shards")
    ap.add_argument("--out_jsonl", default="./data/cleaned/clinicaltrials_flat.jsonl", help="Output JSONL")
    ap.add_argument("--out_csv", default="./data/cleaned/clinicaltrials_flat.csv", help="Output CSV")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    shards = sorted(glob.glob(os.path.join(args.in_dir, "*.jsonl")))
    if not shards:
        print(f"No shards found in {args.in_dir}")
        return

    seen = set()
    rows: List[Dict[str, Any]] = []

    for path in shards:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                study = json.loads(line)
                flat = flatten_study(study)
                key = flat.get("nct_id") or json.dumps(flat, sort_keys=True)  # fallback
                if key in seen:
                    continue
                seen.add(key)
                rows.append(flat)

    # Write JSONL
    with open(args.out_jsonl, "w", encoding="utf-8") as fj:
        for r in rows:
            fj.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write CSV
    fieldnames = [
        "nct_id","brief_title","official_title","overall_status",
        "conditions","interventions","study_type","phases",
        "primary_outcomes","first_posted_date","last_update_posted_date"
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as fc:
        w = csv.DictWriter(fc, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    print(f"OK: wrote {len(rows)} studies → {args.out_jsonl} and {args.out_csv}")

if __name__ == "__main__":
    main()
