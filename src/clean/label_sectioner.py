from __future__ import annotations
from typing import Dict, Any, List, Tuple
import re

# Common section keys in openFDA JSON
OPENFDA_KEYS = {
    "indications_and_usage": "indications",
    "dosage_and_administration": "dosage",
    "contraindications": "contraindications",
    "warnings": "warnings",
    "warnings_and_cautions": "warnings",
    "precautions": "precautions",
    "adverse_reactions": "adverse_reactions",
    "drug_interactions": "interactions",
    "use_in_specific_populations": "use_in_specific_populations",
    "clinical_pharmacology": "clinical_pharmacology",
    "clinical_studies": "clinical_studies",
    "information_for_patients": "patient_information",
    "patient_counseling_information": "patient_information",
    "storage_and_handling": "storage_and_handling",
    "package_label_principal_display_panel": "principal_display_panel",
}

# DailyMed often has sections array with .text/.content + sometimes titles
def build_sections_from_dailymed(raw_item: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    sections = raw_item.get("sections") or raw_item.get("splSections") or []
    for s in sections:
        title = ""
        for k in ("title", "sectionTitle", "heading"):
            if isinstance(s.get(k), str) and s[k].strip():
                title = s[k].strip().lower()
                break
        text = s.get("text") or s.get("content") or ""
        if not isinstance(text, str) or not text.strip():
            continue
        key = None
        # light heuristic mapping
        if "indication" in title:
            key = "indications"
        elif "dosage" in title or "administration" in title:
            key = "dosage"
        elif "contraindication" in title:
            key = "contraindications"
        elif "warning" in title:
            key = "warnings"
        elif "precaution" in title:
            key = "precautions"
        elif "adverse" in title or "reaction" in title:
            key = "adverse_reactions"
        elif "interaction" in title:
            key = "interactions"
        elif "storage" in title or "handling" in title:
            key = "storage_and_handling"
        elif "clinical pharmacology" in title:
            key = "clinical_pharmacology"
        elif "clinical studies" in title:
            key = "clinical_studies"
        elif "patient" in title or "counsel" in title:
            key = "patient_information"

        if key is None:
            key = "other"
        out.setdefault(key, "")
        out[key] += (("\n\n" if out[key] else "") + text.strip())
    return out

def build_sections_from_openfda(raw_item: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, norm in OPENFDA_KEYS.items():
        val = raw_item.get(k)
        if isinstance(val, list):
            text = "\n\n".join([x for x in val if isinstance(x, str) and x.strip()]).strip()
        elif isinstance(val, str):
            text = val.strip()
        else:
            text = ""
        if text:
            out[norm] = text
    return out

def split_drug_label(meta: Dict[str, Any]) -> Dict[str, str]:
    """
    meta["raw"] contains the original source item for both DailyMed and openFDA.
    """
    raw = meta.get("raw") or {}
    if meta.get("source") == "dailymed" or (isinstance(raw, dict) and (raw.get("sections") or raw.get("splSections"))):
        return build_sections_from_dailymed(raw)
    # openfda
    return build_sections_from_openfda(raw)
