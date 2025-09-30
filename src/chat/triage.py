# src/chat/triage.py
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

DANGER_SIGNS = {
    "adult": [
        "severe chest pain", "shortness of breath at rest", "hemoptysis",
        "sudden severe headache", "stiff neck and fever", "confusion",
        "uncontrolled bleeding", "anaphylaxis", "new weakness on one side",
        "black tarry stools", "blood in vomit", "severe abdominal pain",
        "pregnant with abdominal pain/bleeding", "suicidal thoughts"
    ],
    "child": [
        "lethargy", "grunting respiration", "cyanosis",
        "bulging fontanelle", "stiff neck", "non-blanching rash",
        "severe dehydration", "persistent high fever >39.5C"
    ],
    "pregnancy": [
        "severe abdominal pain", "vaginal bleeding", "severe headache with visual change",
        "reduced fetal movements", "hypertension with swelling"
    ]
}

OTC_BLACKLIST = [
    # conditions where *any* drug advice should redirect
    "warfarin use with nsaid", "isotretinoin pregnancy", "infant <3 months fever",
    "chest pain", "suspected stroke", "meningitis", "acute abdomen"
]

@dataclass
class ASMETHOD:
    age: str = ""              # Age/appearance
    self_or_other: str = ""    # Self / someone else
    meds: List[str] = None     # Current medication
    extra_meds: List[str] = None  # Other meds/supplements
    time_course: str = ""      # Duration/onset
    history: List[str] = None  # Conditions/allergies/pregnancy
    other_symptoms: List[str] = None
    danger_symptoms: List[str] = None

@dataclass
class WWHAM:
    who: str = ""
    what_symptoms: str = ""
    how_long: str = ""
    action_taken: str = ""
    medication_used: str = ""
    monitoring: str = ""

@dataclass
class TriageResult:
    safe: bool                 # True → proceed to OTC advice/model; False → refer
    level: str                 # 'OTC' | 'Doctor' | 'Emergency'
    reasons: List[str]         # why the decision
    structured: Dict = None    # echo inputs

def _normalize(s: str) -> str:
    return (s or "").lower().strip()

def _flag_danger(age_bucket: str, text_blobs: List[str]) -> List[str]:
    hits = []
    for t in text_blobs:
        T = _normalize(t)
        for sign in DANGER_SIGNS.get(age_bucket, []):
            if sign in T:
                hits.append(sign)
    return sorted(set(hits))

def _age_bucket(age_text: str) -> str:
    t = _normalize(age_text)
    # crude but robust
    for token in ["pregnan", "pregnant", "gestation"]:
        if token in t:
            return "pregnancy"
    digits = "".join([c for c in t if c.isdigit()])
    if digits:
        v = int(digits)
        if v < 18: return "child"
        else: return "adult"
    # fallback
    return "adult"

def triage(asmethod: ASMETHOD, wwham: WWHAM) -> TriageResult:
    reasons = []
    age_bucket = _age_bucket(asmethod.age)

    # hard blacklist
    joined = " | ".join(filter(None, [
        wwham.what_symptoms, wwham.action_taken, wwham.medication_used,
        " ".join(asmethod.other_symptoms or []), " ".join(asmethod.danger_symptoms or []),
        " ".join(asmethod.history or []), wwham.monitoring
    ])).lower()

    if any(b in joined for b in OTC_BLACKLIST):
        reasons.append("High-risk presentation incompatible with OTC management.")
        return TriageResult(False, "Doctor", reasons, structured={"ASMETHOD": asdict(asmethod), "WWHAM": asdict(wwham)})

    # explicit danger symptoms
    hits = _flag_danger(age_bucket, [
        wwham.what_symptoms, " ".join(asmethod.other_symptoms or []),
        " ".join(asmethod.danger_symptoms or [])
    ])
    if hits:
        level = "Emergency" if any(k in hits for k in ["severe chest pain","suicidal thoughts","stroke","meningitis"]) else "Doctor"
        reasons.append(f"Danger symptoms detected: {', '.join(hits)}")
        return TriageResult(False, level, reasons, structured={"ASMETHOD": asdict(asmethod), "WWHAM": asdict(wwham)})

    # basic suitability gates
    # pregnancy/children → restrict many OTC classes
    if age_bucket in ["child", "pregnancy"]:
        reasons.append(f"Special population detected: {age_bucket} — restrict certain OTCs; prefer doctor if uncertainty.")

    return TriageResult(True, "OTC", reasons, structured={"ASMETHOD": asdict(asmethod), "WWHAM": asdict(wwham)})

# Helper to build the system prompt used if safe == True
def otc_guardrail_prompt(domain_hint: str = "") -> str:
    return (
        "You are a cautious clinical assistant. "
        "Before giving any recommendation, verify the case has passed triage. "
        "Policy:\n"
        "1) ONLY provide OTC/self-care advice. If a prescription drug or uncertain safety is involved, say: "
        "\"This needs a clinician. Here's why: …\" and stop.\n"
        "2) Always include simple steps, dose ranges *only if* standard OTC and patient age allows, "
        "3) Provide 2–3 evidence snippets from retrieved docs with citations, "
        "4) Add short red-flag list for when to seek urgent care, "
        "5) Keep language clear, kind, and non-alarming.\n"
        + (f"Domain hint: {domain_hint}\n" if domain_hint else "")
    )
