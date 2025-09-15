# src/eval/make_gold_seed.py
import json, random, pathlib
random.seed(42)

OUT = pathlib.Path("data/gold")
OUT.mkdir(parents=True, exist_ok=True)
f = (OUT / "seed_100.jsonl").open("w", encoding="utf-8")

ages = ["25", "35", "45", "60", "68", "75", "pregnant", "child 6y", "child 12y"]
dur = ["1 day", "2 days", "3 days", "1 week", "10 days"]
otc_conditions = [
    ("sore throat", "lozenges, warm fluids; NSAID/acetaminophen if needed; refer if >7d or high fever"),
    ("tension headache", "acetaminophen or ibuprofen; hydration; avoid overuse; refer if severe/neurologic deficits"),
    ("seasonal allergies", "OTC antihistamine; saline spray; refer if wheeze or uncontrolled"),
    ("heartburn after meals", "OTC antacid or short PPI course; avoid triggers; refer if alarm features"),
    ("constipation", "bulk fiber + fluids; short-term laxative; refer if bleeding/weight loss"),
    ("mild diarrhea", "oral rehydration; loperamide if no blood/fever; refer if >48h or blood"),
    ("dry cough", "dextromethorphan lozenges; honey; refer if SOB or chest pain"),
    ("insomnia (short-term)", "sleep hygiene; short OTC antihistamine 2–3 nights max; refer if persistent"),
]

danger_cases = [
    ("crushing chest pain and sweating", "Emergency"),
    ("severe headache with stiff neck", "Emergency"),
    ("shortness of breath at rest", "Doctor"),
    ("black tarry stools", "Doctor"),
    ("blood in vomit", "Doctor"),
]

contra_cases = [
    ("pregnant with back pain", "avoid NSAIDs; use acetaminophen; refer if severe"),
    ("child 6y with fever", "avoid aspirin; use acetaminophen weight-based; refer if persistent"),
    ("warfarin user with knee pain", "avoid NSAIDs; topical analgesic or acetaminophen; consult clinician"),
]

interactions = [
    ("taking sertraline wants cough suppressant", "avoid dextromethorphan due to serotonin risk; prefer non-dxm"),
    ("on lisinopril considering potassium supplement", "risk of hyperkalemia; avoid without clinician"),
]

def ex(obj): f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# 40 OTC counselling
for cond, exp in random.sample(otc_conditions, k=len(otc_conditions)):
    for _ in range(5):  # 8 * 5 = 40
        age = random.choice(ages[:-1])  # avoid child for some
        ex({
            "input": f"{age} with {cond} for {random.choice(dur)}, no major illnesses.",
            "age": age,
            "tags": ["OTC", cond.replace(" ", "_")],
            "expected": exp,
            "refer_level": "OTC",
            "notes": "standard OTC"
        })

# 30 triage danger/safe (we'll make 20 danger, 10 safe-but-watch)
for text, level in danger_cases:
    for _ in range(4):  # 5 * 4 = 20
        age = random.choice(ages)
        ex({
            "input": f"{age} with {text}",
            "age": age,
            "tags": ["triage", "danger"],
            "expected": "Immediate referral" if level == "Emergency" else "Non-urgent clinician evaluation",
            "refer_level": level,
            "notes": "danger sign template"
        })
# 10 safe-but-watch
for _ in range(10):
    age = random.choice(ages)
    ex({
        "input": f"{age} with mild runny nose for {random.choice(dur)}, no fever, no SOB.",
        "age": age, "tags": ["triage", "safe"],
        "expected": "Self-care; refer if symptoms persist or worsen.",
        "refer_level": "OTC", "notes": "no red flags"
    })

# 20 contraindication / suitability
for text, exp in contra_cases:
    for _ in range(6 if len(contra_cases) == 3 else 5):  # ~18
        age = text.split()[0]
        ex({
            "input": text,
            "age": age, "tags": ["contraindication"],
            "expected": exp,
            "refer_level": "OTC",
            "notes": "otc with caveat"
        })
# pad to reach 90 (if needed)
while sum(1 for _ in open(OUT / "seed_100.jsonl", "r", encoding="utf-8")) < 90:
    ex({
        "input": "35 with minor muscle ache after gym, 2 days, no meds.",
        "age": "35", "tags": ["OTC", "analgesia"],
        "expected": "OTC acetaminophen/ibuprofen; rest; refer if swelling/fever.",
        "refer_level": "OTC", "notes": "filler"
    })

# 10 interactions
for text, exp in interactions:
    for _ in range(5):  # 2 * 5 = 10
        ex({
            "input": f"Patient {text}.",
            "age": random.choice(ages),
            "tags": ["interaction"],
            "expected": exp,
            "refer_level": "OTC",
            "notes": "interaction awareness"
        })

f.close()
print(f"Wrote {OUT/'seed_100.jsonl'}")
