import json, pathlib
from src.chat.triage import ASMETHOD, WWHAM, triage

p = pathlib.Path("data/gold/examples.jsonl")
ok, total = 0, 0
for line in p.read_text(encoding="utf-8").splitlines():
    ex = json.loads(line)
    asm = ASMETHOD(age=str(ex.get("age","")), other_symptoms=[ex["input"]], meds=[], extra_meds=[], history=[], time_course="", danger_symptoms=[])
    tri = triage(asm, WWHAM())
    pred = "OTC" if tri.safe else tri.level
    ok += int(pred == ex["refer_level"])
    total += 1
print(f"Triage accuracy: {ok}/{total} = {ok/total:.1%}")
