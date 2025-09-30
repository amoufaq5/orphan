# src/chat/patient_cli.py
"""
Interactive CLI to talk 'as a patient'.
Collects ASMETHOD/WWHAM answers, runs local inference endpoint or direct function.
Set API_URL in conf/app.yaml to call the FastAPI server; otherwise imports run_inference.
"""

from __future__ import annotations
import json
import requests
from typing import Dict, Any, Tuple

from src.chat.triage import ASMETHOD, WWHAM, triage
from src.utils.config import load_yaml

CFG = load_yaml("conf/app.yaml")
API_URL = (CFG.get("api") or {}).get("url")  # e.g., "http://127.0.0.1:8000/infer"

def ask(prompt: str) -> str:
    return input(prompt).strip()

def collect_asmethod() -> Tuple[ASMETHOD, WWHAM]:
    print("\nI’ll ask a few quick safety questions (it takes ~30 seconds):")
    age = ask("Age (or type 'pregnant' if applicable): ")
    who = ask("Who is this for (you / someone else)?: ")
    how_long = ask("How long has this been going on?: ")
    what = ask("Main symptoms (comma-separated, short): ")
    meds = ask("Current medicines (comma-separated; blank if none): ")
    other = ask("Any illnesses/allergies/conditions? (comma-separated; blank if none): ")
    danger = ask("Any worrying signs (e.g., severe chest pain, bleeding, confusion)? ")

    asmethod = ASMETHOD(
        age=age,
        self_or_other=who,
        meds=[m.strip() for m in meds.split(",") if m.strip()],
        time_course=how_long,
        history=[x.strip() for x in other.split(",") if x.strip()],
        other_symptoms=[s.strip() for s in what.split(",") if s.strip()],
        danger_symptoms=[danger] if danger else []
    )
    action = ask("What advice or treatment have you been given so far (if any)?: ")
    monitoring = ask("What advice or monitoring have you been given so far (if any)?: ")

    wwham = WWHAM(
        who=who,
        what_symptoms=what,
        how_long=how_long,
        action_taken=action,
        medication_used=meds,
        monitoring=monitoring
    )
    return asmethod, wwham

def call_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not API_URL:
        from src.api.infer import run_inference  # lazy import
        return run_inference(payload)
    r = requests.post(API_URL, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def main():
    print("Patient CLI — Orphan\nType 'quit' to exit.")
    persona = (CFG.get("default_persona") or "patient")

    while True:
        text = ask("\nTell me what’s going on: ")
        if text.lower() in {"q", "quit", "exit"}:
            print("Bye! Take care.")
            break

        asm, wwm = collect_asmethod()
        # quick local triage echo (API will triage again too)
        tri = triage(asm, wwm)
        if not tri.safe:
            print(f"\n⚠️  Please see a clinician ({tri.level}). Reason: {', '.join(tri.reasons)}")
            continue

        payload = {
            "text": text,
            "asmethod": asm.__dict__,
            "wwham": wwm.__dict__,
            "persona": persona
        }
        try:
            res = call_api(payload)
        except Exception as e:
            print(f"\nError contacting API: {e}")
            continue

        print("\n— Response —")
        print(res.get("answer", "(no answer)"))
        tri_info = res.get("triage", {})
        if tri_info:
            print(f"\n[triage: {tri_info.get('level','OTC')}]")
        cits = res.get("citations", [])
        if cits:
            print("\nSources:")
            for i, c in enumerate(cits, 1):
                print(f" {i}. {c}")

if __name__ == "__main__":
    main()
