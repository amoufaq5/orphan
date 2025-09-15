# src/api/infer.py
"""
Guardrailed inference entry: runs ASMETHOD/WWHAM triage before generation.
If triage says Doctor/Emergency -> NO model call; we return a referral.
Otherwise we prepend an OTC-only system prompt and generate.

Expected request payload (dict-like):
{
  "text": "patient free text message",
  "asmethod": {...},   # optional dict
  "wwham": {...},      # optional dict
  "persona": "patient" # optional; falls back to app.yaml personas.default
}
"""
from __future__ import annotations
from typing import Dict, Any, List
from dataclasses import asdict

from src.chat.triage import ASMETHOD, WWHAM, triage, otc_guardrail_prompt
from src.utils.config import load_yaml
from src.utils.logger import get_logger

log = get_logger("infer")

# --- Light wrappers you likely already have elsewhere ---
class ModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tok = tokenizer

    def generate(self, prompt: str, max_new_tokens: int = 384) -> str:
        # Replace with your actual HF generation call
        return self.model.generate_with_tokenizer(self.tok, prompt, max_new_tokens=max_new_tokens)

# Singleton (loaded in api/main.py; kept here for type clarity)
MODEL: ModelWrapper | None = None
APP_CFG: Dict[str, Any] | None = None

def init(model_wrapper: ModelWrapper, app_cfg_path: str = "conf/app.yaml"):
    global MODEL, APP_CFG
    MODEL = model_wrapper
    APP_CFG = load_yaml(app_cfg_path)
    log.info("infer.py initialized")

# --- Core helpers ---
def _persona_cfg(name: str) -> Dict[str, Any]:
    personas = (APP_CFG or {}).get("personas", {})
    if name in personas:
        return personas[name]
    return personas.get("default", {"style": "neutral", "enforce_otc": True, "include_disclaimer": True})

def _system_prompt_for(persona_cfg: Dict[str, Any]) -> str:
    sys = otc_guardrail_prompt(domain_hint="OTC primary care") if persona_cfg.get("enforce_otc", True) \
         else "You are a concise medical research assistant. Cite sources."
    style = persona_cfg.get("style", "neutral")
    disclaimer = ""
    if persona_cfg.get("include_disclaimer", True):
        disclaimer = (
            "\nDisclaimer: I am not a substitute for a clinician. For emergencies, call local emergency services."
        )
    return f"{sys}\nStyle: {style}.{disclaimer}"

# --- Public entry used by FastAPI route ---
def run_inference(request: Dict[str, Any]) -> Dict[str, Any]:
    assert MODEL is not None, "ModelWrapper not initialized"

    asm = ASMETHOD(**(request.get("asmethod") or {}))
    wwm = WWHAM(**(request.get("wwham") or {}))
    persona = str(request.get("persona") or (APP_CFG or {}).get("default_persona", "patient"))
    p_cfg = _persona_cfg(persona)

    # 1) Safety gate
    tr = triage(asm, wwm)
    if not tr.safe:
        msg = (
            f"I recommend seeing a clinician ({tr.level}). "
            f"Reason: {', '.join(tr.reasons) or 'sa
