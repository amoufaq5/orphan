from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ..utils.config import load_yaml
from ..utils.logger import get_logger
from .infer import SPTokenizer, load_model, generate, build_prompt
from .rag import RAG

cfg = load_yaml("conf/app.yaml")
log = get_logger("api", cfg["app"]["log_level"])

app = FastAPI(title=cfg["app"]["name"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg["app"]["api"]["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_tok = None
_model = None
_rag = None

class GenRequest(BaseModel):
    instruction: str
    persona: str | None = None
    context: list[str] | None = None
    enforce_citations: bool = True
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None

class RAGRequest(BaseModel):
    instruction: str
    persona: str | None = None
    enforce_citations: bool = True
    top_k: int | None = None
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None

@app.on_event("startup")
def _startup():
    global _tok, _model, _rag
    rt = cfg["model_runtime"]
    _tok = SPTokenizer(rt["spm_model"])
    _model = load_model(rt["spm_model"], rt["sft_checkpoint"], rt["max_seq_len"])
    _rag = RAG("conf/app.yaml")
    log.info("[api] model, tokenizer, and RAG loaded")

@app.get("/health")
def health():
    return {"status": "ok", "env": cfg["app"]["env"]}

@app.post("/generate")
def generate_endpoint(body: GenRequest):
    rt = cfg["model_runtime"]
    persona = (body.persona or rt.get("persona") or "doctor").lower()
    prompt = build_prompt(persona, body.instruction, body.context or [], enforce_citations=body.enforce_citations)
    text = generate(
        _model, _tok, prompt,
        max_new_tokens=body.max_new_tokens or rt["max_new_tokens"],
        temperature=body.temperature if body.temperature is not None else rt["temperature"],
        top_p=body.top_p if body.top_p is not None else rt["top_p"],
    )
    try:
        ans = text.split("[ASSISTANT]", 1)[1]
    except Exception:
        ans = text
    return {"persona": persona, "answer": ans.strip()}

@app.post("/rag/generate")
def rag_generate(body: RAGRequest):
    rt = cfg["model_runtime"]
    persona = (body.persona or rt.get("persona") or "doctor").lower()
    hits = _rag.retrieve(body.instruction, top_k=body.top_k)
    context_snippets = _rag.format_context(hits)
    prompt = build_prompt(persona, body.instruction, context_snippets, enforce_citations=body.enforce_citations)
    text = generate(
        _model, _tok, prompt,
        max_new_tokens=body.max_new_tokens or rt["max_new_tokens"],
        temperature=body.temperature if body.temperature is not None else rt["temperature"],
        top_p=body.top_p if body.top_p is not None else rt["top_p"],
    )
    try:
        ans = text.split("[ASSISTANT]", 1)[1]
    except Exception:
        ans = text
    # attach lightweight evidence manifest
    evidence = []
    for p,_ in hits:
        evidence.append({
            "id": p.pid,
            "doc_id": p.doc_id,
            "title": p.title,
            "source_url": p.source_url,
            "section": p.meta.get("section")
        })
    return {"persona": persona, "answer": ans.strip(), "evidence": evidence}
