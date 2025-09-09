from __future__ import annotations
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from ..utils.config import load_yaml
from ..utils.logger import get_logger

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

@app.get("/health")
def health():
    return {"status": "ok", "env": cfg["app"]["env"]}

@app.get("/triage")
def triage_placeholder(q: str):
    # Placeholder response until model + RAG are wired
    return {
        "query": q,
        "triage": {
            "recommendation": "OTC-first (placeholder)",
            "refer": False,
            "explain": "Rationale and citations will appear here.",
        }
    }
