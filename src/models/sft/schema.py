from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ContextChunk(BaseModel):
    id: str
    text: str
    cite: Optional[str] = None  # e.g., URL or provenance id

class SFTItem(BaseModel):
    role: str                   # "patient" | "doctor" | "pharmacist"
    instruction: str            # user ask / instruction
    input: Optional[str] = None # optional extra fields (e.g., vitals)
    context: List[ContextChunk] = Field(default_factory=list)
    output: str                 # target answer
    meta: Dict[str, Any] = Field(default_factory=dict)  # {source_id, doc_type, section, lang, …}
