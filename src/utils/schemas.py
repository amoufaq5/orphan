from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class Provenance(BaseModel):
    source: str
    source_url: Optional[str] = None
    license: Optional[str] = None
    retrieved_at: Optional[str] = None
    hash: Optional[str] = None

class RawDoc(BaseModel):
    id: str
    title: Optional[str] = None
    text: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
    prov: Provenance

class CanonicalDoc(BaseModel):
    id: str
    type: str  # "drug_label" | "trial" | "guideline" | "article" | ...
    title: Optional[str] = None
    sections: Dict[str, str] = Field(default_factory=dict)
    lang: Optional[str] = "en"
    codes: Dict[str, List[str]] = Field(default_factory=dict)  # ICD10, ATC, RxNorm, MeSH
    prov: Provenance
