from __future__ import annotations
from typing import Iterable, Dict, Any
from ..utils.schemas import RawDoc, CanonicalDoc, Provenance
from ..utils.pii import scrub_record

SECTION_SPLIT = ["indications", "dosage", "contraindications", "warnings", "adverse_reactions", "references"]

def to_canonical(raw: RawDoc) -> CanonicalDoc:
    rec = scrub_record(raw.model_dump())
    text = (rec.get("text") or "").strip()
    sections = {"body": text}
    # TODO: real sectioning per source type
    prov = Provenance(**rec["prov"])
    return CanonicalDoc(
        id=rec["id"],
        type=rec["meta"].get("type", "article"),
        title=rec.get("title"),
        sections=sections,
        lang=rec["meta"].get("lang", "en"),
        codes={},
        prov=prov,
    )

def normalize_stream(raw_iter: Iterable[RawDoc]) -> Iterable[CanonicalDoc]:
    for r in raw_iter:
        yield to_canonical(r)
