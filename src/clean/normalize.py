from __future__ import annotations
from typing import Iterable, Dict, Any
from ..utils.schemas import RawDoc, CanonicalDoc, Provenance
from ..utils.pii import scrub_record

SECTION_SPLIT = ["indications", "dosage", "contraindications", "warnings", "adverse_reactions", "references"]

def to_canonical(raw: RawDoc) -> CanonicalDoc:
    rec = scrub_record(raw.model_dump())
    meta = rec.get("meta", {})
    doc_type = meta.get("type", "article")
    text = (rec.get("text") or "").strip()
    sections: Dict[str, str | None]

    if doc_type in {"drug_label"}:
        sections = {"body": text}
    elif doc_type in {"trial", "article", "academic_paper", "preprint"}:
        sections = {"abstract": meta.get("abstract"), "body": text}
    else:
        sections = {"summary": text}

    prov = Provenance(**rec["prov"])
    return CanonicalDoc(
        id=rec["id"],
        type=doc_type,
        title=rec.get("title"),
        sections={k: v for k, v in sections.items() if v},
        lang=meta.get("lang", "en"),
        codes={},
        prov=prov,
    )

def normalize_stream(raw_iter: Iterable[RawDoc]) -> Iterable[CanonicalDoc]:
    for r in raw_iter:
        yield to_canonical(r)
