from __future__ import annotations
import os, glob, orjson
from typing import Dict, Any, Iterable, List
from ..utils.logger import get_logger
from ..utils.io import read_jsonl, shard_writer
from ..utils.schemas import RawDoc, CanonicalDoc, Provenance
from ..ontology.icd10 import map_icd10  # stubs
from ..ontology.atc import map_atc
from ..ontology.rxnorm import map_rxnorm
from ..ontology.mesh import map_mesh

log = get_logger("canonical")

SECTION_HINTS = {
    "drug_label": ["indications", "dosage", "contraindications", "warnings", "adverse_reactions", "interactions", "storage"],
    "article":    ["abstract", "body"],
    "trial":      ["summary", "results", "eligibility"],
}

def _row_to_rawdoc(row: Dict[str, Any]) -> RawDoc:
    # pydantic model expects fields; we trust our scrapers but keep guards
    prov = row.get("prov") or {}
    return RawDoc(
        id=row.get("id", ""),
        title=row.get("title"),
        text=row.get("text"),
        meta=row.get("meta") or {},
        prov=Provenance(
            source=prov.get("source", ""),
            source_url=prov.get("source_url"),
            license=prov.get("license"),
            retrieved_at=prov.get("retrieved_at"),
            hash=prov.get("hash"),
        ),
    )

def _canonize(raw: RawDoc) -> CanonicalDoc:
    typ = (raw.meta.get("type") or "article").lower()
    sections = {}
    txt = (raw.text or "").strip()
    if typ == "drug_label":
        sections["body"] = txt
    elif typ == "trial":
        sections["body"] = txt
    else:
        # article
        # If we detect 'Introduction', 'Methods', etc., we could split later. For now keep body.
        sections["body"] = txt

    # Ontology mapping (basic stubs)
    codes = {}
    if typ != "drug_label":
        # map MeSH terms if present
        mesh = raw.meta.get("mesh") or []
        if mesh:
            codes["MeSH"] = map_mesh(mesh)
    # simple examples for others (extend later)
    # rxnorm/atc on labels could be derived from openfda meta in future stage

    return CanonicalDoc(
        id=raw.id,
        type=typ,
        title=raw.title,
        sections=sections,
        lang=raw.meta.get("lang", "en"),
        codes=codes,
        prov=raw.prov,
    )

def convert_shards(shards_dir: str, out_dir: str, prefix_filters: List[str] | None = None, shard_max: int = 5000):
    os.makedirs(out_dir, exist_ok=True)
    patterns = [os.path.join(shards_dir, "*.jsonl")]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files.sort()

    write, close = shard_writer(out_dir, "canonical", shard_max)
    total = 0
    for fp in files:
        base = os.path.basename(fp)
        if prefix_filters and not any(base.startswith(p) for p in prefix_filters):
            continue
        for row in read_jsonl(fp):
            raw = _row_to_rawdoc(row)
            doc = _canonize(raw)
            write(orjson.loads(doc.model_dump_json()))
            total += 1
    close()
    log.info(f"[canonical] wrote={total} to {out_dir}")
