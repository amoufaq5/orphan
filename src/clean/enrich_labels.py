from __future__ import annotations
import os, glob, orjson
from typing import Dict, Any
from ..utils.logger import get_logger
from ..utils.io import read_jsonl, shard_writer
from ..utils.schemas import CanonicalDoc, Provenance
from .label_sectioner import split_drug_label

log = get_logger("enrich-labels")

def _to_canonical(doc: Dict[str, Any]) -> CanonicalDoc:
    prov = doc.get("prov") or {}
    return CanonicalDoc(
        id=doc.get("id", ""),
        type=doc.get("type", "article"),
        title=doc.get("title"),
        sections=doc.get("sections") or {},
        lang=doc.get("lang", "en"),
        codes=doc.get("codes") or {},
        prov=Provenance(
            source=prov.get("source",""),
            source_url=prov.get("source_url"),
            license=prov.get("license"),
            retrieved_at=prov.get("retrieved_at"),
            hash=prov.get("hash"),
        ),
    )

def enrich_dir(canonical_dir: str, out_dir: str, shard_max: int = 5000):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(canonical_dir, "*.jsonl")))
    write, close = shard_writer(out_dir, "canonical_labels_enriched", shard_max)
    total = 0

    for fp in files:
        for row in read_jsonl(fp):
            cdoc = _to_canonical(row)
            if cdoc.type != "drug_label":
                # pass-through non labels
                write(orjson.dumps(row))
                total += 1
                continue

            # extract structured sections if raw meta existed upstream
            meta = (row.get("meta") or {})
            # we stored source & raw at RawDoc stage; propagate lightly if available
            # if missing, we keep the original sections
            structured = split_drug_label(meta) if meta else {}
            if structured:
                cdoc.sections = structured
            # dump
            write(orjson.loads(cdoc.model_dump_json()))
            total += 1

    close()
    log.info(f"[enrich-labels] wrote={total} records -> {out_dir}")
