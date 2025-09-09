from __future__ import annotations
import os, glob, orjson, random
from typing import List, Dict, Any
from ...utils.logger import get_logger
from ...utils.io import read_jsonl, ensure_dir
from ...utils.config import load_yaml
from .schema import SFTItem, ContextChunk
from .templates import render_dialog

log = get_logger("sft-build")

def _mk_ctx(text: str, source_url: str | None, prov_source: str | None, max_chars: int = 800) -> ContextChunk:
    snip = text.strip()
    if len(snip) > max_chars:
        snip = snip[:max_chars] + " ..."
    label = source_url or prov_source or "source"
    return ContextChunk(id=label, text=snip, cite=label)

def _from_label(doc: Dict[str, Any], cfg: Dict[str, Any]) -> List[SFTItem]:
    out: List[SFTItem] = []
    sections: Dict[str,str] = doc.get("sections") or {}
    title = doc.get("title") or "the product"
    prov = doc.get("prov") or {}
    src_url = prov.get("source_url")
    src_name = prov.get("source")

    # Build a few instruction types from common sections
    pairs = [
        ("What are the indications and usual dosing for {title}?", ["indications","dosage"]),
        ("List the key contraindications and serious warnings for {title}.", ["contraindications","warnings"]),
        ("What important drug interactions should be considered for {title}?", ["interactions"]),
        ("Provide patient counseling points and storage instructions for {title}.", ["patient_information","storage_and_handling"]),
    ]
    for instr_t, keys in pairs:
        ctx_chunks = []
        for k in keys:
            if k in sections and len(sections[k]) >= cfg["sft"]["build"]["min_section_chars"]:
                ctx_chunks.append(_mk_ctx(sections[k], src_url, src_name))
        if not ctx_chunks:
            continue
        instruction = instr_t.format(title=title)
        # Draft an expected answer from sections (extractive-abstractive mix)
        # For SFT targets we keep it extractive and ask the model to cite.
        bullets = []
        for c in ctx_chunks:
            bullets.append(c.text)
        target = "Summary:\n" + "\n\n".join(bullets) + "\n\n<cite>" + (src_url or src_name or "label") + "</cite>"
        out.append(SFTItem(
            role="pharmacist",
            instruction=instruction,
            input=None,
            context=ctx_chunks,
            output=target,
            meta={"source_id": doc.get("id"), "doc_type": "drug_label"}
        ))
    return out

def _from_trial(doc: Dict[str, Any], cfg: Dict[str, Any]) -> List[SFTItem]:
    out: List[SFTItem] = []
    meta = doc.get("codes") or {}
    prov = doc.get("prov") or {}
    src_url = prov.get("source_url")
    body = (doc.get("sections") or {}).get("body") or ""
    if len(body) < cfg["sft"]["build"]["min_section_chars"]:
        return out
    nct = doc.get("meta",{}).get("nctid") if isinstance(doc.get("meta"), dict) else None
    instr = f"Summarize the primary outcomes and interventions of trial {nct or ''}. Provide a short, neutral synopsis."
    ctx = [_mk_ctx(body, src_url, "clinicaltrials")]
    target = f"Primary outcomes and interventions summarized from the registry.\n\n{ctx[0].text}\n\n<cite>{src_url or 'ClinicalTrials.gov'}</cite>"
    out.append(SFTItem(role="doctor", instruction=instr, input=None, context=ctx, output=target,
                       meta={"source_id": doc.get("id"), "doc_type": "trial"}))
    return out

def _from_article(doc: Dict[str, Any], cfg: Dict[str, Any]) -> List[SFTItem]:
    out: List[SFTItem] = []
    prov = doc.get("prov") or {}
    src_url = prov.get("source_url")
    title = doc.get("title") or ""
    body = (doc.get("sections") or {}).get("body") or ""
    if len(body) < cfg["sft"]["build"]["min_section_chars"]:
        return out
    instr = f"Summarize the key findings of the article: {title}. Keep it medically precise, 3-5 bullet points."
    ctx = [_mk_ctx(body, src_url, prov.get("source"))]
    target = f"- {ctx[0].text}\n\n<cite>{src_url or 'PubMed/PMC'}</cite>"
    out.append(SFTItem(role="doctor", instruction=instr, input=None, context=ctx, output=target,
                       meta={"source_id": doc.get("id"), "doc_type": "article"}))
    return out

BUILDERS = {
    "drug_label": _from_label,
    "trial": _from_trial,
    "article": _from_article,
}

def _emit(items: List[SFTItem], out_path: str):
    ensure_dir(os.path.dirname(out_path))
    cnt = 0
    with open(out_path, "wb") as f:
        for it in items:
            f.write(orjson.dumps(it.model_dump(mode="json"), option=orjson.OPT_APPEND_NEWLINE))
            cnt += 1
    return cnt

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/train_sft.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    globs = cfg["sft"]["build"]["input_globs"]
    max_per = cfg["sft"]["build"]["max_tasks"]
    files = []
    for g in globs: files.extend(glob.glob(g))
    files.sort()

    all_items: List[SFTItem] = []
    per_type = {"drug_label":0, "trial":0, "article":0}
    for fp in files:
        for row in read_jsonl(fp):
            typ = row.get("type")
            if typ not in BUILDERS: continue
            if per_type[typ] >= max_per.get(typ, 0): continue
            items = BUILDERS[typ](row, cfg)
            if not items: continue
            all_items.extend(items)
            per_type[typ] += 1
    random.shuffle(all_items)
    outp = cfg["sft"]["build"]["out_jsonl"]
    n = _emit(all_items, outp)
    log.info(f"[sft-build] wrote {n} tasks -> {outp}")

if __name__ == "__main__":
    main()
