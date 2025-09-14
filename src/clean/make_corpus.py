# src/clean/make_corpus.py
import os, re, json, glob, argparse
from typing import Dict, Any, Iterable

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            yield json.loads(line)

def safe_join(parts):
    return "\n\n".join([p for p in parts if p and str(p).strip()])

def normalize_pubmed(in_dir: str):
    for p in sorted(glob.glob(os.path.join(in_dir, "*.jsonl"))):
        # infer seed term from filename: "<term>_0001.jsonl"
        m = re.search(r"([A-Za-z0-9_\-]+)_\d+\.jsonl$", os.path.basename(p))
        seed_term = m.group(1).replace("_", " ") if m else None
        for row in iter_jsonl(p):
            pmid = row.get("pmid")
            title = row.get("title")
            abstract = row.get("abstract")
            text = safe_join([title, abstract])
            if not text:
                continue
            yield {
                "id": f"PMID:{pmid}" if pmid else None,
                "source": "pubmed",
                "seed_term": seed_term,
                "title": title,
                "text": text,
                "meta": {
                    "journal": row.get("journal"),
                    "pub_date": row.get("pub_date"),
                    "authors": row.get("authors"),
                    "mesh": row.get("mesh"),
                    "keywords": row.get("keywords"),
                    "doi": row.get("doi"),
                },
            }

def normalize_ctgov(in_dir: str):
    for p in sorted(glob.glob(os.path.join(in_dir, "*.jsonl"))):
        m = re.search(r"([A-Za-z0-9_\-]+)_\d+-?\d*\.jsonl$", os.path.basename(p))
        seed_term = m.group(1).replace("_", " ") if m else None
        for row in iter_jsonl(p):
            # v2 raw study dict from scraper
            # try common paths; tolerate missing fields
            nct = None
            try:
                nct = (
                    row.get("protocolSection", {})
                       .get("identificationModule", {})
                       .get("nctId")
                ) or row.get("nctId")
            except Exception:
                pass
            id_ = f"NCT:{nct}" if nct else None
            brief = None
            official = None
            try:
                ident = row.get("protocolSection", {}).get("identificationModule", {})
                brief = ident.get("briefTitle")
                official = ident.get("officialTitle")
            except Exception:
                pass

            outcomes = []
            try:
                pos = (row.get("protocolSection", {})
                          .get("outcomesModule", {})
                          .get("primaryOutcomes")) or []
                for po in pos:
                    m = po.get("measure") or po.get("description") or po.get("name")
                    if m: outcomes.append(m)
            except Exception:
                pass

            text = safe_join([
                brief or official,
                f"Primary outcomes: { '; '.join(outcomes) }" if outcomes else None
            ])
            if not text:
                continue

            # status & dates
            status = None
            first_posted = None
            last_update = None
            try:
                status_mod = row.get("protocolSection", {}).get("statusModule", {})
                status = status_mod.get("overallStatus")
                fps = status_mod.get("studyFirstPostDateStruct", {})
                first_posted = fps.get("date")
                lup = status_mod.get("lastUpdatePostDateStruct", {})
                last_update = lup.get("date")
            except Exception:
                pass

            # conditions
            conds = None
            try:
                conds = row.get("protocolSection", {}).get("conditionsModule", {}).get("conditions")
            except Exception:
                pass

            yield {
                "id": id_,
                "source": "ctgov",
                "seed_term": seed_term,
                "title": brief or official,
                "text": text,
                "meta": {
                    "overall_status": status,
                    "first_posted_date": first_posted,
                    "last_update_posted_date": last_update,
                    "conditions": conds,
                },
            }

def main():
    ap = argparse.ArgumentParser(description="Unify PubMed + CTGov shards into a single corpus.jsonl")
    ap.add_argument("--pubmed_dir", default="./data/raw/pubmed")
    ap.add_argument("--ctgov_dir", default="./data/raw/clinicaltrials")
    ap.add_argument("--out", default="./data/cleaned/corpus.jsonl")
    ap.add_argument("--min_chars", type=int, default=400, help="Filter very short texts")
    ap.add_argument("--lang", default="en", help="(optional) lang filter placeholder")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    seen = set()
    kept = 0

    with open(args.out, "w", encoding="utf-8") as out:
        # PubMed
        for rec in normalize_pubmed(args.pubmed_dir):
            if not rec["text"] or len(rec["text"]) < args.min_chars:
                continue
            # exact dedupe by id + title hash
            key = (rec["id"], rec["title"])
            if key in seen:
                continue
            seen.add(key)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

        # CTGov
        for rec in normalize_ctgov(args.ctgov_dir):
            if not rec["text"] or len(rec["text"]) < args.min_chars:
                continue
            key = (rec["id"], rec["title"])
            if key in seen:
                continue
            seen.add(key)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"OK: wrote {kept} records → {args.out}")

if __name__ == "__main__":
    main()
