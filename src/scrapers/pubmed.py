# src/scrapers/pubmed.py
import os, math, json, xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote_plus

from .http import make_session, get_text
from src.utils.logger import get_logger

log = get_logger("pubmed")

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

def _api_key() -> Optional[str]:
    return os.getenv("NCBI_API_KEY")  # optional, raises QPS to ~10/s

def _build_query(term: str, tmpl: str, since_year: str, until_year: str) -> str:
    q = tmpl.replace("%TERM%", term)
    if since_year or until_year:
        # publication date filter (pdat)
        sy = since_year or "1800"
        uy = until_year or "3000"
        q = f"({q}) AND ({sy}[pdat] : {uy}[pdat])"
    return q

def _esearch_all_pmids(session, query: str, retmax: int, max_docs: int) -> List[str]:
    """
    Get PMIDs via paging. Returns up to max_docs PMIDs.
    """
    pmids: List[str] = []
    api_key = _api_key()

    # First call to get count
    params = {
        "db": "pubmed", "term": query, "retmode": "json",
        "retmax": str(min(retmax, max_docs)), "retstart": "0",
    }
    if api_key: params["api_key"] = api_key

    data = json.loads(get_text(session, ESEARCH, params))
    count = int(data.get("esearchresult", {}).get("count", 0))
    if count == 0:
        return []

    remaining = min(count, max_docs)
    while len(pmids) < remaining:
        retstart = len(pmids)
        params = {
            "db": "pubmed", "term": query, "retmode": "json",
            "retmax": str(min(retmax, remaining - retstart)),
            "retstart": str(retstart),
        }
        if api_key: params["api_key"] = api_key
        page = json.loads(get_text(session, ESEARCH, params))
        ids = page.get("esearchresult", {}).get("idlist", []) or []
        if not ids:
            break
        pmids.extend(ids)
        if len(ids) < int(params["retmax"]):
            break  # short page

    # unique-preserve order
    seen = set()
    uniq = []
    for x in pmids:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq[:max_docs]

def _parse_pubmed_xml(xml_text: str) -> List[Dict]:
    """
    Parse efetch XML into a list of dicts (id, title, abstract, journal, date, authors, mesh, keywords, doi).
    """
    out: List[Dict] = []
    root = ET.fromstring(xml_text)
    for art in root.findall("./PubmedArticle"):
        # Basic paths
        mc = art.find("./MedlineCitation")
        article = mc.find("./Article") if mc is not None else None
        pmid = (mc.findtext("./PMID") if mc is not None else None) or art.findtext("./PubmedData/ArticleIdList/ArticleId[@IdType='pubmed']")

        title = article.findtext("./ArticleTitle") if article is not None else None

        # Abstract (concatenate labeled sections)
        abstract = None
        if article is not None:
            abs_nodes = article.findall(".//Abstract/AbstractText")
            parts = []
            for n in abs_nodes:
                label = n.get("Label") or n.get("NlmCategory")
                text = (n.text or "").strip()
                if not text:
                    continue
                parts.append(f"{label}: {text}" if label else text)
            abstract = "\n".join(parts) if parts else None

        # Journal + date
        journal_title = None
        pub_year, pub_month, pub_day = None, None, None
        if article is not None:
            jt = article.findtext(".//Journal/Title")
            jiso = article.findtext(".//Journal/ISOAbbreviation")
            journal_title = jt or jiso
            pub_year = article.findtext(".//JournalIssue/PubDate/Year")
            pub_month = article.findtext(".//JournalIssue/PubDate/Month")
            pub_day = article.findtext(".//JournalIssue/PubDate/Day")
        pub_date = "-".join([x for x in [pub_year, pub_month, pub_day] if x])

        # Authors
        authors = []
        if article is not None:
            for au in article.findall(".//AuthorList/Author"):
                cn = au.findtext("./CollectiveName")
                if cn:
                    authors.append(cn)
                    continue
                ln = au.findtext("./LastName") or ""
                fn = au.findtext("./ForeName") or ""
                full = (fn + " " + ln).strip()
                if full:
                    authors.append(full)

        # MeSH terms
        mesh = []
        for m in mc.findall(".//MeshHeadingList/MeshHeading/DescriptorName") if mc is not None else []:
            if m.text:
                mesh.append(m.text)

        # Keywords
        keywords = []
        for k in mc.findall(".//KeywordList/Keyword") if mc is not None else []:
            if k.text:
                keywords.append(k.text)

        # DOI
        doi = None
        for aid in art.findall(".//ArticleIdList/ArticleId"):
            if (aid.get("IdType") or "").lower() == "doi":
                doi = aid.text
                break

        out.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal_title,
            "pub_date": pub_date or None,
            "authors": authors or None,
            "mesh": mesh or None,
            "keywords": keywords or None,
            "doi": doi,
        })
    return out

def _efetch_batch(session, pmids: List[str]) -> List[Dict]:
    api_key = _api_key()
    params = {"db": "pubmed", "retmode": "xml", "rettype": "abstract", "id": ",".join(pmids)}
    if api_key:
        params["api_key"] = api_key
    xml_text = get_text(session, EFETCH, params)
    return _parse_pubmed_xml(xml_text)

def fetch_term(
    term: str,
    out_dir: str,
    retmax: int = 1000,
    max_docs_per_term: int = 5000,
    efetch_batch_size: int = 200,
    search_query_template: str = '("%TERM%"[Title/Abstract]) OR ("%TERM%"[MeSH Terms])',
    since_year: str = "",
    until_year: str = "",
    session=None
) -> int:
    """
    Fetch PubMed articles for one disease term:
      1) ESearch to collect PMIDs (paged),
      2) EFetch in batches → sharded JSONL.
    Returns number of articles saved.
    """
    os.makedirs(out_dir, exist_ok=True)
    session = session or make_session()

    query = _build_query(term, search_query_template, since_year, until_year)
    pmids = _esearch_all_pmids(session, query, retmax=retmax, max_docs=max_docs_per_term)
    if not pmids:
        log.info(f"[pubmed] 0 articles for '{term}'")
        return 0

    saved = 0
    shard_idx = 0
    for i in range(0, len(pmids), efetch_batch_size):
        batch = pmids[i:i+efetch_batch_size]
        try:
            arts = _efetch_batch(session, batch)
        except Exception as e:
            log.warning(f"[pubmed] efetch failed term='{term}' ids[{i}:{i+len(batch)}]: {e}")
            continue
        if not arts:
            continue
        shard_idx += 1
        shard = os.path.join(out_dir, f"{term.replace(' ','_').lower()}_{shard_idx:04d}.jsonl")
        with open(shard, "w", encoding="utf-8") as f:
            for a in arts:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
        log.info(f"[pubmed] {term}: saved {len(arts)} → {shard}")
        saved += len(arts)

    return saved
