from __future__ import annotations
import os, math, datetime as dt
from typing import Any, Dict, Iterable, List, Tuple
import httpx
from lxml import etree
from .base_scraper import BaseScraper, ScrapeResult
from .registry import register
from ..utils.logger import get_logger
from ..utils.schemas import RawDoc, Provenance

log = get_logger("pubmed")

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
API_KEY = os.getenv("PUBMED_API_KEY", os.getenv("NCBI_API_KEY", ""))

@register("pubmed_abstracts")
class PubMedAbstractsScraper(BaseScraper):
    """
    Two-phase: ESearch for PMIDs -> batch EFetch (XML) -> parse title+abstract+meSH.
    Respects API key if provided. DOES NOT fetch full text, abstracts only.
    """

    name = "pubmed_abstracts"

    def __init__(self, *args, query: str = "clinical[Title/Abstract] OR medicine[All Fields]", retmax: int = 200, **kwargs):
        super().__init__(*args, **kwargs)
        self.query = query
        self.retmax = max(1, min(200, retmax))

    async def run(self) -> ScrapeResult:
        # Phase 1: ESearch to get list of PMIDs
        async with await self._client() as client:
            ids: List[str] = []
            page = 0
            while page < self.max_pages and len(ids) < self.max_docs:
                params = {
                    "db": "pubmed",
                    "term": self.query,
                    "retmode": "json",
                    "retmax": self.retmax,
                    "retstart": page * self.retmax,
                }
                if API_KEY:
                    params["api_key"] = API_KEY
                data = await self._get_json(client, f"{EUTILS}/esearch.fcgi", params)
                idlist = (((data.get("esearchresult") or {}).get("idlist")) or [])
                if not idlist:
                    break
                ids.extend(idlist)
                total = int((data.get("esearchresult") or {}).get("count", "0"))
                max_pages_est = math.ceil(min(total, self.max_docs) / self.retmax) if self.retmax else 0
                page += 1
                if page >= max_pages_est:
                    break

            if not ids:
                # nothing to write, create empty shard
                from ..utils.io import shard_writer
                write, close = shard_writer(self.shards_dir, f"{self.name}", self.shard_max_records)
                close()
                return ScrapeResult(0, self.shards_dir)

            # Phase 2: batch EFetch in chunks of retmax
            from ..utils.io import shard_writer
            write, close = shard_writer(self.shards_dir, f"{self.name}", self.shard_max_records)
            fetched = 0
            batch = 200
            for i in range(0, min(len(ids), self.max_docs), batch):
                chunk = ids[i:i+batch]
                params = {
                    "db": "pubmed",
                    "id": ",".join(chunk),
                    "retmode": "xml",
                }
                if API_KEY:
                    params["api_key"] = API_KEY
                self.bucket.take(1)
                r = await client.get(f"{EUTILS}/efetch.fcgi", params=params)
                r.raise_for_status()
                xml = r.text
                for doc in self._parse_pubmed_xml(xml):
                    write(doc.model_dump(mode="json"))
                    fetched += 1
                    if fetched >= self.max_docs:
                        break
                if fetched >= self.max_docs:
                    break
            close()
            return ScrapeResult(fetched, self.shards_dir)

    def _parse_pubmed_xml(self, xml_text: str) -> Iterable[RawDoc]:
        """Parse PubMed efetch XML for ArticleTitle, AbstractText, MeSH"""
        try:
            root = etree.fromstring(xml_text.encode("utf-8"))
        except Exception as e:
            log.error(f"[pubmed] XML parse error: {e}")
            return []

        ns = {}  # PubMed XML doesn't require xmlns
        articles = root.findall(".//PubmedArticle", namespaces=ns)
        out: List[RawDoc] = []
        for a in articles:
            pmid_el = a.find(".//PMID", namespaces=ns)
            pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else None

            title_el = a.find(".//ArticleTitle", namespaces=ns)
            title = etree.tostring(title_el, method="text", encoding="unicode").strip() if title_el is not None else None

            abs_els = a.findall(".//Abstract/AbstractText", namespaces=ns)
            abstract_parts: List[str] = []
            for ab in abs_els or []:
                label = ab.get("Label")
                txt = etree.tostring(ab, method="text", encoding="unicode").strip()
                if label:
                    abstract_parts.append(f"{label}: {txt}")
                else:
                    abstract_parts.append(txt)
            abstract = "\n\n".join([t for t in abstract_parts if t])

            mesh_terms = []
            for mh in a.findall(".//MeshHeading", namespaces=ns):
                d = mh.find("./DescriptorName", namespaces=ns)
                if d is not None and d.text:
                    mesh_terms.append(d.text.strip())

            prov = Provenance(
                source="pubmed",
                source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "https://pubmed.ncbi.nlm.nih.gov/",
                license="PubMed metadata and abstracts per NLM terms",
                retrieved_at=dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                hash=None,
            )
            out.append(
                RawDoc(
                    id=f"pubmed:{pmid}" if pmid else f"pubmed:auto:{abs(hash(abstract))%10**12}",
                    title=title,
                    text=abstract or None,
                    meta={"type": "article", "mesh": mesh_terms, "lang": "en"},
                    prov=prov,
                )
            )
        return out
