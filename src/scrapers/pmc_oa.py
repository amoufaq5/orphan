from __future__ import annotations
import os, math, datetime as dt
from typing import Any, Dict, Iterable, List
import httpx
from lxml import etree
from .base_scraper import BaseScraper, ScrapeResult
from .registry import register
from ..utils.logger import get_logger
from ..utils.schemas import RawDoc, Provenance

log = get_logger("pmc")

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
API_KEY = os.getenv("PUBMED_API_KEY", os.getenv("NCBI_API_KEY", ""))

@register("pmc_open_access")
class PMCOpenAccessScraper(BaseScraper):
    """
    ESearch db=pmc to collect PMCIDs; EFetch XML to parse title + abstract + (optionally) body
    Only uses content where PMC Open Access (license permitting). We keep metadata + abstract by default.
    """

    name = "pmc_open_access"

    def __init__(self, *args, query: str = "open access[filter]", retmax: int = 200, include_body: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.query = query
        self.retmax = max(1, min(200, retmax))
        self.include_body = include_body  # false by default to keep parsing light

    async def run(self) -> ScrapeResult:
        async with await self._client() as client:
            # ESearch for pmc ids (like PMC1234567)
            ids: List[str] = []
            page = 0
            while page < self.max_pages and len(ids) < self.max_docs:
                params = {
                    "db": "pmc",
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
                page += 1
                if page >= math.ceil(min(total, self.max_docs) / self.retmax):
                    break

            if not ids:
                from ..utils.io import shard_writer
                write, close = shard_writer(self.shards_dir, f"{self.name}", self.shard_max_records)
                close()
                return ScrapeResult(0, self.shards_dir)

            from ..utils.io import shard_writer
            write, close = shard_writer(self.shards_dir, f"{self.name}", self.shard_max_records)
            fetched = 0
            batch = 100
            for i in range(0, min(len(ids), self.max_docs), batch):
                chunk = ids[i:i+batch]
                params = {
                    "db": "pmc",
                    "id": ",".join(chunk),
                    "retmode": "xml",
                }
                if API_KEY:
                    params["api_key"] = API_KEY
                self.bucket.take(1)
                r = await client.get(f"{EUTILS}/efetch.fcgi", params=params)
                r.raise_for_status()
                xml = r.text
                for doc in self._parse_pmc_xml(xml):
                    write(doc.model_dump(mode="json"))
                    fetched += 1
                    if fetched >= self.max_docs:
                        break
                if fetched >= self.max_docs:
                    break
            close()
            return ScrapeResult(fetched, self.shards_dir)

    def _parse_pmc_xml(self, xml_text: str) -> Iterable[RawDoc]:
        try:
            root = etree.fromstring(xml_text.encode("utf-8"))
        except Exception as e:
            log.error(f"[pmc] XML parse error: {e}")
            return []

        ns = {"x": "http://www.ncbi.nlm.nih.gov/JATS1"}  # JATS sometimes present; we use local tag names
        articles = root.findall(".//article")
        out: List[RawDoc] = []
        for a in articles:
            # pmcid (front/article-id pub-id-type="pmcid")
            pmcid = None
            for aid in a.findall(".//article-id"):
                if aid.get("pub-id-type") == "pmcid" and aid.text:
                    pmcid = aid.text.strip()
            # title
            title_el = a.find(".//article-title")
            title = etree.tostring(title_el, method="text", encoding="unicode").strip() if title_el is not None else None
            # abstract
            abs_el = a.find(".//abstract")
            abstract = etree.tostring(abs_el, method="text", encoding="unicode").strip() if abs_el is not None else None
            # license statement (for awareness; we still do metadata/abstract primarily)
            license_el = a.find(".//license")
            license_text = etree.tostring(license_el, method="text", encoding="unicode").strip() if license_el is not None else None

            body_text = None
            if self.include_body:
                body_el = a.find(".//body")
                if body_el is not None:
                    body_text = etree.tostring(body_el, method="text", encoding="unicode").strip()

            text = abstract if not self.include_body else "\n\n".join([t for t in [abstract, body_text] if t])

            prov = Provenance(
                source="pmc_oa",
                source_url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/" if pmcid else "https://www.ncbi.nlm.nih.gov/pmc/",
                license=license_text or "PMC Open Access (see article license)",
                retrieved_at=dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                hash=None,
            )

            out.append(
                RawDoc(
                    id=f"pmc:{pmcid}" if pmcid else f"pmc:auto:{abs(hash(text))%10**12}",
                    title=title,
                    text=text or None,
                    meta={"type": "article", "lang": "en"},
                    prov=prov,
                )
            )
        return out
