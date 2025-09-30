from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlencode

import httpx
from bs4 import BeautifulSoup
from lxml import etree

from .base_scraper import BaseScraper, ScrapeResult
from .registry import register
from ..utils.logger import get_logger
from ..utils.schemas import Provenance, RawDoc

log = get_logger("scraper.extended")


def _now_iso() -> str:
    return dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _strip_html(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    soup = BeautifulSoup(value, "html.parser")
    text = soup.get_text(" ", strip=True)
    return text or None


def _xml_text(element: Optional[etree._Element]) -> Optional[str]:
    if element is None:
        return None
    return etree.tostring(element, method="text", encoding="unicode").strip() or None


class TermScraper(BaseScraper):
    """Common helper for scrapers driven by a list of search terms."""

    def __init__(
        self,
        *args,
        terms: Optional[List[str]] = None,
        terms_file: Optional[str] = None,
        per_term_limit: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._raw_terms = [t.strip() for t in (terms or []) if t and t.strip()]
        if terms_file:
            try:
                with open(terms_file, "r", encoding="utf-8") as fh:
                    for line in fh:
                        term = line.strip()
                        if term:
                            self._raw_terms.append(term)
            except FileNotFoundError:
                log.error(f"[{self.name}] terms_file not found: {terms_file}")
        self.per_term_limit = per_term_limit

    def _resolve_terms(self) -> List[str]:
        seen = set()
        resolved: List[str] = []
        for term in self._raw_terms:
            if term not in seen:
                resolved.append(term)
                seen.add(term)
        return resolved

    async def fetch_term(self, client: httpx.AsyncClient, term: str, limit: int) -> Iterable[RawDoc]:
        raise NotImplementedError

    async def run(self) -> ScrapeResult:
        terms = self._resolve_terms()
        if not terms:
            log.warning(f"[{self.name}] No terms provided; nothing to do.")
            from ..utils.io import shard_writer

            write, close = shard_writer(self.shards_dir, self.name, self.shard_max_records)
            close()
            return ScrapeResult(0, self.shards_dir, {"terms": []})

        from ..utils.io import shard_writer

        write, close = shard_writer(self.shards_dir, self.name, self.shard_max_records)
        total = 0

        async with await self._client() as client:
            for term in terms:
                remaining = self.max_docs - total
                if remaining <= 0:
                    break
                limit = min(self.per_term_limit or remaining, remaining)
                try:
                    docs = await self.fetch_term(client, term, limit)
                except Exception as exc:  # noqa: BLE001 - need broad catch for network layer
                    log.error(f"[{self.name}] term='{term}' failed: {exc}")
                    continue

                count_for_term = 0
                for doc in docs:
                    write(doc.model_dump(mode="json"))
                    total += 1
                    count_for_term += 1
                    if total >= self.max_docs or count_for_term >= limit:
                        break

        close()
        return ScrapeResult(total, self.shards_dir, {"terms": terms})


@register("chembl")
class ChemBLScraper(TermScraper):
    name = "chembl"
    BASE_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"

    def __init__(self, *args, page_size: int = 50, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.page_size = max(1, min(page_size, 200))

    async def fetch_term(self, client: httpx.AsyncClient, term: str, limit: int) -> Iterable[RawDoc]:
        params = {
            "format": "json",
            "limit": min(limit, self.page_size),
            "pref_name__icontains": term,
        }
        payload = await self._get_json(client, self.BASE_URL, params=params)
        molecules = payload.get("molecules") or []
        docs: List[RawDoc] = []
        for mol in molecules:
            chembl_id = mol.get("molecule_chembl_id")
            title = mol.get("pref_name") or chembl_id or term
            props = mol.get("molecule_properties") or {}
            text_parts: List[str] = []
            for key, value in props.items():
                if value:
                    text_parts.append(f"{key}: {value}")
            synonyms = mol.get("molecule_synonyms") or []
            synonym_values: List[str] = []
            for syn in synonyms:
                if isinstance(syn, dict):
                    value = syn.get("synonyms") or syn.get("synonym")
                    if value:
                        synonym_values.append(str(value))
                elif syn:
                    synonym_values.append(str(syn))
            if synonym_values:
                text_parts.append("Synonyms: " + ", ".join(synonym_values))
            mechanism = mol.get("mechanism_of_action")
            if mechanism:
                text_parts.append(f"Mechanism: {mechanism}")
            text = "\n".join(text_parts) or None
            docs.append(
                RawDoc(
                    id=f"chembl:{chembl_id or term}",
                    title=title,
                    text=text,
                    meta={
                        "type": "compound",
                        "term": term,
                        "chembl_id": chembl_id,
                        "max_phase": mol.get("max_phase"),
                    },
                    prov=Provenance(
                        source="chembl",
                        source_url=f"https://www.ebi.ac.uk/chembl/compound_report_card/{chembl_id}/"
                        if chembl_id
                        else "https://www.ebi.ac.uk/chembl/",
                        license="ChEMBL data licence (CC0)",
                        retrieved_at=_now_iso(),
                        hash=None,
                    ),
                )
            )
        return docs


@register("clinvar")
class ClinVarScraper(TermScraper):
    name = "clinvar"
    EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, *args, retmax: int = 100, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.retmax = max(1, min(retmax, 500))

    async def fetch_term(self, client: httpx.AsyncClient, term: str, limit: int) -> Iterable[RawDoc]:
        search_params = {
            "db": "clinvar",
            "term": term,
            "retmode": "json",
            "retmax": min(limit, self.retmax),
        }
        search_data = await self._get_json(client, f"{self.EUTILS}/esearch.fcgi", params=search_params)
        id_list = ((search_data.get("esearchresult") or {}).get("idlist")) or []
        if not id_list:
            return []

        docs: List[RawDoc] = []
        chunk = 50
        for i in range(0, min(len(id_list), limit), chunk):
            subset = id_list[i : i + chunk]
            summary_params = {
                "db": "clinvar",
                "id": ",".join(subset),
                "retmode": "json",
            }
            summary = await self._get_json(client, f"{self.EUTILS}/esummary.fcgi", params=summary_params)
            result = summary.get("result") or {}
            for uid in result.get("uids", []):
                item = result.get(uid) or {}
                title = item.get("title") or item.get("record_title") or term
                clin_sig = (item.get("clinical_significance") or {}).get("description")
                accession = item.get("accession")
                condition_list = []
                traits = item.get("trait_set") or []
                for trait in traits:
                    name = trait.get("trait_name")
                    if name:
                        condition_list.append(name)
                text_parts: List[str] = []
                if clin_sig:
                    text_parts.append(f"Clinical significance: {clin_sig}")
                review_status = item.get("review_status")
                if review_status:
                    text_parts.append(f"Review status: {review_status}")
                if condition_list:
                    text_parts.append("Conditions: " + ", ".join(condition_list))
                origin = item.get("origin")
                if origin:
                    text_parts.append(f"Origin: {origin}")
                description = item.get("description")
                if description:
                    text_parts.append(description)
                text = "\n".join(text_parts) or None
                docs.append(
                    RawDoc(
                        id=f"clinvar:{accession or uid}",
                        title=title,
                        text=text,
                        meta={
                            "type": "variant",
                            "term": term,
                            "accession": accession,
                            "clinical_significance": clin_sig,
                        },
                        prov=Provenance(
                            source="clinvar",
                            source_url=f"https://www.ncbi.nlm.nih.gov/clinvar/{accession or uid}/",
                            license="NCBI ClinVar usage policies",
                            retrieved_at=_now_iso(),
                            hash=None,
                        ),
                    )
                )
                if len(docs) >= limit:
                    break
            if len(docs) >= limit:
                break
        return docs


@register("medlineplus")
class MedlinePlusScraper(TermScraper):
    name = "medlineplus"
    BASE_URL = "https://wsearch.nlm.nih.gov/ws/query"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    async def fetch_term(self, client: httpx.AsyncClient, term: str, limit: int) -> Iterable[RawDoc]:
        params = {
            "db": "healthTopics",
            "term": term,
        }
        response = await client.get(self.BASE_URL, params=params)
        response.raise_for_status()
        root = etree.fromstring(response.content)
        docs: List[RawDoc] = []
        documents = root.findall(".//document")
        for doc_el in documents:
            title = _xml_text(doc_el.find("content[@name='title']")) or term
            summary = _xml_text(doc_el.find("content[@name='FullSummary']"))
            alt_summary = _xml_text(doc_el.find("content[@name='Summary']"))
            url = _xml_text(doc_el.find("content[@name='url']"))
            audience = _xml_text(doc_el.find("content[@name='audience']"))
            text = summary or alt_summary
            docs.append(
                RawDoc(
                    id=f"medlineplus:{doc_el.get('id') or title}",
                    title=title,
                    text=text,
                    meta={
                        "type": "patient_education",
                        "term": term,
                        "audience": audience,
                    },
                    prov=Provenance(
                        source="medlineplus",
                        source_url=url,
                        license="MedlinePlus Usage Guidelines",
                        retrieved_at=_now_iso(),
                        hash=None,
                    ),
                )
            )
            if len(docs) >= limit:
                break
        return docs


@register("cdc_health")
class CDCPublicHealthScraper(TermScraper):
    name = "cdc_health"
    BASE_URL = "https://tools.cdc.gov/api/v2/resources/media"

    def __init__(self, *args, page_size: int = 50, media_type: Optional[str] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.page_size = max(1, min(page_size, 100))
        self.media_type = media_type

    async def fetch_term(self, client: httpx.AsyncClient, term: str, limit: int) -> Iterable[RawDoc]:
        params = {
            "format": "json",
            "max": min(limit, self.page_size),
            "keyword": term,
            "sort": "Date",
        }
        if self.media_type:
            params["mediaType"] = self.media_type
        data = await self._get_json(client, self.BASE_URL, params=params)
        results = data.get("results") or []
        docs: List[RawDoc] = []
        for item in results:
            title = item.get("title") or term
            description = _strip_html(item.get("description"))
            body = _strip_html(item.get("body"))
            combined = "\n\n".join([part for part in [description, body] if part]) or None
            docs.append(
                RawDoc(
                    id=f"cdc:{item.get('id') or title}",
                    title=title,
                    text=combined,
                    meta={
                        "type": "public_health",
                        "term": term,
                        "topics": item.get("topics"),
                    },
                    prov=Provenance(
                        source="cdc_health",
                        source_url=item.get("url") or item.get("sourceUrl"),
                        license="CDC API Terms of Service",
                        retrieved_at=_now_iso(),
                        hash=None,
                    ),
                )
            )
            if len(docs) >= limit:
                break
        return docs


@register("nih_wellness")
class NIHWellnessScraper(TermScraper):
    name = "nih_wellness"
    BASE_URL = "https://health.gov/myhealthfinder/api/v3/topicsearch.json"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    async def fetch_term(self, client: httpx.AsyncClient, term: str, limit: int) -> Iterable[RawDoc]:
        params = {"keyword": term}
        data = await self._get_json(client, self.BASE_URL, params=params)
        resources = (((data.get("Result") or {}).get("Resources") or {}).get("Resource")) or []
        if isinstance(resources, dict):
            resources = [resources]
        docs: List[RawDoc] = []
        for resource in resources:
            title = resource.get("Title") or term
            sections = resource.get("Sections") or {}
            section_texts: List[str] = []
            for section in sections.get("section", []):  # type: ignore[attr-defined]
                if isinstance(section, dict):
                    content = section.get("Content")
                    if isinstance(content, list):
                        for entry in content:
                            if isinstance(entry, dict):
                                txt = entry.get("Text")
                                if txt:
                                    section_texts.append(_strip_html(txt) or "")
                    elif isinstance(content, dict):
                        txt = content.get("Text")
                        if txt:
                            section_texts.append(_strip_html(txt) or "")
            text = "\n".join([t for t in section_texts if t]) or None
            url = resource.get("AccessibleVersion") or resource.get("Url")
            docs.append(
                RawDoc(
                    id=f"nihwellness:{resource.get('Id') or title}",
                    title=title,
                    text=text,
                    meta={
                        "type": "wellness",
                        "term": term,
                        "audience": resource.get("Audience"),
                    },
                    prov=Provenance(
                        source="nih_wellness",
                        source_url=url,
                        license="Health.gov MyHealthfinder Terms",
                        retrieved_at=_now_iso(),
                        hash=None,
                    ),
                )
            )
            if len(docs) >= limit:
                break
        return docs


@register("semantic_scholar")
class SemanticScholarScraper(TermScraper):
    name = "semantic_scholar"
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self, *args, page_size: int = 50, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.page_size = max(1, min(page_size, 100))

    async def fetch_term(self, client: httpx.AsyncClient, term: str, limit: int) -> Iterable[RawDoc]:
        params = {
            "query": term,
            "limit": min(limit, self.page_size),
            "fields": "title,abstract,url,venue,year,authors,journal",
        }
        data = await self._get_json(client, self.BASE_URL, params=params)
        results = data.get("data") or []
        docs: List[RawDoc] = []
        for item in results:
            title = item.get("title") or term
            abstract = item.get("abstract")
            venue = item.get("venue") or item.get("journal", {}).get("name") if isinstance(item.get("journal"), dict) else None
            authors = ", ".join([a.get("name") for a in item.get("authors", []) if a.get("name")])
            text_parts = [abstract, f"Venue: {venue}" if venue else None]
            if authors:
                text_parts.append(f"Authors: {authors}")
            text = "\n".join([p for p in text_parts if p]) or None
            docs.append(
                RawDoc(
                    id=f"semanticscholar:{item.get('paperId') or title}",
                    title=title,
                    text=text,
                    meta={
                        "type": "academic_paper",
                        "term": term,
                        "paper_id": item.get("paperId"),
                        "year": item.get("year"),
                    },
                    prov=Provenance(
                        source="semantic_scholar",
                        source_url=item.get("url"),
                        license="Semantic Scholar API Terms",
                        retrieved_at=_now_iso(),
                        hash=None,
                    ),
                )
            )
            if len(docs) >= limit:
                break
        return docs


class EuropePMCScraper(TermScraper):
    BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    source_filter: Optional[str] = None
    doc_type: str = "article"

    def __init__(self, *args, page_size: int = 50, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.page_size = max(1, min(page_size, 1000))

    async def fetch_term(self, client: httpx.AsyncClient, term: str, limit: int) -> Iterable[RawDoc]:
        query = term
        if self.source_filter:
            query = f"{term} AND SRC:{self.source_filter}"
        params = {
            "query": query,
            "format": "json",
            "pageSize": min(limit, self.page_size),
        }
        data = await self._get_json(client, self.BASE_URL, params=params)
        results = ((data.get("resultList") or {}).get("result")) or []
        docs: List[RawDoc] = []
        for item in results:
            title = item.get("title") or term
            abstract = item.get("abstractText")
            journal = item.get("journalTitle")
            year = item.get("pubYear")
            doi = item.get("doi")
            src = item.get("source")
            text_parts = [abstract, f"Journal: {journal}" if journal else None, f"Year: {year}" if year else None]
            text = "\n".join([p for p in text_parts if p]) or None
            docs.append(
                RawDoc(
                    id=f"europepmc:{item.get('id') or title}",
                    title=title,
                    text=text,
                    meta={
                        "type": self.doc_type,
                        "term": term,
                        "source": src,
                        "doi": doi,
                    },
                    prov=Provenance(
                        source="europe_pmc",
                        source_url=item.get("fullTextUrlList", {}).get("fullTextUrl", [{}])[0].get("url")
                        if isinstance(item.get("fullTextUrlList"), dict)
                        else item.get("pageInfoUrl"),
                        license="Europe PMC terms",
                        retrieved_at=_now_iso(),
                        hash=None,
                    ),
                )
            )
            if len(docs) >= limit:
                break
        return docs


@register("europe_pmc")
class EuropePMCGeneralScraper(EuropePMCScraper):
    name = "europe_pmc"
    source_filter = None
    doc_type = "article"


@register("europe_pmc_biorxiv")
class EuropePMCBioRxivScraper(EuropePMCScraper):
    name = "europe_pmc_biorxiv"
    source_filter = "BIORxiv"
    doc_type = "preprint"


@register("europe_pmc_plos")
class EuropePMCPLOSScraper(EuropePMCScraper):
    name = "europe_pmc_plos"
    source_filter = "PLOS"
    doc_type = "article"


class FeedScraper(BaseScraper):
    """Generic RSS/Atom feed scraper."""

    doc_type: str = "feed"

    def __init__(
        self,
        *args,
        feeds: Optional[List[str]] = None,
        max_items_per_feed: int = 40,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.feeds = feeds or []
        self.max_items_per_feed = max(1, max_items_per_feed)

    def _parse_feed(self, xml_text: str) -> List[Dict[str, Any]]:
        root = etree.fromstring(xml_text.encode("utf-8"))
        items = root.findall(".//item")
        if not items:
            items = root.findall(".//{http://www.w3.org/2005/Atom}entry")
        entries: List[Dict[str, Any]] = []
        for el in items:
            title = _xml_text(el.find("title")) or _xml_text(el.find("{http://www.w3.org/2005/Atom}title"))
            description = _xml_text(el.find("description")) or _xml_text(el.find("{http://www.w3.org/2005/Atom}summary"))
            link_el = el.find("link") or el.find("{http://www.w3.org/2005/Atom}link")
            link = link_el.get("href") if link_el is not None else _xml_text(link_el)
            pub_date = _xml_text(el.find("pubDate")) or _xml_text(el.find("{http://www.w3.org/2005/Atom}updated"))
            entries.append({
                "title": title,
                "description": description,
                "link": link,
                "pub_date": pub_date,
            })
        return entries

    async def run(self) -> ScrapeResult:
        if not self.feeds:
            log.warning(f"[{self.name}] No feeds configured; nothing to scrape.")
            from ..utils.io import shard_writer

            write, close = shard_writer(self.shards_dir, self.name, self.shard_max_records)
            close()
            return ScrapeResult(0, self.shards_dir)

        from ..utils.io import shard_writer

        write, close = shard_writer(self.shards_dir, self.name, self.shard_max_records)
        total = 0

        async with await self._client() as client:
            for feed_url in self.feeds:
                if total >= self.max_docs:
                    break
                try:
                    text = await self._get_text(client, feed_url)
                except Exception as exc:  # noqa: BLE001
                    log.error(f"[{self.name}] feed '{feed_url}' failed: {exc}")
                    continue
                try:
                    entries = self._parse_feed(text)
                except Exception as exc:  # noqa: BLE001
                    log.error(f"[{self.name}] feed '{feed_url}' parse error: {exc}")
                    continue

                for entry in entries[: self.max_items_per_feed]:
                    write(
                        RawDoc(
                            id=f"{self.name}:{entry.get('link') or entry.get('title')}",
                            title=entry.get("title") or "",
                            text=_strip_html(entry.get("description")) or None,
                            meta={
                                "type": self.doc_type,
                                "feed_url": feed_url,
                                "published": entry.get("pub_date"),
                            },
                            prov=Provenance(
                                source=self.name,
                                source_url=entry.get("link"),
                                license="See publisher feed terms",
                                retrieved_at=_now_iso(),
                                hash=None,
                            ),
                        ).model_dump(mode="json")
                    )
                    total += 1
                    if total >= self.max_docs:
                        break

        close()
        return ScrapeResult(total, self.shards_dir, {"feeds": self.feeds})


@register("cme_list")
class CMEFeedScraper(FeedScraper):
    name = "cme_list"
    doc_type = "cme"


@register("acc_guidelines")
class ACCGuidelinesScraper(FeedScraper):
    name = "acc_guidelines"
    doc_type = "guideline"


@register("drugbank")
class DrugBankScraper(TermScraper):
    name = "drugbank"
    SEARCH_URL = "https://go.drugbank.com/unearth/q"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    async def fetch_term(self, client: httpx.AsyncClient, term: str, limit: int) -> Iterable[RawDoc]:
        params = {
            "searcher": "drugs",
            "query": term,
        }
        html_text = await self._get_text(client, self.SEARCH_URL, params=params)
        soup = BeautifulSoup(html_text, "html.parser")
        cards = soup.select("div.search-result") or soup.select("tbody tr")
        docs: List[RawDoc] = []
        for card in cards:
            link = card.find("a", href=True)
            if not link:
                continue
            href = link.get("href")
            if href and href.startswith("/"):
                url = f"https://go.drugbank.com{href}"
            else:
                url = href
            title = link.get_text(strip=True) or term
            summary_el = card.find("p") or card.find("td")
            summary = summary_el.get_text(" ", strip=True) if summary_el else None
            docs.append(
                RawDoc(
                    id=f"drugbank:{href or title}",
                    title=title,
                    text=summary,
                    meta={
                        "type": "drug_profile",
                        "term": term,
                    },
                    prov=Provenance(
                        source="drugbank",
                        source_url=url,
                        license="DrugBank public access terms",
                        retrieved_at=_now_iso(),
                        hash=None,
                    ),
                )
            )
            if len(docs) >= limit:
                break
        return docs


@register("rxnorm")
class RxNormScraper(TermScraper):
    name = "rxnorm"
    BASE_URL = "https://rxnav.nlm.nih.gov/REST"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    async def _fetch_properties(self, client: httpx.AsyncClient, rxcui: str) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/rxcui/{rxcui}/properties.json"
        try:
            data = await self._get_json(client, url)
        except Exception:  # noqa: BLE001 - best effort
            return {}
        return data.get("properties") or {}

    async def fetch_term(self, client: httpx.AsyncClient, term: str, limit: int) -> Iterable[RawDoc]:
        params = {
            "term": term,
            "maxEntries": limit,
        }
        data = await self._get_json(client, f"{self.BASE_URL}/approximateTerm.json", params=params)
        candidates = ((data.get("approximateGroup") or {}).get("candidate")) or []
        docs: List[RawDoc] = []
        if isinstance(candidates, dict):
            candidates = [candidates]
        for cand in candidates:
            rxcui = cand.get("rxcui")
            if not rxcui:
                continue
            props = await self._fetch_properties(client, rxcui)
            name = props.get("name") or cand.get("rxcui") or term
            synonym = props.get("synonym")
            tty = props.get("tty")
            text_parts = [f"Concept: {name}"]
            if synonym:
                text_parts.append(f"Synonym: {synonym}")
            if tty:
                text_parts.append(f"TTY: {tty}")
            if props.get("fullName"):
                text_parts.append(f"Full name: {props.get('fullName')}")
            text = "\n".join(text_parts)
            docs.append(
                RawDoc(
                    id=f"rxnorm:{rxcui}",
                    title=name,
                    text=text,
                    meta={
                        "type": "rxnorm_concept",
                        "term": term,
                        "rxcui": rxcui,
                        "score": cand.get("score"),
                    },
                    prov=Provenance(
                        source="rxnorm",
                        source_url=f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}",
                        license="U.S. NLM RxNorm terms",
                        retrieved_at=_now_iso(),
                        hash=None,
                    ),
                )
            )
            if len(docs) >= limit:
                break
        return docs
