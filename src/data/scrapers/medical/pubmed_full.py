"""
Enhanced PubMed Full-Text Article Scraper

Scrapes complete articles from PubMed Central (PMC) and other sources,
not just abstracts. Includes advanced parsing and content extraction.
"""

import os
import time
import requests
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import json
import re
from urllib.parse import urljoin, urlparse
import logging

from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from ..base.scraper import BaseScraper
from ....utils.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FullTextArticle:
    """Full-text article data structure."""
    pmid: str
    pmcid: Optional[str]
    doi: Optional[str]
    title: str
    authors: List[str]
    journal: str
    publication_date: str
    abstract: str
    full_text: str
    sections: Dict[str, str]  # section_name -> content
    references: List[Dict[str, str]]
    keywords: List[str]
    mesh_terms: List[str]
    figures: List[Dict[str, str]]
    tables: List[Dict[str, str]]
    supplementary_data: List[Dict[str, str]]
    metadata: Dict[str, Any]


class PubMedFullTextScraper(BaseScraper):
    """
    Enhanced PubMed scraper for full-text articles.
    
    Features:
    - PMC full-text extraction
    - Publisher website scraping
    - PDF text extraction
    - Structured content parsing
    - Reference extraction
    - Figure and table extraction
    - Multi-source fallback
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
        # API endpoints
        self.eutils_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.pmc_base = "https://www.ncbi.nlm.nih.gov/pmc/articles/"
        self.pubmed_base = "https://pubmed.ncbi.nlm.nih.gov/"
        
        # Initialize selenium driver for dynamic content
        self.driver = None
        self._setup_selenium()
        
        # Content extraction patterns
        self.section_patterns = self._initialize_section_patterns()
        
        # Publisher-specific extractors
        self.publisher_extractors = self._initialize_publisher_extractors()
        
        logger.info("PubMed Full-Text Scraper initialized")
    
    def _setup_selenium(self):
        """Setup Selenium WebDriver for dynamic content."""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            
            self.driver = webdriver.Chrome(options=chrome_options)
            logger.info("Selenium WebDriver initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Selenium: {e}")
            self.driver = None
    
    def _initialize_section_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for identifying article sections."""
        return {
            "introduction": [
                r"introduction", r"background", r"rationale", r"objective"
            ],
            "methods": [
                r"methods?", r"methodology", r"materials?\s+and\s+methods?",
                r"experimental\s+(?:design|procedure)", r"study\s+design"
            ],
            "results": [
                r"results?", r"findings", r"outcomes?"
            ],
            "discussion": [
                r"discussion", r"interpretation", r"implications?"
            ],
            "conclusion": [
                r"conclusions?", r"summary", r"final\s+remarks?"
            ],
            "references": [
                r"references?", r"bibliography", r"citations?"
            ]
        }
    
    def _initialize_publisher_extractors(self) -> Dict[str, callable]:
        """Initialize publisher-specific extraction methods."""
        return {
            "nature.com": self._extract_nature_article,
            "sciencedirect.com": self._extract_sciencedirect_article,
            "springer.com": self._extract_springer_article,
            "wiley.com": self._extract_wiley_article,
            "bmj.com": self._extract_bmj_article,
            "nejm.org": self._extract_nejm_article,
            "thelancet.com": self._extract_lancet_article
        }
    
    def search_and_extract_articles(
        self,
        query: str,
        max_results: int = 100,
        full_text_only: bool = True
    ) -> List[FullTextArticle]:
        """
        Search PubMed and extract full-text articles.
        
        Args:
            query: Search query
            max_results: Maximum number of articles to extract
            full_text_only: Only return articles with full text available
            
        Returns:
            List of full-text articles
        """
        logger.info(f"Searching PubMed for: {query}")
        
        # Search PubMed for PMIDs
        pmids = self._search_pubmed(query, max_results)
        logger.info(f"Found {len(pmids)} articles")
        
        # Extract full-text articles
        articles = []
        for i, pmid in enumerate(pmids):
            try:
                logger.info(f"Processing article {i+1}/{len(pmids)}: {pmid}")
                article = self._extract_full_article(pmid)
                
                if article and (not full_text_only or article.full_text):
                    articles.append(article)
                    logger.info(f"Successfully extracted: {article.title[:100]}...")
                else:
                    logger.warning(f"No full text available for PMID: {pmid}")
                
                # Rate limiting
                time.sleep(self.config.get("request_delay", 1))
                
            except Exception as e:
                logger.error(f"Failed to extract article {pmid}: {e}")
                continue
        
        logger.info(f"Successfully extracted {len(articles)} full-text articles")
        return articles
    
    def _search_pubmed(self, query: str, max_results: int) -> List[str]:
        """Search PubMed and return PMIDs."""
        search_url = f"{self.eutils_base}esearch.fcgi"
        
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance"
        }
        
        response = self.session.get(search_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        return data.get("esearchresult", {}).get("idlist", [])
    
    def _extract_full_article(self, pmid: str) -> Optional[FullTextArticle]:
        """Extract full article content for a given PMID."""
        # Get basic article info from PubMed
        article_info = self._get_pubmed_info(pmid)
        if not article_info:
            return None
        
        # Try to get full text from multiple sources
        full_text_content = None
        
        # 1. Try PMC if available
        if article_info.get("pmcid"):
            full_text_content = self._extract_pmc_full_text(article_info["pmcid"])
        
        # 2. Try publisher website
        if not full_text_content and article_info.get("doi"):
            full_text_content = self._extract_publisher_full_text(article_info["doi"])
        
        # 3. Try alternative sources
        if not full_text_content:
            full_text_content = self._extract_alternative_sources(pmid, article_info)
        
        # Create article object
        if full_text_content:
            return self._create_article_object(pmid, article_info, full_text_content)
        
        return None
    
    def _get_pubmed_info(self, pmid: str) -> Optional[Dict]:
        """Get basic article information from PubMed."""
        fetch_url = f"{self.eutils_base}efetch.fcgi"
        
        params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml"
        }
        
        response = self.session.get(fetch_url, params=params)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.content)
        article = root.find(".//PubmedArticle")
        
        if article is None:
            return None
        
        # Extract basic information
        info = {
            "pmid": pmid,
            "title": self._get_xml_text(article, ".//ArticleTitle"),
            "abstract": self._get_xml_text(article, ".//AbstractText"),
            "journal": self._get_xml_text(article, ".//Journal/Title"),
            "authors": self._extract_authors(article),
            "publication_date": self._extract_publication_date(article),
            "doi": self._get_xml_text(article, ".//ELocationID[@EIdType='doi']"),
            "pmcid": self._get_xml_text(article, ".//ArticleId[@IdType='pmc']"),
            "keywords": self._extract_keywords(article),
            "mesh_terms": self._extract_mesh_terms(article)
        }
        
        return info
    
    def _extract_pmc_full_text(self, pmcid: str) -> Optional[Dict]:
        """Extract full text from PMC."""
        # Clean PMC ID
        pmcid = pmcid.replace("PMC", "")
        
        # Get PMC XML
        pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/?report=xml"
        
        try:
            response = self.session.get(pmc_url)
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.content)
            
            # Extract structured content
            content = {
                "full_text": self._extract_pmc_text(root),
                "sections": self._extract_pmc_sections(root),
                "references": self._extract_pmc_references(root),
                "figures": self._extract_pmc_figures(root),
                "tables": self._extract_pmc_tables(root)
            }
            
            return content
            
        except Exception as e:
            logger.warning(f"Failed to extract PMC full text for {pmcid}: {e}")
            return None
    
    def _extract_publisher_full_text(self, doi: str) -> Optional[Dict]:
        """Extract full text from publisher website."""
        # Resolve DOI to publisher URL
        doi_url = f"https://doi.org/{doi}"
        
        try:
            response = self.session.get(doi_url, allow_redirects=True)
            final_url = response.url
            
            # Identify publisher
            domain = urlparse(final_url).netloc.lower()
            
            # Use publisher-specific extractor
            for publisher_domain, extractor in self.publisher_extractors.items():
                if publisher_domain in domain:
                    return extractor(final_url)
            
            # Generic extraction if no specific extractor
            return self._extract_generic_article(final_url)
            
        except Exception as e:
            logger.warning(f"Failed to extract publisher full text for DOI {doi}: {e}")
            return None
    
    def _extract_nature_article(self, url: str) -> Optional[Dict]:
        """Extract article from Nature website."""
        if not self.driver:
            return None
        
        try:
            self.driver.get(url)
            
            # Wait for content to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "c-article-body"))
            )
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            # Extract content
            content = {
                "full_text": self._extract_text_from_soup(soup, ".c-article-body"),
                "sections": self._extract_sections_from_soup(soup),
                "references": self._extract_references_from_soup(soup),
                "figures": self._extract_figures_from_soup(soup),
                "tables": self._extract_tables_from_soup(soup)
            }
            
            return content
            
        except Exception as e:
            logger.warning(f"Failed to extract Nature article: {e}")
            return None
    
    def _extract_sciencedirect_article(self, url: str) -> Optional[Dict]:
        """Extract article from ScienceDirect."""
        # Implementation for ScienceDirect extraction
        return self._extract_generic_article(url)
    
    def _extract_springer_article(self, url: str) -> Optional[Dict]:
        """Extract article from Springer."""
        return self._extract_generic_article(url)
    
    def _extract_wiley_article(self, url: str) -> Optional[Dict]:
        """Extract article from Wiley."""
        return self._extract_generic_article(url)
    
    def _extract_bmj_article(self, url: str) -> Optional[Dict]:
        """Extract article from BMJ."""
        return self._extract_generic_article(url)
    
    def _extract_nejm_article(self, url: str) -> Optional[Dict]:
        """Extract article from NEJM."""
        return self._extract_generic_article(url)
    
    def _extract_lancet_article(self, url: str) -> Optional[Dict]:
        """Extract article from The Lancet."""
        return self._extract_generic_article(url)
    
    def _extract_generic_article(self, url: str) -> Optional[Dict]:
        """Generic article extraction for unknown publishers."""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                element.decompose()
            
            # Extract text content
            full_text = soup.get_text(separator="\n", strip=True)
            
            content = {
                "full_text": full_text,
                "sections": self._extract_sections_from_text(full_text),
                "references": [],
                "figures": [],
                "tables": []
            }
            
            return content
            
        except Exception as e:
            logger.warning(f"Failed generic extraction for {url}: {e}")
            return None
    
    def _extract_alternative_sources(self, pmid: str, article_info: Dict) -> Optional[Dict]:
        """Try alternative sources for full text."""
        # Try Semantic Scholar, arXiv, bioRxiv, etc.
        # This would implement additional source checking
        return None
    
    def _create_article_object(
        self,
        pmid: str,
        article_info: Dict,
        full_text_content: Dict
    ) -> FullTextArticle:
        """Create FullTextArticle object from extracted data."""
        return FullTextArticle(
            pmid=pmid,
            pmcid=article_info.get("pmcid"),
            doi=article_info.get("doi"),
            title=article_info.get("title", ""),
            authors=article_info.get("authors", []),
            journal=article_info.get("journal", ""),
            publication_date=article_info.get("publication_date", ""),
            abstract=article_info.get("abstract", ""),
            full_text=full_text_content.get("full_text", ""),
            sections=full_text_content.get("sections", {}),
            references=full_text_content.get("references", []),
            keywords=article_info.get("keywords", []),
            mesh_terms=article_info.get("mesh_terms", []),
            figures=full_text_content.get("figures", []),
            tables=full_text_content.get("tables", []),
            supplementary_data=[],
            metadata={
                "extraction_method": "enhanced_scraper",
                "extraction_timestamp": time.time(),
                "source_url": full_text_content.get("source_url", "")
            }
        )
    
    def _extract_pmc_text(self, root: ET.Element) -> str:
        """Extract full text from PMC XML."""
        text_parts = []
        
        # Extract from body
        body = root.find(".//body")
        if body is not None:
            text_parts.append(ET.tostring(body, encoding='unicode', method='text'))
        
        return "\n".join(text_parts)
    
    def _extract_pmc_sections(self, root: ET.Element) -> Dict[str, str]:
        """Extract sections from PMC XML."""
        sections = {}
        
        for sec in root.findall(".//sec"):
            title_elem = sec.find("title")
            if title_elem is not None:
                title = title_elem.text or ""
                content = ET.tostring(sec, encoding='unicode', method='text')
                sections[title.lower()] = content
        
        return sections
    
    def _extract_pmc_references(self, root: ET.Element) -> List[Dict[str, str]]:
        """Extract references from PMC XML."""
        references = []
        
        for ref in root.findall(".//ref"):
            ref_data = {
                "id": ref.get("id", ""),
                "text": ET.tostring(ref, encoding='unicode', method='text')
            }
            references.append(ref_data)
        
        return references
    
    def _extract_pmc_figures(self, root: ET.Element) -> List[Dict[str, str]]:
        """Extract figures from PMC XML."""
        figures = []
        
        for fig in root.findall(".//fig"):
            fig_data = {
                "id": fig.get("id", ""),
                "caption": self._get_xml_text(fig, "caption"),
                "label": self._get_xml_text(fig, "label")
            }
            figures.append(fig_data)
        
        return figures
    
    def _extract_pmc_tables(self, root: ET.Element) -> List[Dict[str, str]]:
        """Extract tables from PMC XML."""
        tables = []
        
        for table in root.findall(".//table-wrap"):
            table_data = {
                "id": table.get("id", ""),
                "caption": self._get_xml_text(table, "caption"),
                "label": self._get_xml_text(table, "label")
            }
            tables.append(table_data)
        
        return tables
    
    def _get_xml_text(self, element: ET.Element, xpath: str) -> str:
        """Get text content from XML element."""
        found = element.find(xpath)
        return found.text if found is not None else ""
    
    def _extract_authors(self, article: ET.Element) -> List[str]:
        """Extract author names from PubMed XML."""
        authors = []
        
        for author in article.findall(".//Author"):
            last_name = self._get_xml_text(author, "LastName")
            first_name = self._get_xml_text(author, "ForeName")
            
            if last_name:
                full_name = f"{first_name} {last_name}".strip()
                authors.append(full_name)
        
        return authors
    
    def _extract_publication_date(self, article: ET.Element) -> str:
        """Extract publication date from PubMed XML."""
        pub_date = article.find(".//PubDate")
        if pub_date is not None:
            year = self._get_xml_text(pub_date, "Year")
            month = self._get_xml_text(pub_date, "Month")
            day = self._get_xml_text(pub_date, "Day")
            
            return f"{year}-{month}-{day}".strip("-")
        
        return ""
    
    def _extract_keywords(self, article: ET.Element) -> List[str]:
        """Extract keywords from PubMed XML."""
        keywords = []
        
        for keyword in article.findall(".//Keyword"):
            if keyword.text:
                keywords.append(keyword.text)
        
        return keywords
    
    def _extract_mesh_terms(self, article: ET.Element) -> List[str]:
        """Extract MeSH terms from PubMed XML."""
        mesh_terms = []
        
        for mesh in article.findall(".//MeshHeading/DescriptorName"):
            if mesh.text:
                mesh_terms.append(mesh.text)
        
        return mesh_terms
    
    def save_articles(self, articles: List[FullTextArticle], output_dir: str):
        """Save extracted articles to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for article in articles:
            # Save as JSON
            article_data = {
                "pmid": article.pmid,
                "pmcid": article.pmcid,
                "doi": article.doi,
                "title": article.title,
                "authors": article.authors,
                "journal": article.journal,
                "publication_date": article.publication_date,
                "abstract": article.abstract,
                "full_text": article.full_text,
                "sections": article.sections,
                "references": article.references,
                "keywords": article.keywords,
                "mesh_terms": article.mesh_terms,
                "figures": article.figures,
                "tables": article.tables,
                "metadata": article.metadata
            }
            
            filename = f"pmid_{article.pmid}.json"
            with open(output_path / filename, 'w', encoding='utf-8') as f:
                json.dump(article_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(articles)} articles to {output_dir}")
    
    def __del__(self):
        """Cleanup Selenium driver."""
        if self.driver:
            self.driver.quit()
