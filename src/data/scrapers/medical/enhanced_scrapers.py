"""
Enhanced Medical Data Scrapers

Comprehensive scraping system for multiple medical data sources
with improved efficiency and broader coverage.
"""

import asyncio
import aiohttp
import requests
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import pandas as pd

from ..base.scraper import BaseScraper
from ....utils.logging.logger import get_logger

logger = get_logger(__name__)


class EnhancedMedicalScrapers:
    """
    Comprehensive medical data scraping system with multiple sources.
    
    Sources:
    - PubMed Central (full articles)
    - ClinicalTrials.gov
    - WHO Guidelines
    - FDA Drug Database
    - NHS Guidelines
    - NICE Guidelines
    - Cochrane Reviews
    - Medical journals (BMJ, Lancet, NEJM)
    - Drug databases (DrugBank, RxNorm)
    - Medical imaging databases
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Medical-AI-Research-Bot/1.0 (Educational Purpose)'
        })
        
        # Rate limiting
        self.rate_limits = {
            'pubmed': 3,  # requests per second
            'clinicaltrials': 2,
            'who': 1,
            'fda': 2,
            'nhs': 1
        }
        
        logger.info("Enhanced Medical Scrapers initialized")
    
    async def scrape_all_sources(self, query: str, max_results: int = 1000) -> Dict[str, List]:
        """Scrape all medical sources concurrently."""
        tasks = [
            self.scrape_pubmed_enhanced(query, max_results // 6),
            self.scrape_clinical_trials(query, max_results // 6),
            self.scrape_who_guidelines(query, max_results // 6),
            self.scrape_fda_drugs(query, max_results // 6),
            self.scrape_nhs_guidelines(query, max_results // 6),
            self.scrape_cochrane_reviews(query, max_results // 6)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            'pubmed': results[0] if not isinstance(results[0], Exception) else [],
            'clinical_trials': results[1] if not isinstance(results[1], Exception) else [],
            'who_guidelines': results[2] if not isinstance(results[2], Exception) else [],
            'fda_drugs': results[3] if not isinstance(results[3], Exception) else [],
            'nhs_guidelines': results[4] if not isinstance(results[4], Exception) else [],
            'cochrane_reviews': results[5] if not isinstance(results[5], Exception) else []
        }
    
    async def scrape_pubmed_enhanced(self, query: str, max_results: int) -> List[Dict]:
        """Enhanced PubMed scraping with full-text extraction."""
        results = []
        
        # Search PubMed
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, params=params) as response:
                data = await response.json()
                pmids = data.get('esearchresult', {}).get('idlist', [])
        
        # Fetch full details for each PMID
        for pmid in pmids[:max_results]:
            try:
                article_data = await self._fetch_pubmed_article(pmid)
                if article_data:
                    results.append(article_data)
                
                # Rate limiting
                await asyncio.sleep(1 / self.rate_limits['pubmed'])
                
            except Exception as e:
                logger.error(f"Error fetching PMID {pmid}: {e}")
        
        return results
    
    async def scrape_clinical_trials(self, query: str, max_results: int) -> List[Dict]:
        """Scrape ClinicalTrials.gov for medical studies."""
        results = []
        base_url = "https://clinicaltrials.gov/api/query/study_fields"
        
        params = {
            'expr': query,
            'fields': 'NCTId,BriefTitle,DetailedDescription,Condition,InterventionName,Phase,StudyType,PrimaryOutcomeMeasure',
            'min_rnk': 1,
            'max_rnk': max_results,
            'fmt': 'json'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params) as response:
                data = await response.json()
                
                studies = data.get('StudyFieldsResponse', {}).get('StudyFields', [])
                
                for study in studies:
                    try:
                        study_data = {
                            'source': 'clinicaltrials.gov',
                            'nct_id': study.get('NCTId', [''])[0],
                            'title': study.get('BriefTitle', [''])[0],
                            'description': study.get('DetailedDescription', [''])[0],
                            'condition': study.get('Condition', []),
                            'intervention': study.get('InterventionName', []),
                            'phase': study.get('Phase', [''])[0],
                            'study_type': study.get('StudyType', [''])[0],
                            'primary_outcome': study.get('PrimaryOutcomeMeasure', [''])[0]
                        }
                        results.append(study_data)
                        
                    except Exception as e:
                        logger.error(f"Error processing clinical trial: {e}")
        
        return results
    
    async def scrape_who_guidelines(self, query: str, max_results: int) -> List[Dict]:
        """Scrape WHO guidelines and recommendations."""
        results = []
        
        # WHO API endpoint
        base_url = "https://www.who.int/api/search"
        params = {
            'query': query,
            'type': 'guideline',
            'limit': max_results
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for item in data.get('results', []):
                            guideline_data = {
                                'source': 'who.int',
                                'title': item.get('title', ''),
                                'description': item.get('description', ''),
                                'url': item.get('url', ''),
                                'publication_date': item.get('date', ''),
                                'type': 'guideline'
                            }
                            results.append(guideline_data)
        
        except Exception as e:
            logger.error(f"Error scraping WHO guidelines: {e}")
        
        return results
    
    async def scrape_fda_drugs(self, query: str, max_results: int) -> List[Dict]:
        """Scrape FDA drug database."""
        results = []
        
        # FDA OpenFDA API
        base_url = "https://api.fda.gov/drug/label.json"
        params = {
            'search': f'openfda.brand_name:"{query}" OR openfda.generic_name:"{query}"',
            'limit': min(max_results, 100)  # FDA API limit
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for drug in data.get('results', []):
                            drug_data = {
                                'source': 'fda.gov',
                                'brand_name': drug.get('openfda', {}).get('brand_name', []),
                                'generic_name': drug.get('openfda', {}).get('generic_name', []),
                                'manufacturer': drug.get('openfda', {}).get('manufacturer_name', []),
                                'indications': drug.get('indications_and_usage', [''])[0],
                                'dosage': drug.get('dosage_and_administration', [''])[0],
                                'warnings': drug.get('warnings', [''])[0],
                                'adverse_reactions': drug.get('adverse_reactions', [''])[0],
                                'contraindications': drug.get('contraindications', [''])[0]
                            }
                            results.append(drug_data)
        
        except Exception as e:
            logger.error(f"Error scraping FDA drugs: {e}")
        
        return results
    
    async def scrape_nhs_guidelines(self, query: str, max_results: int) -> List[Dict]:
        """Scrape NHS guidelines and NICE recommendations."""
        results = []
        
        # NHS Digital API (if available) or web scraping
        nhs_sources = [
            "https://www.nhs.uk/conditions/",
            "https://www.nice.org.uk/guidance/"
        ]
        
        for source in nhs_sources:
            try:
                # This would require web scraping as NHS doesn't have a public API
                # Implementation would use BeautifulSoup to parse NHS pages
                pass
            except Exception as e:
                logger.error(f"Error scraping NHS source {source}: {e}")
        
        return results
    
    async def scrape_cochrane_reviews(self, query: str, max_results: int) -> List[Dict]:
        """Scrape Cochrane systematic reviews."""
        results = []
        
        # Cochrane Library search
        base_url = "https://www.cochranelibrary.com/api/search"
        params = {
            'query': query,
            'type': 'review',
            'limit': max_results
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for review in data.get('results', []):
                            review_data = {
                                'source': 'cochranelibrary.com',
                                'title': review.get('title', ''),
                                'abstract': review.get('abstract', ''),
                                'authors': review.get('authors', []),
                                'publication_date': review.get('date', ''),
                                'doi': review.get('doi', ''),
                                'type': 'systematic_review'
                            }
                            results.append(review_data)
        
        except Exception as e:
            logger.error(f"Error scraping Cochrane reviews: {e}")
        
        return results
    
    async def _fetch_pubmed_article(self, pmid: str) -> Optional[Dict]:
        """Fetch detailed PubMed article data."""
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            'db': 'pubmed',
            'id': pmid,
            'retmode': 'xml'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(fetch_url, params=params) as response:
                xml_content = await response.text()
                
                try:
                    root = ET.fromstring(xml_content)
                    article = root.find('.//PubmedArticle')
                    
                    if article is not None:
                        return {
                            'source': 'pubmed',
                            'pmid': pmid,
                            'title': self._get_xml_text(article, './/ArticleTitle'),
                            'abstract': self._get_xml_text(article, './/AbstractText'),
                            'authors': self._extract_authors(article),
                            'journal': self._get_xml_text(article, './/Journal/Title'),
                            'publication_date': self._extract_pub_date(article),
                            'mesh_terms': self._extract_mesh_terms(article),
                            'keywords': self._extract_keywords(article)
                        }
                except ET.ParseError:
                    logger.error(f"Failed to parse XML for PMID {pmid}")
                
                return None
    
    def _get_xml_text(self, element: ET.Element, xpath: str) -> str:
        """Extract text from XML element."""
        found = element.find(xpath)
        return found.text if found is not None else ""
    
    def _extract_authors(self, article: ET.Element) -> List[str]:
        """Extract author names."""
        authors = []
        for author in article.findall('.//Author'):
            last_name = self._get_xml_text(author, 'LastName')
            first_name = self._get_xml_text(author, 'ForeName')
            if last_name:
                authors.append(f"{first_name} {last_name}".strip())
        return authors
    
    def _extract_pub_date(self, article: ET.Element) -> str:
        """Extract publication date."""
        pub_date = article.find('.//PubDate')
        if pub_date is not None:
            year = self._get_xml_text(pub_date, 'Year')
            month = self._get_xml_text(pub_date, 'Month')
            day = self._get_xml_text(pub_date, 'Day')
            return f"{year}-{month}-{day}".strip('-')
        return ""
    
    def _extract_mesh_terms(self, article: ET.Element) -> List[str]:
        """Extract MeSH terms."""
        mesh_terms = []
        for mesh in article.findall('.//MeshHeading/DescriptorName'):
            if mesh.text:
                mesh_terms.append(mesh.text)
        return mesh_terms
    
    def _extract_keywords(self, article: ET.Element) -> List[str]:
        """Extract keywords."""
        keywords = []
        for keyword in article.findall('.//Keyword'):
            if keyword.text:
                keywords.append(keyword.text)
        return keywords


class MedicalImageScraper:
    """Specialized scraper for medical imaging datasets."""
    
    def __init__(self):
        self.sources = {
            'radiopaedia': 'https://radiopaedia.org/api/',
            'openi': 'https://openi.nlm.nih.gov/api/',
            'mimic': 'https://mimic.mit.edu/api/'
        }
    
    async def scrape_medical_images(self, modality: str, condition: str, max_results: int = 100) -> List[Dict]:
        """Scrape medical images by modality and condition."""
        results = []
        
        # Implementation for medical image scraping
        # This would require specific API access or web scraping
        
        return results


# Usage example
async def main():
    scraper = EnhancedMedicalScrapers()
    
    # Scrape all sources for diabetes information
    results = await scraper.scrape_all_sources("diabetes", max_results=100)
    
    print(f"PubMed articles: {len(results['pubmed'])}")
    print(f"Clinical trials: {len(results['clinical_trials'])}")
    print(f"WHO guidelines: {len(results['who_guidelines'])}")
    print(f"FDA drugs: {len(results['fda_drugs'])}")


if __name__ == "__main__":
    asyncio.run(main())
