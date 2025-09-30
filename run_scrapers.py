#!/usr/bin/env python3
"""
Comprehensive Medical Data Collection Script
Runs both modern and legacy scraper systems for maximum data coverage
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_banner():
    print("🏥" + "="*60 + "🏥")
    print("   ORPHAN MEDICAL AI - COMPREHENSIVE DATA COLLECTION")
    print("   NHS SAMD Compliant Medical Data Scraping System")
    print("🏥" + "="*60 + "🏥")
    print()

async def run_modern_scrapers():
    """Run modern async scrapers from /src/data/scrapers/"""
    print("🚀 Phase 1: Modern Async Scrapers")
    print("-" * 40)
    
    try:
        # Import modern scrapers
        from src.data.scrapers.medical.enhanced_scrapers import EnhancedMedicalScrapers
        from src.data.scrapers.kaggle.downloader import KaggleDatasetDownloader
        
        # 1. Kaggle Datasets
        print("📊 Downloading Kaggle medical datasets...")
        downloader = KaggleDatasetDownloader()
        kaggle_results = downloader.download_and_process_all()
        
        print(f"✅ Kaggle: {kaggle_results.get('successfully_processed', 0)} datasets processed")
        
        # 2. Live Medical Literature
        print("📚 Scraping live medical literature...")
        scraper = EnhancedMedicalScrapers()
        
        medical_topics = [
            "diabetes mellitus treatment guidelines",
            "hypertension management protocols", 
            "chest pain differential diagnosis",
            "respiratory infection treatment",
            "cardiovascular risk assessment",
            "mental health screening tools"
        ]
        
        total_articles = 0
        for topic in medical_topics:
            print(f"  📖 Scraping: {topic}")
            results = await scraper.scrape_all_sources(topic, max_results=200)
            topic_total = sum(len(articles) for articles in results.values())
            total_articles += topic_total
            print(f"    ✅ {topic_total} articles collected")
            
            # Rate limiting between topics
            await asyncio.sleep(2)
        
        print(f"✅ Modern scrapers: {total_articles} total articles")
        return {"kaggle_datasets": kaggle_results.get('successfully_processed', 0), 
                "live_articles": total_articles}
        
    except Exception as e:
        print(f"❌ Modern scrapers failed: {e}")
        return {"error": str(e)}

def run_legacy_scrapers(selected_sources: List[str] | None = None, dry_run: bool = False) -> Dict[str, Any]:
    """Run legacy/registry scrapers via src.scrapers.run."""
    print("\n🔧 Phase 2: Legacy Specialized Scrapers")
    print("-" * 40)

    config_path = "scrape_config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return {"error": "Config not found"}

    from src.scrapers.run import load_config, _run_registered_scraper, run_clinicaltrials, run_pubmed
    cfg = load_config(config_path)

    sources = selected_sources or list(cfg.get("sources", {}).keys())
    print(f"📋 Running configured medical data scraping ({len(sources)} sources)")

    results: Dict[str, int] = {}
    total = 0

    overrides = argparse.Namespace(
        dry_run=dry_run,
        page_size=None,
        max_pages=None,
        status_filter=None,
        only=None,
        max_terms=None,
        workers=2,
    )

    for src in sources:
        if src in ("clinicaltrials", "pubmed"):
            count = run_clinicaltrials(cfg.get("sources", {}), overrides) if src == "clinicaltrials" else run_pubmed(cfg.get("sources", {}), overrides)
        else:
            count = _run_registered_scraper(src, cfg, overrides)
        results[src] = count
        total += count
        print(f"   ✅ {src}: {count} documents")

    print(f"✅ Legacy scrapers total: {total}")
    return {"sources": results, "legacy_total": total}

def run_individual_scrapers(sample_terms: Dict[str, str] | None = None) -> Dict[str, Any]:
    """Run quick sample pulls for spotlight sources using the new registry scrapers."""
    print("\n🎯 Phase 3: Individual Spotlight Scrapes")
    print("-" * 40)

    sample_terms = sample_terms or {
        "openfda_labels": "diabetes",
        "dailymed_spls": "insulin",
        "pmc_open_access": "hypertension",
        "chembl": "metformin",
        "medlineplus": "asthma",
        "cdc_health": "vaccination",
        "semantic_scholar": "cardiology",
    }

    from src.scrapers.registry import get_scraper

    results: Dict[str, int] = {}

    async def _run_sample(name: str, term: str) -> int:
        ScraperCls = get_scraper(name)
        scraper = ScraperCls(max_docs=20, rate_limit_per_sec=1.0, terms=[term]) if hasattr(ScraperCls, "__init__") else None
        if scraper is None:
            return 0

        res = await scraper.run()
        return res.total_fetched

    async def _runner():
        total = 0
        for src, term in sample_terms.items():
            try:
                fetched = await _run_sample(src, term)
            except Exception as exc:  # noqa: BLE001
                print(f"   ❌ {src} failed: {exc}")
                fetched = 0
            else:
                print(f"   ✅ {src}: {fetched} sample docs for term '{term}'")
            results[src] = fetched
            total += fetched
        print(f"✅ Spotlight samples total: {total}")
        return total

    total = asyncio.run(_runner())
    results["spotlight_total"] = total
    return results

async def main():
    """Main execution function"""
    start_time = time.time()
    print_banner()
    # Phase 1: Modern Scrapers
    modern_results = await run_modern_scrapers()
    
    # Phase 2: Legacy Scrapers
    legacy_results = run_legacy_scrapers()
    
    # Phase 3: Individual Scrapers
    individual_results = run_individual_scrapers()
    
    # Summary
    print("\n🎉 SCRAPING COMPLETE!")
    print("=" * 50)
    
    total_data_points = 0
    
    if "kaggle_datasets" in modern_results:
        print(f"📊 Kaggle Datasets: {modern_results['kaggle_datasets']}")
        total_data_points += modern_results['kaggle_datasets'] * 1000  # Estimate records per dataset
    
    if "live_articles" in modern_results:
        print(f"📚 Live Articles: {modern_results['live_articles']}")
        total_data_points += modern_results['live_articles']
    
    if "legacy_total" in legacy_results:
        print(f"🔧 Legacy Articles: {legacy_results['legacy_total']}")
        total_data_points += legacy_results['legacy_total']
    
    individual_total = individual_results.get("spotlight_total", sum(v for k, v in individual_results.items() if k != "spotlight_total"))
    print(f"🎯 Individual Sources: {individual_total}")
    total_data_points += individual_total
    
    elapsed_time = time.time() - start_time
    print(f"\n📈 TOTAL DATA COLLECTED: ~{total_data_points:,} medical records")
    print(f"⏱️  Total Time: {elapsed_time:.1f} seconds")
    print(f"🚀 Ready for NHS SAMD compliant medical AI training!")
    
    # Create summary file
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "modern_results": modern_results,
        "legacy_results": legacy_results,
        "individual_results": individual_results,
        "total_estimated_records": total_data_points,
        "execution_time_seconds": elapsed_time
    }
    
    import json
    with open("scraping_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Summary saved to: scraping_summary.json")

if __name__ == "__main__":
    # Run the comprehensive scraping system
    asyncio.run(main())
