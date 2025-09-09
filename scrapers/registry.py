from __future__ import annotations
from typing import Dict, Type
from .base_scraper import BaseScraper

_REGISTRY: Dict[str, Type[BaseScraper]] = {}

def register(name: str):
    def _wrap(cls):
        _REGISTRY[name] = cls
        return cls
    return _wrap

def get_scraper(name: str) -> Type[BaseScraper]:
    return _REGISTRY[name]

def list_scrapers():
    return list(_REGISTRY.keys())
