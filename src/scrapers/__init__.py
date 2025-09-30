# src/scrapers/__init__.py
"""
Lightweight init for the scrapers package.

Avoid importing submodules here to prevent import-time failures
when optional dependencies or legacy symbols are missing.
Access submodules directly, e.g.:
  from src.scrapers.ctgov import fetch_term as ctgov_fetch_term
  from src.scrapers.pubmed import fetch_term as pubmed_fetch_term
"""

__all__ = []  # keep empty; no side-effect imports

# register extended scrapers on import (safe side effect)
from . import extended_sources  # noqa: F401
