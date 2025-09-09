from .registry import register, get_scraper, list_scrapers  # re-export
# Keep existing imports
from .dailymed import DailyMedSPLScraper  # noqa: F401
from .openfda_labels import OpenFDALabelsScraper  # noqa: F401

# NEW:
from .pubmed import PubMedAbstractsScraper  # noqa: F401
from .pmc_oa import PMCOpenAccessScraper     # noqa: F401
