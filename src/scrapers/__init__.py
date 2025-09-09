from .registry import register, get_scraper, list_scrapers  # re-export
# concrete scrapers import here so they're registered when module loads
from .dailymed import DailyMedSPLScraper  # noqa: F401
from .openfda_labels import OpenFDALabelsScraper  # noqa: F401
