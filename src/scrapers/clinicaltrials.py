from __future__ import annotations
import datetime as dt
from typing import Any, Dict, Iterable, List, Tuple, Optional
from .base_scraper import BaseScraper
from .registry import register
from ..utils.logger import get_logger
from ..utils.schemas import RawDoc, Provenance

log = get_logger("ctgov")

API = "https://clinicaltrials.gov/api/query/study_fields"

# ====== Built-in disease term list (edit/extend freely) ======
DISEASE_TERMS: List[str] = [
    # Common chronic
    "asthma", "chronic obstructive pulmonary disease", "copd",
    "diabetes mellitus", "hypertension", "hyperlipidemia",
    "ischemic heart disease", "coronary artery disease",
    "heart failure", "atrial fibrillation", "stroke",
    "chronic kidney disease", "ckd", "end stage renal disease",
    "nonalcoholic fatty liver disease", "nafld", "cirrhosis",
    # Infectious
    "covid-19", "influenza", "pneumonia", "tuberculosis",
    "hepatitis b", "hepatitis c", "hiv infection", "aids",
    # Neuro / Psych
    "migraine", "epilepsy", "multiple sclerosis",
    "parkinson disease", "alzheimer disease", "depression", "anxiety",
    "schizophrenia", "bipolar disorder",
    # Rheum / Pain
    "osteoarthritis", "rheumatoid arthritis", "psoriatic arthritis",
    "gout", "fibromyalgia", "lupus",
    # Oncology (high-level terms)
    "breast cancer", "lung cancer
