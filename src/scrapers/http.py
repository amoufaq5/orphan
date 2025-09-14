# src/scrapers/http.py
import time, random
from typing import Dict, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_DEFAULT_UA = "orphan-scraper/1.0 (+https://example.org) requests"

def make_session(timeout: float = 30.0, max_retries: int = 5) -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        backoff_factor=0.6,             # exponential backoff
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD", "OPTIONS"),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": _DEFAULT_UA, "Accept": "application/json"})
    s.request_timeout = timeout
    return s

def backoff_sleep(min_s=0.25, max_s=0.45):
    # polite ~3 req/sec avg (add jitter)
    time.sleep(random.uniform(min_s, max_s))

def get_json(session: requests.Session, url: str, params: Dict, timeout: Optional[float] = None) -> dict:
    backoff_sleep()
    resp = session.get(url, params=params, timeout=timeout or session.request_timeout)
    # If server returned an error after retries, surface details
    if resp.status_code >= 400:
        # Try to include response body in logs for debugging upstream errors
        snippet = resp.text[:500] if resp.text else ""
        raise requests.HTTPError(f"HTTP {resp.status_code} for {resp.url}\nBody: {snippet}")
    return resp.json()
