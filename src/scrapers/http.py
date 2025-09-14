# src/scrapers/http.py (only the sleep helper changed)
import os, time, random
from typing import Dict, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_DEFAULT_UA = "orphan-scraper/1.0 (+https://github.com/your-org) requests"

_MIN = float(os.getenv("ORPH_HTTP_MIN_SLEEP", "0.25"))
_MAX = float(os.getenv("ORPH_HTTP_MAX_SLEEP", "0.45"))

def make_session(timeout: float = 30.0, max_retries: int = 5) -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        backoff_factor=0.7,
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

def get_json(session: requests.Session, url: str, params: Dict, timeout: Optional[float] = None) -> dict:
    time.sleep(random.uniform(_MIN, _MAX))
    resp = session.get(url, params=params, timeout=timeout or session.request_timeout)
    if resp.status_code >= 400:
        body = (resp.text or "")[:600]
        raise requests.HTTPError(f"HTTP {resp.status_code} for {resp.url}\n{body}")
    return resp.json()
