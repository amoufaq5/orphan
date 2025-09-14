# src/scrapers/http.py
import os
import time
import random
from typing import Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---- Configurable pacing via env vars (good when using --workers) ----
_MIN_SLEEP = float(os.getenv("ORPH_HTTP_MIN_SLEEP", "0.25"))  # seconds
_MAX_SLEEP = float(os.getenv("ORPH_HTTP_MAX_SLEEP", "0.45"))  # seconds

_DEFAULT_UA = os.getenv(
    "ORPH_HTTP_UA",
    "orphan-scraper/1.0 (+https://github.com/your-org) requests",
)

_STATUS_FORCELIST = (429, 500, 502, 503, 504)


def _polite_sleep() -> None:
    """Sleep a small, jittered interval to stay polite with public APIs."""
    low = min(_MIN_SLEEP, _MAX_SLEEP)
    high = max(_MIN_SLEEP, _MAX_SLEEP)
    if high > 0:
        time.sleep(random.uniform(low, high))


def make_session(timeout: float = 30.0, max_retries: int = 5) -> requests.Session:
    """
    Create a Requests session with robust retries and connection pooling.
    - Retries on 429/5xx with exponential backoff (honors Retry-After).
    - Small connection pool to keep things efficient under light concurrency.
    """
    s = requests.Session()

    retries = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        status=max_retries,
        backoff_factor=0.7,                      # exponential backoff (0.7, 1.4, 2.8, …)
        status_forcelist=_STATUS_FORCELIST,
        allowed_methods=("GET", "HEAD", "OPTIONS"),
        raise_on_status=False,
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(
        max_retries=retries,
        pool_connections=20,
        pool_maxsize=20,
    )

    s.mount("https://", adapter)
    s.mount("http://", adapter)

    # Sensible defaults
    s.headers.update({
        "User-Agent": _DEFAULT_UA,
        "Accept": "*/*",
    })

    # Stash default timeout on the session for helpers to use
    s.request_timeout = timeout
    return s


def _request(
    session: requests.Session,
    url: str,
    params: Optional[Dict] = None,
    timeout: Optional[float] = None,
) -> requests.Response:
    """
    Internal GET with polite sleep + explicit HTTP error surfacing.
    """
    _polite_sleep()
    resp = session.get(url, params=params or {}, timeout=timeout or session.request_timeout)
    if resp.status_code >= 400:
        # Surface a concise, useful error (include a snippet of the body)
        body = (resp.text or "")[:600]
        raise requests.HTTPError(f"HTTP {resp.status_code} for {resp.url}\n{body}")
    return resp


def get_json(
    session: requests.Session,
    url: str,
    params: Optional[Dict] = None,
    timeout: Optional[float] = None,
) -> Dict:
    """
    GET JSON with retries/jitter; raises on HTTP errors or invalid JSON.
    """
    resp = _request(session, url, params=params, timeout=timeout)
    return resp.json()


def get_text(
    session: requests.Session,
    url: str,
    params: Optional[Dict] = None,
    timeout: Optional[float] = None,
) -> str:
    """
    GET text with retries/jitter; raises on HTTP errors.
    """
    resp = _request(session, url, params=params, timeout=timeout)
    # Respect encoding if server sets it; Requests usually guesses correctly.
    return resp.text


def get_bytes(
    session: requests.Session,
    url: str,
    params: Optional[Dict] = None,
    timeout: Optional[float] = None,
) -> bytes:
    """
    GET raw bytes (e.g., PDFs) with retries/jitter; raises on HTTP errors.
    """
    resp = _request(session, url, params=params, timeout=timeout)
    return resp.content
