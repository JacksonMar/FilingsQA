"""HTTP client for SEC EDGAR: identifying User-Agent, rate limiting, retries."""

import logging
import time
from collections.abc import Callable
from typing import Any

import httpx

from filingsqa.edgar.rate_limit import RateLimiter

log = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class EdgarError(RuntimeError):
    pass


class EdgarClient:
    def __init__(
        self,
        user_agent: str,
        max_requests_per_second: float = 10,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if "@" not in user_agent:
            raise ValueError("SEC requires a User-Agent with a contact email")
        self._http = httpx.Client(
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=30.0,
            follow_redirects=True,
            transport=transport,
        )
        self._limiter = RateLimiter(max_requests_per_second, sleep=sleep)
        self._max_retries = max_retries
        self._backoff = backoff_seconds
        self._sleep = sleep

    def get(self, url: str) -> httpx.Response:
        """GET with rate limiting and exponential backoff on retryable failures."""
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            self._limiter.wait()
            try:
                response = self._http.get(url)
            except httpx.TransportError as exc:
                last_error = exc
            else:
                if response.status_code not in RETRYABLE_STATUS:
                    if response.is_error:
                        raise EdgarError(f"GET {url} -> HTTP {response.status_code}")
                    return response
                last_error = EdgarError(f"GET {url} -> HTTP {response.status_code}")
            if attempt < self._max_retries:
                delay = self._backoff * 2**attempt
                log.warning("Retrying %s in %.1fs (%s)", url, delay, last_error)
                self._sleep(delay)
        raise EdgarError(f"GET {url} failed after {self._max_retries + 1} attempts") from last_error

    def get_json(self, url: str) -> Any:
        return self.get(url).json()

    def get_text(self, url: str) -> str:
        return self.get(url).text

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "EdgarClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def submissions_url(cik10: str) -> str:
    return f"https://data.sec.gov/submissions/CIK{cik10}.json"


def companyfacts_url(cik10: str) -> str:
    return f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"


def document_url(cik: int, accession: str, primary_document: str) -> str:
    return (
        f"https://www.sec.gov/Archives/edgar/data/{cik}/"
        f"{accession.replace('-', '')}/{primary_document}"
    )
