"""Cached, rate-limited HTTP fetching shared by all three ingestion stages.

Every remote read in this project goes through `fetch_bytes`. That gives us one
place to enforce the on-disk cache, the inter-request delay, and the retry
policy -- which matters because Understat is scraped rather than served through
an API, and we do not want to hammer it while iterating on the pipeline.
"""

from __future__ import annotations

import time
from pathlib import Path

import requests

from src.config import PARAMS, USER_AGENT, get_logger

log = get_logger("fetch")

_INGEST = PARAMS["ingest"]
_DELAY_S: float = _INGEST["request_delay_s"]
_TIMEOUT_S: int = _INGEST["timeout_s"]
_MAX_RETRIES: int = _INGEST["max_retries"]
_USE_CACHE: bool = _INGEST["use_cache"]

# Timestamp of the last outbound request, so the delay applies across calls
# rather than only within a single loop.
_last_request_at: float = 0.0


def _throttle() -> None:
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < _DELAY_S:
        time.sleep(_DELAY_S - elapsed)
    _last_request_at = time.monotonic()


def fetch_bytes(
    url: str,
    cache_path: Path,
    *,
    use_cache: bool | None = None,
    headers: dict[str, str] | None = None,
) -> bytes:
    """GET `url`, caching the raw response body at `cache_path`.

    Returns the response body as bytes. Bytes rather than text because
    football-data.co.uk serves latin-1 with occasional stray characters, and we
    want the cache to hold exactly what the server sent so that re-parsing never
    depends on a decoding decision made at download time.
    """
    if use_cache is None:
        use_cache = _USE_CACHE

    request_headers = {"User-Agent": USER_AGENT}
    if headers:
        request_headers.update(headers)

    if use_cache and cache_path.exists() and cache_path.stat().st_size > 0:
        log.debug("cache hit  %s", cache_path.name)
        return cache_path.read_bytes()

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    last_error: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            _throttle()
            log.info("GET %s (attempt %d/%d)", url, attempt, _MAX_RETRIES)
            response = requests.get(url, headers=request_headers, timeout=_TIMEOUT_S)
            response.raise_for_status()
            body = response.content
            if not body:
                raise ValueError("empty response body")
            cache_path.write_bytes(body)
            return body
        except Exception as exc:  # noqa: BLE001 -- retry on anything transient
            last_error = exc
            log.warning("  failed: %s", exc)
            if attempt < _MAX_RETRIES:
                # Linear backoff is plenty here; these are small static files.
                time.sleep(_DELAY_S * attempt * 2)

    raise RuntimeError(f"failed to fetch {url} after {_MAX_RETRIES} attempts: {last_error}")


def fetch_text(
    url: str,
    cache_path: Path,
    *,
    encoding: str = "utf-8",
    use_cache: bool | None = None,
    headers: dict[str, str] | None = None,
) -> str:
    """`fetch_bytes` plus a decode. Errors are replaced, never raised."""
    body = fetch_bytes(url, cache_path, use_cache=use_cache, headers=headers)
    return body.decode(encoding, errors="replace")
