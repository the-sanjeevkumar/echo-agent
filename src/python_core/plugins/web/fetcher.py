"""Fetch backends for the web plugin: Scrapling first, httpx as fallback.

Backend ladder (auto-detected at import time — heavy deps degrade gracefully,
consistent with the rest of echo-agent):

  1. ``scrapling.fetchers.Fetcher``   — fast HTTP with browser TLS impersonation
  2. ``scrapling.fetchers.StealthyFetcher`` — headless browser; bypasses
     Cloudflare Turnstile/Interstitial (opt-in per call via ``stealth=True``;
     needs ``scrapling[fetchers]`` + ``scrapling install``)
  3. plain ``httpx``                  — always available, no bypass powers

Every path returns the SAME bounded envelope, so the agent loop never sees an
exception from the web layer — errors are data, exactly like tool envelopes.
Scrapling's sync API runs in a worker thread so the asyncio loop never blocks.
"""

from __future__ import annotations

import asyncio
import html as _html
import logging
import re
from typing import Any

logger = logging.getLogger("echo.web")

try:  # scrapling core (parser) — installed alone it still imports the parser
    from scrapling.parser import Selector as _ScraplingSelector
except Exception:  # noqa: BLE001 — optional dependency
    _ScraplingSelector = None  # type: ignore[assignment]

try:  # fetchers extra: curl_cffi-backed fast HTTP
    from scrapling.fetchers import Fetcher as _ScraplingFetcher
except Exception:  # noqa: BLE001 — optional dependency
    _ScraplingFetcher = None  # type: ignore[assignment]

try:  # fetchers extra: playwright-backed stealth browser
    from scrapling.fetchers import StealthyFetcher as _ScraplingStealthy
except Exception:  # noqa: BLE001 — optional dependency
    _ScraplingStealthy = None  # type: ignore[assignment]

import httpx

_MAX_BYTES = 2_000_000  # download cap for the httpx path (memory-bomb guard)
_TAG_NOISE = re.compile(r"(?is)<(script|style|noscript|svg|head|iframe)[^>]*>.*?</\1>")
_TAG_ANY = re.compile(r"(?s)<[^>]+>")
_TITLE = re.compile(r"(?is)<title[^>]*>(.*?)</title>")
_WS = re.compile(r"\s+")


def _strip_html(raw: str) -> str:
    """Crude-but-safe text extraction for the httpx fallback path."""
    text = _TAG_ANY.sub(" ", _TAG_NOISE.sub(" ", raw))
    return _WS.sub(" ", _html.unescape(text)).strip()


def _extract_text(page: Any) -> str:
    """Best-effort LLM-ready text from a Scrapling page object."""
    for getter in ("markdown", "get_all_text"):  # rag extra → plain text
        method = getattr(page, getter, None)
        if callable(method):
            try:
                out = method()
                if isinstance(out, str) and out.strip():
                    return out.strip()
            except Exception:  # noqa: BLE001 — rag extra may be absent
                logger.debug("text extractor %s unavailable", getter)
    text = getattr(page, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    return ""


def _extract_title(page: Any) -> str:
    """Best-effort page title across Scrapling versions (css_first isn't always present)."""
    try:
        titles = page.css("title::text")
        if titles:
            return str(titles[0]).strip()[:200]
    except Exception as exc:  # noqa: BLE001 — title is cosmetic
        logger.debug("title extraction failed: %s", exc)
    return ""


def _add_text_hint(result: dict[str, Any]) -> dict[str, Any]:
    """Tell the model when a page has little/no visible text (usually JS-rendered),
    so it can retry with ``stealth=true`` instead of concluding the page is empty."""
    content = str(result.get("content") or "").strip()
    if not content:
        result["hint"] = (
            "no readable text found; the page is likely JavaScript-rendered. "
            "Retry with stealth=true (needs the Scrapling browser fetcher)."
        )
    elif len(content) < 80:
        result["hint"] = (
            "very little readable text; the page may be JavaScript-rendered. "
            "Retry with stealth=true for a browser-rendered version."
        )
    return result


def _run_selector(page: Any, selector: str, limit: int) -> list[str]:
    """CSS/XPath-ish selection; returns string matches only (Scrapling's
    ``::text`` pseudo yields strings directly; elements are stringified)."""
    items = page.css(selector)
    out: list[str] = []
    for item in items[:limit]:
        out.append(item if isinstance(item, str) else str(item))
    return [s for s in (t.strip() for t in out) if s]


def _richness(result: dict[str, Any]) -> int:
    """Crude measure of how useful a fetch result is (text or selector matches)."""
    matches = result.get("matches")
    if isinstance(matches, list):
        return len(matches)
    return len(str(result.get("content") or "").strip())


def _needs_browser(result: dict[str, Any]) -> bool:
    """True when a plain fetch clearly got a JavaScript shell instead of content."""
    if not result.get("ok"):
        return False
    if isinstance(result.get("matches"), list):
        return len(result["matches"]) == 0
    return len(str(result.get("content") or "").strip()) < 80


class WebFetcher:
    """Stateless fetch facade. ``backend`` reports what actually runs.

    When a fast fetch yields little/no readable text (classic JavaScript-only
    page), it automatically retries once with the headless browser backend and
    returns the richer result. That fallback is what makes modern SPA/news
    sites usable without the LLM having to guess that it needs ``stealth=true``.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
        *,
        backend: str = "auto",
        auto_stealth: bool = True,
    ) -> None:
        self.timeout = timeout
        self._transport = transport  # injectable for offline tests
        self._backend = backend
        self.auto_stealth = auto_stealth

    @property
    def backend(self) -> str:
        if self._transport is not None or self._backend == "httpx":
            return "httpx"
        if _ScraplingFetcher is not None:
            return "scrapling"
        return "httpx"

    def has_stealth(self) -> bool:
        return _ScraplingStealthy is not None

    # ── public API ─────────────────────────────────────────────────────────
    async def fetch(
        self,
        url: str,
        *,
        selector: str | None = None,
        stealth: bool = False,
        max_chars: int = 8000,
        selector_limit: int = 50,
    ) -> dict[str, Any]:
        """Fetch *url* → bounded result envelope. NEVER raises."""
        try:
            if stealth:
                return await self._fetch_stealthy(url, selector, max_chars, selector_limit)

            plain = await self._fetch_plain(url, selector, max_chars, selector_limit)

            if self.auto_stealth and _needs_browser(plain) and self.has_stealth():
                browser = await self._fetch_stealthy(url, selector, max_chars, selector_limit)
                if browser.get("ok") and _richness(browser) > _richness(plain):
                    browser["auto_stealth"] = True
                    browser.pop("hint", None)
                    return browser
            return plain
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — web failures are data, not crashes
            logger.debug("web fetch failed for %s: %s", url, exc)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "url": url}

    async def _fetch_plain(self, url: str, selector: str | None, max_chars: int, sel_limit: int) -> dict[str, Any]:
        use_scrapling = (
            self._transport is None
            and self._backend in ("auto", "scrapling")
            and _ScraplingFetcher is not None
        )
        if use_scrapling:
            return await self._fetch_scrapling(url, selector, max_chars, sel_limit)
        return await self._fetch_httpx(url, selector, max_chars, sel_limit)

    # ── backends ───────────────────────────────────────────────────────────
    async def _fetch_scrapling(self, url: str, selector: str | None, max_chars: int, sel_limit: int) -> dict[str, Any]:
        def _sync() -> dict[str, Any]:
            page = _ScraplingFetcher.get(url, timeout=self.timeout, stealthy_headers=True)
            status = int(getattr(page, "status", 0) or 0)
            if status >= 400:
                return {"ok": False, "error": f"HTTP {status}", "url": url, "status": status, "backend": "scrapling"}
            result: dict[str, Any] = {
                "ok": True,
                "url": url,
                "final_url": str(getattr(page, "url", url) or url),
                "status": status,
                "backend": "scrapling",
            }
            if selector:
                result["matches"] = _run_selector(page, selector, sel_limit)
            else:
                content = _extract_text(page)
                result["content"] = content[:max_chars]
                result["truncated"] = len(content) > max_chars
                result["title"] = _extract_title(page)
                _add_text_hint(result)
            return result

        return await asyncio.to_thread(_sync)

    async def _fetch_stealthy(self, url: str, selector: str | None, max_chars: int, sel_limit: int) -> dict[str, Any]:
        if _ScraplingStealthy is None:
            return {
                "ok": False,
                "error": (
                    "stealth mode needs Scrapling's browser fetcher: uv sync && "
                    "uv run scrapling install (or python -m patchright install chromium)"
                ),
                "url": url,
            }

        def _sync() -> dict[str, Any]:
            # Scrapling's browser timeout is in MILLISECONDS (playwright
            # convention) while our config is in seconds — converting avoids
            # the pathological "Timeout 30ms exceeded" failure.
            page = _ScraplingStealthy.fetch(
                url,
                headless=True,
                timeout=int(self.timeout * 1000),
                network_idle=True,
                retries=1,
            )
            status = int(getattr(page, "status", 0) or 0)
            if status >= 400:
                return {"ok": False, "error": f"HTTP {status}", "url": url, "status": status, "backend": "stealthy"}
            result: dict[str, Any] = {
                "ok": True,
                "url": url,
                "final_url": str(getattr(page, "url", url) or url),
                "status": status,
                "backend": "stealthy",
            }
            if selector:
                result["matches"] = _run_selector(page, selector, sel_limit)
            else:
                content = _extract_text(page)
                result["content"] = content[:max_chars]
                result["truncated"] = len(content) > max_chars
                result["title"] = _extract_title(page)
                _add_text_hint(result)
            return result

        return await asyncio.to_thread(_sync)

    async def _fetch_httpx(self, url: str, selector: str | None, max_chars: int, sel_limit: int) -> dict[str, Any]:
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) echo-agent/0.1"}
        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            transport=self._transport,
        ) as client, client.stream("GET", url, headers=headers) as resp:
                status = resp.status_code
                if status >= 400:
                    return {"ok": False, "error": f"HTTP {status}", "url": url, "status": status, "backend": "httpx"}
                chunks: list[str] = []
                total = 0
                # NB: iter_text() is a *sync* generator; inside an async client
                # the correct API is aiter_text(). Using the wrong one made
                # every fallback fetch fail before a single byte was read.
                async for chunk in resp.aiter_text():
                    chunks.append(chunk)
                    total += len(chunk)
                    if total >= _MAX_BYTES:
                        break
        raw = "".join(chunks)[:_MAX_BYTES]
        result: dict[str, Any] = {
            "ok": True,
            "url": url,
            "final_url": str(resp.url),
            "status": status,
            "backend": "httpx",
        }
        if selector and _ScraplingSelector is not None:
            page = _ScraplingSelector(raw)
            result["matches"] = _run_selector(page, selector, sel_limit)
        else:
            content = _strip_html(raw)
            result["content"] = content[:max_chars]
            result["truncated"] = len(content) > max_chars
            m = _TITLE.search(raw)
            result["title"] = _html.unescape(m.group(1)).strip()[:200] if m else ""
            _add_text_hint(result)
        return result
