"""Web plugin: SSRF policy + bounded, self-correcting tool envelopes."""
import socket
from unittest.mock import patch

from python_core.core.config import EchoConfig
from python_core.core.tool_registry import ToolRegistry
from python_core.plugins.web.plugin import WebPlugin
from python_core.plugins.web.safety import UrlGuard, WebPolicyError


def _guard(allowed=(), blocked=()) -> UrlGuard:
    return UrlGuard(allowed_domains=tuple(allowed), blocked_domains=tuple(blocked))


def _patch_resolve(ip: str):
    """Patch socket.getaddrinfo to return a single address (offline-friendly)."""
    def fake(host, port, *a, **kw):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, int(port or 80)))]
    return patch("python_core.plugins.web.safety.socket.getaddrinfo", fake)


def test_guard_rejects_bad_schemes() -> None:
    g = _guard()
    for url in ("file:///etc/passwd", "ftp://x", "gopher://x", "javascript:alert(1)"):
        try:
            g.check(url)
            raised = False
        except WebPolicyError:
            raised = True
        assert raised, url


def test_guard_blocks_private_and_loopback() -> None:
    for ip in ("127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "172.16.0.1"):
        g = _guard()
        with _patch_resolve(ip):
            try:
                g.check("https://example.com")
                raised = False
            except WebPolicyError:
                raised = True
            assert raised, ip


def test_guard_allows_public_hosts() -> None:
    g = _guard()
    with _patch_resolve("93.184.216.34"):
        assert g.check("https://example.com/path?q=1").startswith("https://example.com/")


def test_guard_domain_policy() -> None:
    g = _guard(blocked=("evil.com",))
    with _patch_resolve("93.184.216.34"):
        # blocked wins, including subdomains
        try:
            g.check("https://sub.evil.com")
            raised = False
        except WebPolicyError:
            raised = True
        assert raised

    g2 = _guard(allowed=("example.com",))
    with _patch_resolve("93.184.216.34"):
        try:
            g2.check("https://example.com")
            ok = True
        except WebPolicyError:
            ok = False
        assert ok
        try:
            g2.check("https://other.com")
            raised = False
        except WebPolicyError:
            raised = True
        assert raised


def test_guard_rejects_credentials_and_missing_host() -> None:
    g = _guard()
    with _patch_resolve("93.184.216.34"):
        try:
            g.check("https://user:pass@example.com")
            raised = False
        except WebPolicyError:
            raised = True
        assert raised
    with _patch_resolve("93.184.216.34"):
        try:
            g.check("https://")
            raised = False
        except WebPolicyError:
            raised = True
        assert raised


def _plugin() -> WebPlugin:
    return WebPlugin(EchoConfig.load())


async def test_web_fetch_blocked_url_returns_data() -> None:
    p = _plugin()
    out = await p._tool_fetch({"url": "http://127.0.0.1:8080/secret"})
    assert out["ok"] is False
    assert "blocked by policy" in out["error"]


async def test_web_fetch_missing_url() -> None:
    p = _plugin()
    out = await p._tool_fetch({})
    assert out["ok"] is False
    assert "required" in out["error"]


async def test_web_extract_requires_selector() -> None:
    p = _plugin()
    out = await p._tool_extract({"url": "https://example.com"})
    assert out["ok"] is False
    assert "selector" in out["error"].lower()


async def test_web_rate_limit_per_domain() -> None:
    p = _plugin()
    p.config.web_rate_limit_per_min = 2
    # bypass actual fetch by monkeypatching the fetcher
    async def fake_fetch(url, **kw):
        return {"ok": True, "url": url, "status": 200, "content": "hi", "truncated": False, "backend": "httpx"}
    p._fetcher.fetch = fake_fetch  # type: ignore[assignment]

    with _patch_resolve("93.184.216.34"):
        r1 = await p._tool_fetch({"url": "https://example.com/a"})
        r2 = await p._tool_fetch({"url": "https://example.com/b"})
        r3 = await p._tool_fetch({"url": "https://example.com/c"})
    assert r1["ok"] is True
    assert r2["ok"] is True
    assert r3["ok"] is False
    assert "rate limit" in r3["error"]


async def test_web_fetch_truncates_large_content() -> None:
    p = _plugin()
    big = "x" * 100_000

    async def fake_fetch(url, **kw):
        return {"ok": True, "url": url, "status": 200, "content": big, "truncated": False, "backend": "httpx"}

    p._fetcher.fetch = fake_fetch  # type: ignore[assignment]
    with _patch_resolve("93.184.216.34"):
        out = await p._tool_fetch({"url": "https://example.com", "max_chars": 1000})
    assert out["ok"] is True
    assert len(out["content"]) == 1000
    assert out["truncated"] is True


async def test_web_tools_registered_on_start() -> None:
    from python_core.core.config import EchoConfig
    from python_core.core.event_bus import EventBus
    from python_core.core.interfaces import KernelContext

    p = WebPlugin(EchoConfig.load())
    reg = ToolRegistry()
    captured: list[dict] = []

    async def capture(event):
        captured.append(event.payload)

    bus = EventBus()
    bus.subscribe("tool:register", capture)
    ctx = KernelContext(bus=bus, config=EchoConfig.load(), tools=reg, data_dir=EchoConfig.load().data_dir)
    await p.start(ctx)

    names = {e["spec"].name for e in captured}
    assert {"web_fetch", "web_extract", "web_links"} <= names
    assert ctx.services["web"] is p


# ── httpx fallback backend (must work without Scrapling installed) ──────────
async def test_httpx_backend_extracts_readable_text() -> None:
    import httpx

    from python_core.plugins.web.fetcher import WebFetcher

    pages = {
        "/": (
            "<html><head><title>Hello Page</title></head><body>"
            "<h1>Hello</h1><script>alert('noise')</script>"
            "<p>World &amp; friends</p></body></html>"
        )
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=pages[request.url.path],
            headers={"content-type": "text/html; charset=utf-8"},
        )

    fetcher = WebFetcher(timeout=5, transport=httpx.MockTransport(handler), auto_stealth=False)
    out = await fetcher.fetch("https://example.com/")

    assert out["ok"] is True
    assert out["status"] == 200
    assert out["backend"] == "httpx"
    assert out["title"] == "Hello Page"
    assert "Hello" in out["content"] and "World & friends" in out["content"]
    assert "alert" not in out["content"]


async def test_httpx_backend_reports_http_errors_as_data() -> None:
    import httpx

    from python_core.plugins.web.fetcher import WebFetcher

    transport = httpx.MockTransport(lambda request: httpx.Response(404, text="missing"))
    out = await WebFetcher(timeout=5, transport=transport, auto_stealth=False).fetch("https://example.com/nope")

    assert out["ok"] is False
    assert out["status"] == 404
    assert "HTTP 404" in out["error"]


async def test_auto_stealth_retries_js_only_pages() -> None:
    from python_core.plugins.web.fetcher import WebFetcher

    fetcher = WebFetcher(timeout=1, backend="httpx")

    async def plain(url, selector, max_chars, sel_limit):
        return {"ok": True, "url": url, "status": 200, "content": "JS shell", "backend": "httpx"}

    async def stealth(url, selector, max_chars, sel_limit):
        return {
            "ok": True,
            "url": url,
            "status": 200,
            "content": "Rendered article text. " * 20,
            "backend": "stealthy",
        }

    fetcher._fetch_httpx = plain  # type: ignore[method-assign]
    fetcher._fetch_stealthy = stealth  # type: ignore[method-assign]
    fetcher.has_stealth = lambda: True  # type: ignore[method-assign]

    out = await fetcher.fetch("https://example.com/js-app")

    assert out["ok"] is True
    assert out["auto_stealth"] is True
    assert out["backend"] == "stealthy"
    assert len(out["content"]) > 80


async def test_auto_stealth_failure_keeps_plain_result() -> None:
    from python_core.plugins.web.fetcher import WebFetcher

    fetcher = WebFetcher(timeout=1, backend="httpx")

    async def plain(url, selector, max_chars, sel_limit):
        return {"ok": True, "url": url, "status": 200, "content": "tiny", "backend": "httpx"}

    async def stealth(url, selector, max_chars, sel_limit):
        return {"ok": False, "url": url, "error": "browser not installed"}

    fetcher._fetch_httpx = plain  # type: ignore[method-assign]
    fetcher._fetch_stealthy = stealth  # type: ignore[method-assign]
    fetcher.has_stealth = lambda: True  # type: ignore[method-assign]

    out = await fetcher.fetch("https://example.com/js-app")

    assert out["content"] == "tiny"
    assert "auto_stealth" not in out


async def test_redirect_to_private_address_is_blocked() -> None:
    p = _plugin()

    async def fake_fetch(url, **kw):
        return {
            "ok": True,
            "url": url,
            "final_url": "http://127.0.0.1:8080/internal",
            "status": 200,
            "content": "should never reach the model",
        }

    p._fetcher.fetch = fake_fetch  # type: ignore[assignment]
    out = await p._tool_fetch({"url": "https://example.com/redirect"})

    assert out["ok"] is False
    assert "redirect blocked" in out["error"]
