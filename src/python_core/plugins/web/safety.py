"""URL safety policy for the web plugin.

The agent must never become a proxy into the local network (SSRF) or a fetcher
of dangerous schemes. :class:`UrlGuard` enforces, before any bytes are fetched:

  * scheme: ``http``/``https`` only (no file://, ftp://, gopher:// …)
  * host resolution: *every* resolved IP must be globally routable — loopback,
    private, link-local, and reserved ranges are rejected (blocks
    http://127.0.0.1:11434/v1-style internal endpoints, cloud metadata at
    169.254.169.254, and DNS names that resolve into private space)
  * domain policy: optional allowlist (empty ⇒ all public hosts) and a
    denylist that always wins (suffix match, so ``evil.example.com`` blocks
    ``sub.evil.example.com`` too)

Pure stdlib (``ipaddress`` + ``socket``), zero dependencies, fully unit-
testable without network access (tests patch ``socket.getaddrinfo``).
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlparse


class WebPolicyError(ValueError):
    """A URL violates the web access policy. Message is safe to show the LLM."""


@dataclass(slots=True)
class UrlGuard:
    """Immutable policy bundle; ``check()`` raises :class:`WebPolicyError`."""

    allowed_domains: tuple[str, ...] = ()   # empty ⇒ every public host is allowed
    blocked_domains: tuple[str, ...] = ()

    def check(self, url: str) -> str:
        """Validate *url* and return the normalized form (string)."""
        parsed = urlparse(url.strip())
        scheme = parsed.scheme.lower()
        if scheme not in ("http", "https"):
            raise WebPolicyError(f"scheme '{scheme or 'none'}' not allowed (use http/https)")
        host = parsed.hostname
        if not host:
            raise WebPolicyError("URL has no hostname")
        if parsed.username or parsed.password:
            raise WebPolicyError("credentials in URLs are not allowed")

        # ── domain policy (suffix match; denylist wins) ────────────────────
        host_l = host.lower().rstrip(".")
        if _matches(host_l, self.blocked_domains):
            raise WebPolicyError(f"domain '{host_l}' is blocked by policy")
        if self.allowed_domains and not _matches(host_l, self.allowed_domains):
            raise WebPolicyError(f"domain '{host_l}' is not in the allowed list")

        # ── resolve and screen every address (SSRF core defense) ───────────
        try:
            infos = socket.getaddrinfo(host_l, parsed.port or (443 if scheme == "https" else 80),
                                       proto=socket.IPPROTO_TCP)
        except socket.gaierror as exc:
            raise WebPolicyError(f"cannot resolve host '{host_l}': {exc}") from exc
        if not infos:
            raise WebPolicyError(f"host '{host_l}' resolves to no addresses")
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if not ip.is_global:
                raise WebPolicyError(
                    f"host '{host_l}' resolves to a non-public address ({ip}) — blocked"
                )
        return parsed.geturl()


def _matches(host: str, patterns: tuple[str, ...]) -> bool:
    """Suffix/domain match: ``evil.com`` matches ``sub.evil.com`` and itself."""
    return any(host == p or host.endswith("." + p) for p in patterns)
