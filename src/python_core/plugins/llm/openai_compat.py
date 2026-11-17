"""OpenAI-compatible chat-completion provider (local-first by default).

Works with any server that speaks the ``/v1/chat/completions`` protocol:
Ollama, llama.cpp, vLLM, LM Studio, or hosted APIs. Structured-output mode
(MODULE 4) requests strict ``json_schema`` response formats; when a server
does not support them, we degrade gracefully by appending the schema to the
prompt and parsing defensively.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from python_core.core.config import EchoConfig
from python_core.core.interfaces import KernelContext
from python_core.core.utils import extract_json

logger = logging.getLogger("echo.llm")

# Statuses worth waiting out: rate limits and transient server errors.
# 4xx validation errors (400/422) are NOT retried — the caller handles the
# json_schema-fallback path for those.
_RETRYABLE: frozenset[int] = frozenset({429, 500, 502, 503, 504})
_MAX_RETRY_AFTER_WAIT = 5.0  # never stall the interactive loop longer than this


class LLMProviderError(RuntimeError):
    """Provider-level failure after bounded retries.

    Carries a clean, user-safe message (no raw httpx internals) plus the HTTP
    status so callers can react specifically (e.g. 429 → "slow down").
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class OpenAICompatProvider:
    """Minimal, dependency-light provider. One short-lived HTTP client per
    call keeps it safe under any event loop (including test loops)."""

    name = "llm_openai_compat"
    version = "0.1.0"
    capabilities = ["llm", "structured-output"]

    def __init__(self, config: EchoConfig) -> None:
        self.config = config
        self._ctx: KernelContext | None = None

    # ── plugin lifecycle ───────────────────────────────────────────────────
    def start(self, ctx: KernelContext) -> None:
        self._ctx = ctx
        ctx.services["llm"] = self  # other plugins may route through this
        ctx.logger.info("LLM provider ready: %s (model=%s)", self.config.llm_base_url, self.config.llm_model)

    def stop(self) -> None:
        self._ctx = None

    # ── completion ─────────────────────────────────────────────────────────
    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Return ``{"content": <str|dict>, "usage": {...}}``.

        When ``schema`` is given we try native structured output first and
        fall back to prompt-embedded JSON instructions for servers that
        ignore ``response_format``.
        """
        body: dict[str, Any] = {
            "model": self.config.llm_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            body["max_tokens"] = max_tokens

        native_schema = False
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "echo_output", "schema": schema, "strict": True},
            }
            native_schema = True

        headers = {"Content-Type": "application/json"}
        if self.config.llm_api_key:
            headers["Authorization"] = f"Bearer {self.config.llm_api_key}"

        url = self.config.llm_base_url.rstrip("/") + "/chat/completions"
        try:
            resp = await self._post(url=url, body=body, headers=headers)
        except httpx.HTTPStatusError as exc:
            # Some servers reject json_schema response_format; retry plain.
            if native_schema and exc.response.status_code in (400, 422):
                logger.info("server rejected json_schema; retrying with prompt-embedded schema")
                return await self._prompt_embedded(messages, schema, temperature, max_tokens, headers, url)
            raise LLMProviderError(
                f"LLM provider returned HTTP {exc.response.status_code}", status=exc.response.status_code
            ) from exc
        except httpx.HTTPError as exc:
            # Connection refused/timeout/DNS: a typed, friendly error the UI can
            # show as one line — never a raw httpx traceback in the REPL.
            raise LLMProviderError(
                f"LLM endpoint unreachable ({type(exc).__name__})", status=0
            ) from exc
        data = resp.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        if schema is not None:
            # Defensive: accept dict content, JSON strings, or embedded JSON.
            parsed = content if isinstance(content, dict) else extract_json(content if isinstance(content, str) else "")
            return {"content": parsed, "raw": content, "usage": usage}
        return {"content": content, "usage": usage}

    async def _post(self, *, url: str, body: dict[str, Any], headers: dict[str, str]) -> httpx.Response:
        """POST with bounded retries on 429/5xx.

        Honors ``Retry-After`` when the server sends it (capped so an
        interactive session never stalls long), else exponential backoff.
        Non-retryable statuses raise immediately; after exhausting retries a
        clean :class:`LLMProviderError` is raised.
        """
        retries = max(0, int(getattr(self.config, "llm_retries", 3)))
        attempt = 0
        async with httpx.AsyncClient(timeout=self.config.llm_timeout) as client:
            while True:
                resp = await client.post(url, json=body, headers=headers)
                if resp.status_code not in _RETRYABLE or attempt >= retries:
                    resp.raise_for_status()  # success (or final, non-retryable failure)
                    return resp
                retry_after = resp.headers.get("retry-after", "")
                try:
                    wait = min(float(retry_after), _MAX_RETRY_AFTER_WAIT) if retry_after else 0.0
                except ValueError:  # Retry-After can be an HTTP-date; ignore then
                    wait = 0.0
                if wait <= 0:
                    wait = min(1.0 * (2**attempt), 4.0)
                logger.info(
                    "LLM HTTP %d; retrying (%d/%d) in %.1fs", resp.status_code, attempt + 1, retries, wait
                )
                await asyncio.sleep(wait)
                attempt += 1

    async def _prompt_embedded(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        temperature: float,
        max_tokens: int | None,
        headers: dict[str, str],
        url: str,
    ) -> dict[str, Any]:
        """Fallback path: schema appended as a plain instruction."""
        instruction = (
            "CRITICAL: respond with a single JSON object and nothing else. "
            f"It must satisfy this JSON Schema:\n{json.dumps(schema)}"
        )
        augmented = messages + [{"role": "user", "content": instruction}]
        body: dict[str, Any] = {
            "model": self.config.llm_model,
            "messages": augmented,
            "temperature": temperature,
        }
        if max_tokens:
            body["max_tokens"] = max_tokens
        try:
            resp = await self._post(url=url, body=body, headers=headers)
        except httpx.HTTPStatusError as exc:
            raise LLMProviderError(
                f"LLM provider returned HTTP {exc.response.status_code}", status=exc.response.status_code
            ) from exc
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"content": extract_json(content if isinstance(content, str) else ""), "raw": content, "usage": data.get("usage", {})}


def create_plugin() -> OpenAICompatProvider:
    """Plugin discovery contract entry point (uses loaded EchoConfig)."""
    return OpenAICompatProvider(EchoConfig.load())
