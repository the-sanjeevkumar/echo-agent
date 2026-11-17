"""LLM provider resilience: bounded retries on 429/5xx with backoff, and
clean typed errors — the user never sees raw httpx tracebacks."""

import asyncio

import httpx
import pytest

from python_core.core.config import EchoConfig
from python_core.core.event_bus import EventBus
from python_core.core.tool_registry import ToolRegistry
from python_core.orchestrator import Orchestrator
from python_core.plugins.llm.openai_compat import LLMProviderError, OpenAICompatProvider


class _FakeResp:
    def __init__(self, status: int, headers: dict[str, str] | None = None) -> None:
        self.status_code = status
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            req = httpx.Request("POST", "http://fake/v1/chat/completions")
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=req, response=self  # type: ignore[arg-type]
            )

    def json(self) -> dict:
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}


def _cfg(retries: int) -> EchoConfig:
    cfg = EchoConfig.load()
    cfg.llm_retries = retries
    return cfg


def _patch_http(monkeypatch: pytest.MonkeyPatch, *responses: _FakeResp):
    """Queue of canned responses; the last one repeats for extra calls."""
    queue = list(responses)
    calls: list[_FakeResp] = []

    async def fake_post(self, url, **kwargs):  # noqa: ANN001
        resp = queue.pop(0) if len(queue) > 1 else queue[0]
        calls.append(resp)
        return resp

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    return calls


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Capture backoff waits without actually stalling the test."""
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return waits


async def test_recovers_from_transient_rate_limit(monkeypatch, no_sleep) -> None:
    calls = _patch_http(monkeypatch, _FakeResp(429), _FakeResp(429), _FakeResp(200))
    p = OpenAICompatProvider(_cfg(retries=2))

    out = await p.complete([{"role": "user", "content": "hi"}])

    assert out["content"] == "ok"
    assert len(calls) == 3
    assert no_sleep  # backoff was scheduled between attempts


async def test_honors_retry_after_header_capped(monkeypatch, no_sleep) -> None:
    _patch_http(monkeypatch, _FakeResp(429, {"retry-after": "60"}), _FakeResp(200))
    p = OpenAICompatProvider(_cfg(retries=1))

    out = await p.complete([{"role": "user", "content": "hi"}])

    assert out["content"] == "ok"
    # Server asked for 60s; we cap the interactive stall.
    assert no_sleep == [5.0]


async def test_exhausted_retries_raise_clean_error(monkeypatch, no_sleep) -> None:
    calls = _patch_http(monkeypatch, _FakeResp(429))
    p = OpenAICompatProvider(_cfg(retries=2))

    with pytest.raises(LLMProviderError) as ei:
        await p.complete([{"role": "user", "content": "hi"}])

    assert ei.value.status == 429
    assert len(calls) == 3  # initial + 2 retries
    assert "httpx" not in str(ei.value)  # no raw internals in the message


async def test_validation_error_falls_back_then_fails_clean(monkeypatch, no_sleep) -> None:
    """400s are never retried: schema fallback runs once, then a clean error."""
    calls = _patch_http(monkeypatch, _FakeResp(400))
    p = OpenAICompatProvider(_cfg(retries=2))

    with pytest.raises(LLMProviderError) as ei:
        await p.complete([{"role": "user", "content": "hi"}], schema={"type": "object"})

    assert ei.value.status == 400
    assert len(calls) == 2  # native attempt + prompt-embedded attempt
    assert no_sleep == []  # validation errors are not slept on


class _RateLimitedProvider:
    async def complete(self, *args, **kwargs):
        raise LLMProviderError("rate limited by provider", status=429)


async def test_orchestrator_surfaces_friendly_rate_limit_message() -> None:
    orch = Orchestrator(
        bus=EventBus(), config=EchoConfig.load(), tools=ToolRegistry(), provider=_RateLimitedProvider()
    )
    action = await orch._ask([{"role": "user", "content": "hello"}])

    assert action["action"] == "respond"
    assert "rate limited" in action["response"]
    assert "httpx" not in action["response"]
    assert "429" not in action["response"]  # user-facing text stays human


async def test_connection_failure_is_typed_and_clean(monkeypatch) -> None:
    async def refuse(self, url, **kwargs):  # noqa: ANN001
        raise httpx.ConnectError("connection refused", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", refuse)
    p = OpenAICompatProvider(_cfg(retries=2))

    with pytest.raises(LLMProviderError) as ei:
        await p.complete([{"role": "user", "content": "hi"}])

    assert ei.value.status == 0
    assert "unreachable" in str(ei.value)
    assert "httpx" not in str(ei.value)


class _UnreachableProvider:
    async def complete(self, *args, **kwargs):
        raise LLMProviderError("LLM endpoint unreachable (ConnectError)", status=0)


async def test_orchestrator_surfaces_friendly_unreachable_message() -> None:
    orch = Orchestrator(
        bus=EventBus(), config=EchoConfig.load(), tools=ToolRegistry(), provider=_UnreachableProvider()
    )
    action = await orch._ask([{"role": "user", "content": "hello"}])

    assert action["action"] == "respond"
    assert "can't reach the LLM" in action["response"]
    assert "httpx" not in action["response"]
