"""PHASE 7: supervisor mode — registry, health, delegation, heartbeat, digest.

The manifest protocol is exercised against a fake in-process client that speaks
the same JSON-RPC shapes as the manifest server (/manifest.json, /health,
/execute), so no network and no live daemon are required.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx

from python_core.core.config import EchoConfig
from python_core.core.event_bus import EventBus
from python_core.core.interfaces import KernelContext
from python_core.core.plugin_manager import PluginManager
from python_core.core.tool_registry import ToolRegistry
from python_core.plugins.scheduler import SchedulerPlugin
from python_core.plugins.server_ops import ServerOpsPlugin
from python_core.plugins.supervisor import SupervisorPlugin


class FakeClient:
    """Speaks the manifest protocol locally: task succeeds or fails on demand."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def execute(
        self, base_url: str, tool: str, args: dict[str, Any] | None = None, *, token: str | None = None
    ) -> dict[str, Any]:
        self.calls.append((tool, dict(args or {}), token))
        if self.fail:
            return {"ok": False, "error": "boom", "agent": base_url, "tool": tool}
        return {"ok": True, "result": "done", "agent": base_url, "tool": tool}


async def _until(predicate, *, timeout: float = 4.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("condition not met in time")


async def test_register_health_delegate_over_fake_manifest(tmp_path: Path) -> None:
    config = EchoConfig(data_dir=tmp_path / "data", enable_semantic_cache=False,
                        consolidation_idle_seconds=9999.0,
                        supervisor_alerts_reply_channel="telegram:111")
    config.ensure_dirs()
    bus = EventBus()
    supervisor = SupervisorPlugin(config)
    supervisor._bus = bus
    fake = FakeClient()
    supervisor.client.execute = fake.execute  # type: ignore[method-assign]

    class FakeResp:
        status_code = 200
        headers = {"content-type": "application/json"}
        def json(self):
            return {"status": "ok"}

    async def fake_get(url: str, headers: dict[str, str]) -> FakeResp:
        return FakeResp()

    supervisor._http_get = fake_get

    assert supervisor.register("fake-agent", "http://fake-agent")["ok"]
    health = await supervisor.health("fake-agent")
    assert health["ok"] and health["status"] == "ok"

    result = await supervisor.delegate("fake-agent", "summarize the logs")
    assert result["ok"] and result["result"] == "done"
    assert fake.calls and fake.calls[0][0] == "task"

    # unknown agent
    assert (await supervisor.delegate("nope", "x"))["ok"] is False


async def test_delegation_failure_alerts_with_backoff(tmp_path: Path) -> None:
    config = EchoConfig(data_dir=tmp_path / "data", enable_semantic_cache=False,
                        consolidation_idle_seconds=9999.0,
                        supervisor_retries=2,
                        supervisor_alerts_reply_channel="telegram:111")
    config.ensure_dirs()
    bus = EventBus()
    supervisor = SupervisorPlugin(config)
    supervisor._bus = bus
    fake = FakeClient(fail=True)
    supervisor.client.execute = fake.execute  # type: ignore[method-assign]
    supervisor.register("fake-agent", "http://fake-agent")

    captured: list[Any] = []
    bus.subscribe("supervisor:alert", lambda e: captured.append(getattr(e, "payload", e)), label="test.alert")

    result = await supervisor.delegate("fake-agent", "broken task")
    assert result["ok"] is False and result["attempts"] == 3
    assert len(fake.calls) == 3  # initial + 2 retries with backoff
    await _until(lambda: len(captured) >= 1)
    alert = captured[0]
    assert alert["severity"] == "critical" and "fake-agent" in alert["message"]


async def test_health_records_unreachable(tmp_path: Path) -> None:
    config = EchoConfig(data_dir=tmp_path / "data", enable_semantic_cache=False,
                        consolidation_idle_seconds=9999.0,
                        supervisor_alerts_reply_channel="telegram:111")
    config.ensure_dirs()
    bus = EventBus()
    supervisor = SupervisorPlugin(config)
    supervisor._bus = bus
    fake = FakeClient(fail=True)
    supervisor.client.execute = fake.execute  # type: ignore[method-assign]
    supervisor.register("down-agent", "http://down.example")

    health = await supervisor.health("down-agent")
    assert health["ok"] is False
    assert health["attempts"] == 3  # retries with backoff


async def test_supervisor_cron_heartbeat_and_digest(tmp_path: Path) -> None:
    config = EchoConfig(data_dir=tmp_path / "data", enable_semantic_cache=False,
                        consolidation_idle_seconds=9999.0,
                        supervisor_alerts_reply_channel="telegram:111")
    config.ensure_dirs()
    bus = EventBus()
    registry = ToolRegistry(bus)
    ctx = KernelContext(bus=bus, config=config, tools=registry, data_dir=config.data_dir)
    pm = PluginManager(bus, config)
    ctx.services["plugin_manager"] = pm
    pm.register(ServerOpsPlugin(config))
    pm.register(SchedulerPlugin(config))
    pm.register(SupervisorPlugin(config))
    await pm.start_all(ctx)
    try:
        scheduler = pm.get("scheduler")
        jobs = await scheduler.list_jobs()
        jobs_payload = jobs.get("jobs", jobs)
        names = [j.get("name") for j in (jobs_payload.values() if isinstance(jobs_payload, dict) else jobs_payload)]
        assert "agent-heartbeat" in names and "agent-digest" in names

        supervisor = ctx.service("supervisor")
        supervisor.register("down-agent", "http://down.example")
        async def failing_get(url: str, headers: dict[str, str]) -> Any:
            raise httpx.ConnectError("boom")
        supervisor._http_get = failing_get

        # heartbeat with an unreachable agent fires an alert
        captured: list[Any] = []
        bus.subscribe("supervisor:alert", lambda e: captured.append(getattr(e, "payload", e)), label="test2.alert")
        await supervisor.heartbeat()
        await _until(lambda: len(captured) >= 1)
        assert any("unreachable" in str(p.get("message", "")) for p in captured)

        digest = await supervisor.digest()
        assert "Supervisor daily digest" in digest["text"]
    finally:
        await pm.stop_all()
