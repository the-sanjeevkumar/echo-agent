"""PHASE 6: server-ops pack — read-only metrics, allowlist, confirmation flow,
scheduled health checks and owner-chat alerts."""

from __future__ import annotations

import asyncio
from pathlib import Path

from python_core.core.config import EchoConfig
from python_core.core.event_bus import EventBus
from python_core.core.interfaces import KernelContext
from python_core.core.plugin_manager import PluginManager
from python_core.core.tool_registry import ToolRegistry
from python_core.plugins.scheduler import SchedulerPlugin
from python_core.plugins.server_ops import ServerOpsPlugin
from python_core.plugins.server_ops import metrics as ops_metrics


async def _until(predicate, *, timeout: float = 4.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("condition not met in time")


# ── read-only tools ───────────────────────────────────────────────────────────

async def test_sys_metrics_and_readonly_tools(kernel) -> None:
    result = await kernel.ctx.tools.execute("sys_metrics", {})
    assert result["ok"] and result["result"]["ok"]
    for key in ("cpu_percent", "ram", "disk", "load_avg", "uptime_seconds"):
        assert key in result["result"]

    usage = await kernel.ctx.tools.execute("disk_usage", {"path": "/"})
    assert usage["ok"] and usage["result"]["ok"] and "percent" in usage["result"]

    docker = await kernel.ctx.tools.execute("docker_ps", {})
    assert docker["ok"] and "containers" in docker["result"] or "error" in docker["result"]

    # every action is audited to episodic memory
    memory = kernel.ctx.service("memory")
    assert any(e["action"].startswith("server_ops:") for e in memory.em.recent(limit=20))


# ── confirmation flow ─────────────────────────────────────────────────────────

async def test_service_restart_confirmation_flow(kernel, monkeypatch) -> None:
    ops = kernel.ctx.service("server_ops")
    ops.config.server_ops_units = ["nginx.service"]
    calls: list[str] = []

    async def fake_restart(unit: str) -> dict:
        calls.append(unit)
        return {"unit": unit, "ok": True, "returncode": 0, "stderr": ""}

    monkeypatch.setattr(ops_metrics, "unit_restart", fake_restart)

    pending = await kernel.ctx.tools.execute("service_restart", {"unit": "nginx.service"})
    assert pending["result"]["ok"] and pending["result"]["pending"]
    confirm_id = pending["result"]["confirmation_id"]
    assert calls == [], "no mutating action before confirmation"

    reply = await kernel.orchestrator.handle_task(
        f"/confirm {confirm_id}", source="cli", user_id="owner", session_id="phase6"
    )
    assert "Confirmed" in reply.get("response", "")
    assert calls == ["nginx.service"]

    # confirmation is single-use
    again = await ops.confirm(confirm_id)
    assert again["ok"] is False


async def test_confirmation_is_owner_only(kernel) -> None:
    reply = await kernel.orchestrator.handle_task(
        "/confirm sc-whatever", source="telegram", user_id="999000111", session_id="phase6"
    )
    assert "owner-only" in reply.get("response", "")


# ── allowlist ─────────────────────────────────────────────────────────────────

async def test_allowlist_rejection(kernel) -> None:
    restart = await kernel.ctx.tools.execute("service_restart", {"unit": "evil.service"})
    assert restart["result"]["ok"] is False
    assert "not allowlisted" in restart["result"]["error"]

    status = await kernel.ctx.tools.execute("service_status", {"unit": "evil.service"})
    assert status["result"]["ok"] is False and "not allowlisted" in status["result"]["error"]

    journal = await kernel.ctx.tools.execute("journal_tail", {"unit": "evil.service", "lines": 10})
    assert journal["result"]["ok"] is False


# ── health checks / alerts ────────────────────────────────────────────────────

async def test_health_alert_on_threshold_breach(kernel, monkeypatch) -> None:
    kernel.record("server:alert")
    ops = kernel.ctx.service("server_ops")
    ops.config.server_ops_units = []
    ops.config.server_ops_disk_threshold = 90.0
    monkeypatch.setattr(ops_metrics, "sys_metrics", lambda: {
        "cpu_percent": 10.0,
        "ram": {"total_mb": 16000, "available_mb": 8000, "percent": 50.0},
        "disk": {"path": "/", "percent": 95.0},
        "load_avg": [0.1, 0.1, 0.1],
        "uptime_seconds": 100.0,
    })

    report = ops.check_health()
    assert report["checked"] and any(a["kind"] == "disk" for a in report["alerts"])
    await _until(lambda: len(kernel.captured["server:alert"]) >= 1)
    alert = kernel.captured["server:alert"][0]
    assert alert["kind"] == "disk" and "95" in alert["message"]


async def test_health_service_down_alert(kernel, monkeypatch) -> None:
    kernel.record("server:alert")
    ops = kernel.ctx.service("server_ops")
    ops.config.server_ops_units = ["nginx.service"]
    monkeypatch.setattr(ops_metrics, "sys_metrics", lambda: {
        "cpu_percent": 10.0,
        "ram": {"percent": 20.0},
        "disk": {"percent": 20.0},
        "load_avg": [], "uptime_seconds": 10.0,
    })
    monkeypatch.setattr(ops_metrics, "unit_status", lambda unit: {"unit": unit, "active": "inactive", "ok": False})

    report = ops.check_health()
    assert any(a["kind"] == "service" and "nginx.service" in a["message"] for a in report["alerts"])


async def test_health_cron_registered_via_scheduler(tmp_path: Path) -> None:
    config = EchoConfig(data_dir=tmp_path / "data", enable_semantic_cache=False,
                        consolidation_idle_seconds=9999.0)
    config.ensure_dirs()
    bus = EventBus()
    registry = ToolRegistry(bus)
    ctx = KernelContext(bus=bus, config=config, tools=registry, data_dir=config.data_dir)
    pm = PluginManager(bus, config)
    ctx.services["plugin_manager"] = pm
    pm.register(SchedulerPlugin(config))
    pm.register(ServerOpsPlugin(config))
    await pm.start_all(ctx)
    try:
        scheduler = pm.get("scheduler")
        assert scheduler is not None
        jobs = await scheduler.list_jobs()
        payload = jobs.get("jobs", jobs)
        names = [j.get("name") for j in (payload.values() if isinstance(payload, dict) else payload)]
        assert "server-health" in names
    finally:
        for plugin in (ctx.service("server_ops"), ctx.service("scheduler")):
            if plugin is not None:
                await plugin.stop()
