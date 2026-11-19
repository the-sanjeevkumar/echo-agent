"""Supervisor mode: registry, health, delegation, heartbeat + digest (PHASE 7).

Reuses the existing manifest protocol / universal client (JSON-RPC /execute)
and the scheduler cron. Agents are registered in ``<data_dir>/agents.json``.

Delegation progress is streamed as Phase-5 ``task:progress`` events with
kind="supervisor"; outcomes feed reliability lessons into MEMORY.md through
the curated-memory tool path (Phase 4).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

from python_core.core.config import EchoConfig
from python_core.core.event_bus import EventBus
from python_core.core.interfaces import KernelContext, ToolSpec
from python_core.inter_agent.universal_client import UniversalAgentClient

logger = logging.getLogger("echo.supervisor")


class SupervisorPlugin:
    name = "supervisor"
    version = "0.1.0"
    capabilities = ["supervisor", "agent-registry", "delegation"]

    def __init__(self, config: EchoConfig | None = None) -> None:
        self.config = config or EchoConfig.load()
        self._transport: Any | None = None  # injectable for tests (httpx.MockTransport)
        self.client = UniversalAgentClient(transport=self._transport)
        self._ctx: KernelContext | None = None
        self._bus: EventBus | None = None
        self._subs: list[Any] = []
        self.registry_path = Path(self.config.data_dir) / "agents.json"
        self.agents: dict[str, dict[str, Any]] = self._load()
        self.reliability: dict[str, dict[str, int]] = {}
        self._progress_ctx: dict[str, Any] = {}

    # ── persistence ──────────────────────────────────────────────────────
    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): v for k, v in data.items() if isinstance(v, dict)}

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".agents-", dir=str(self.registry_path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.agents, fh, ensure_ascii=False, indent=2)
            os.replace(tmp_name, self.registry_path)
            tmp_name = None
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass

    # ── lifecycle ────────────────────────────────────────────────────────
    async def start(self, ctx: KernelContext) -> None:
        self._ctx = ctx
        self._bus = ctx.bus
        self.config = ctx.config
        self.registry_path = Path(ctx.config.data_dir) / "agents.json"
        self.agents = self._load()
        ctx.services["supervisor"] = self

        tools = [
            ("agents_list", "List registered agents.", {}, self._tool_list),
            ("agent_register", "Register an agent by name, base URL and optional token.",
             {"name": {"type": "string"}, "url": {"type": "string"}, "token": {"type": "string"}},
             self._tool_register),
            ("agent_health", "Check whether a registered agent is reachable.",
             {"name": {"type": "string"}}, self._tool_health),
            ("agent_delegate", "Delegate a task to a registered agent over the manifest protocol.",
             {"name": {"type": "string"}, "task": {"type": "string"}}, self._tool_delegate),
        ]
        for name, description, properties, handler in tools:
            await ctx.bus.emit("tool:register", {
                "spec": ToolSpec(
                    name=name, description=description,
                    parameters={
                        "type": "object", "properties": properties,
                        "required": ["name"] if "name" in properties else [],
                    },
                    source="plugin:supervisor", tags=["supervisor", "agents"],
                ),
                "handler": handler,
            })

        # Cron heartbeat + daily digest through the existing scheduler.
        scheduler = ctx.service("scheduler")
        if scheduler is None:
            pm = ctx.service("plugin_manager")
            if pm is not None and hasattr(pm, "get"):
                scheduler = pm.get("scheduler")
        if scheduler is not None and hasattr(scheduler, "add_job"):
            for job_name, schedule in (
                ("agent-heartbeat", getattr(ctx.config, "supervisor_heartbeat_interval", "every 5m")),
                ("agent-digest", getattr(ctx.config, "supervisor_digest_schedule", "daily at 9am")),
            ):
                try:
                    await scheduler.add_job(
                        job_name, schedule, f"Supervisor {job_name}.", session_id=job_name,
                        reply_channel=getattr(ctx.config, "supervisor_alerts_reply_channel", "") or None,
                    )
                except Exception:  # noqa: BLE001
                    logger.debug("could not register %s cron", job_name, exc_info=True)

        async def _on_job(event: Any) -> None:
            payload = getattr(event, "payload", event) or {}
            job_name = str(payload.get("job_name", ""))
            if job_name == "agent-heartbeat":
                await self.heartbeat()
            elif job_name == "agent-digest":
                await self.digest()

        self._subs.append(ctx.bus.subscribe("scheduler:job_started", _on_job, label="supervisor.cron"))

        async def _capture_progress(event: Any) -> None:
            payload = getattr(event, "payload", event) or {}
            if payload.get("reply_channel"):
                self._progress_ctx = {
                    k: payload.get(k) for k in ("reply_channel", "platform", "session_id") if payload.get(k)
                }

        self._subs.append(ctx.bus.subscribe("task:progress", _capture_progress, label="supervisor.progress-ctx"))

    async def stop(self) -> None:
        for sub in self._subs:
            if self._bus is not None:
                self._bus.unsubscribe(sub)
        self._subs.clear()

    # ── registry API ─────────────────────────────────────────────────────
    def register(self, name: str, url: str, token: str = "") -> dict[str, Any]:
        name = str(name).strip()
        url = str(url).strip().rstrip("/")
        if not name or not url.startswith(("http://", "https://")):
            return {"ok": False, "error": "name and http(s) url are required"}
        self.agents[name] = {"name": name, "url": url, "token": token, "registered_at": time.time()}
        self._save()
        return {"ok": True, "agent": self.agents[name]}

    def _get(self, name: str) -> dict[str, Any] | None:
        return self.agents.get(str(name).strip())

    def _headers(self, agent: dict[str, Any]) -> dict[str, str]:
        token = str(agent.get("token") or "")
        return {"Authorization": f"Bearer {token}"} if token else {}

    # ── health / delegation ──────────────────────────────────────────────
    async def _http_get(self, url: str, headers: dict[str, str]) -> httpx.Response:
        async with httpx.AsyncClient(timeout=10.0) as client:
            return await client.get(url, headers=headers)

    async def health(self, name: str) -> dict[str, Any]:
        agent = self._get(name)
        if agent is None:
            return {"ok": False, "error": f"unknown agent {name!r}"}
        url = str(agent["url"]).rstrip("/")
        retries = max(0, int(getattr(self.config, "supervisor_retries", 2)))
        last_error = ""
        for attempt in range(retries + 1):
            try:
                resp = await self._http_get(url + "/health", headers=self._headers(agent))
                if resp.status_code == 200:
                    data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                    return {"ok": True, "name": agent["name"], "status": data.get("status", "ok"), "attempts": attempt + 1}
                last_error = f"HTTP {resp.status_code}"
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                await asyncio.sleep(0.2 * (2 ** attempt))  # bounded backoff
        return {"ok": False, "name": agent["name"], "error": last_error, "attempts": retries + 1}

    async def delegate(self, name: str, task: str, *, reply_channel: str | None = None) -> dict[str, Any]:
        agent = self._get(name)
        if agent is None:
            return {"ok": False, "error": f"unknown agent {name!r}"}
        # Phase 5: stream delegation progress.
        await self._progress(f"🛰 Delegated to {agent['name']}", reply_channel=reply_channel)
        retries = max(0, int(getattr(self.config, "supervisor_retries", 2)))
        last_error = ""
        for attempt in range(retries + 1):
            result = await self.client.execute(
                str(agent["url"]).rstrip("/"), "task",
                {"task": task, "text": task},
                token=str(agent.get("token") or "") or None,
            )
            if result.get("ok"):
                self._record_reliability(agent["name"], success=True)
                await self._lesson(agent["name"], "succeeded", task)
                await self._progress(f"🛰 {agent['name']} completed the delegation", reply_channel=reply_channel)
                return {"ok": True, "agent": agent["name"], "result": result.get("result"), "attempts": attempt + 1}
            last_error = str(result.get("error", "delegation failed"))
            if attempt < retries:
                await asyncio.sleep(0.2 * (2 ** attempt))
        self._record_reliability(agent["name"], success=False)
        await self._lesson(agent["name"], f"failed: {last_error[:160]}", task)
        await self._alert(f"Agent {agent['name']} delegation failed after {retries + 1} attempts: {last_error[:160]}")
        return {"ok": False, "agent": agent["name"], "error": last_error, "attempts": retries + 1}

    # ── heartbeat / digest ───────────────────────────────────────────────
    async def heartbeat(self) -> dict[str, Any]:
        results = []
        for name in list(self.agents):
            status = await self.health(name)
            results.append({"name": name, **status})
            if not status.get("ok"):
                await self._alert(f"Agent {name} is unreachable: {status.get('error', 'unknown')}")
        if self._bus is not None:
            self._bus.publish("supervisor:heartbeat", {"results": results}, source="supervisor")
        return {"ok": True, "results": results}

    async def digest(self) -> dict[str, Any]:
        lines = []
        for name in list(self.agents):
            status = await self.health(name)
            stats = self.reliability.get(name, {"successes": 0, "failures": 0})
            lines.append(
                f"- {name}: {'online' if status.get('ok') else 'OFFLINE'} "
                f"({stats['successes']} ok / {stats['failures']} failed)"
            )
        text = "🛰 Supervisor daily digest\n" + ("\n".join(lines) if lines else "(no agents registered)")
        if self._bus is not None:
            self._bus.publish(
                "supervisor:digest",
                {"text": text, "reply_channel": getattr(self.config, "supervisor_alerts_reply_channel", "") or None,
                 "platform": str(getattr(self.config, "supervisor_alerts_reply_channel", "") or "").split(":")[0] or None},
                source="supervisor",
            )
        return {"ok": True, "text": text}

    # ── helpers ──────────────────────────────────────────────────────────
    def _record_reliability(self, name: str, *, success: bool) -> None:
        stats = self.reliability.setdefault(name, {"successes": 0, "failures": 0})
        stats["successes" if success else "failures"] += 1

    async def _lesson(self, name: str, outcome: str, task: str) -> None:
        """Feed per-agent reliability into MEMORY.md via the tool path."""
        memory = self._ctx.service("memory") if self._ctx else None
        if memory is None or not hasattr(memory, "curated_apply"):
            return
        try:
            memory.curated_apply("memory", [{
                "action": "add",
                "content": f"Agent {name} {outcome} on task: {str(task)[:120]}",
            }])
        except Exception:  # noqa: BLE001
            logger.debug("supervisor lesson write failed", exc_info=True)

    async def _progress(self, detail: str, *, reply_channel: str | None = None) -> None:
        if self._bus is None:
            return
        ctx = dict(self._progress_ctx)
        if reply_channel:
            ctx["reply_channel"] = reply_channel
        if not ctx.get("reply_channel"):
            ctx["reply_channel"] = getattr(self.config, "supervisor_alerts_reply_channel", "") or None
        payload = {"kind": "supervisor", "detail": detail[:400], **ctx}
        self._bus.publish("task:progress", payload, source="supervisor")
        self._bus.publish("supervisor:delegated", {"detail": detail}, source="supervisor")

    async def _alert(self, message: str) -> None:
        if self._bus is None:
            return
        self._bus.publish("supervisor:alert", {
            "severity": "critical", "kind": "agent", "message": message,
            "reply_channel": getattr(self.config, "supervisor_alerts_reply_channel", "") or None,
            "platform": str(getattr(self.config, "supervisor_alerts_reply_channel", "") or "").split(":")[0] or None,
        }, source="supervisor")

    # ── tool handlers ───────────────────────────────────────────────────
    def _tool_list(self, args: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "agents": [
            {"name": a.get("name"), "url": a.get("url")} for a in self.agents.values()
        ]}

    def _tool_register(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.register(str(args.get("name", "")), str(args.get("url", "")), str(args.get("token", "") or ""))

    async def _tool_health(self, args: dict[str, Any]) -> dict[str, Any]:
        return await self.health(str(args.get("name", "")))

    async def _tool_delegate(self, args: dict[str, Any]) -> dict[str, Any]:
        return await self.delegate(str(args.get("name", "")), str(args.get("task", "")))


def create_plugin(config: EchoConfig | None = None) -> SupervisorPlugin:
    return SupervisorPlugin(config)
