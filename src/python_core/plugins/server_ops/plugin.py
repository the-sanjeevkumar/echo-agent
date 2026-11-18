"""Server-ops pack plugin (PHASE 6).

Read-only server tools by default; mutating service restarts require an
allowlisted systemd unit AND a short-lived chat confirmation (`/confirm <id>`)
unless the unit is pre-approved in config. Every action is audited to
Episodic Memory. Health checks are scheduled through the existing scheduler
plugin and alert the owner chat on threshold breaches.
"""

from __future__ import annotations

import inspect
import json
import logging
import time
import uuid
from typing import Any

from python_core.core.config import EchoConfig
from python_core.core.event_bus import EventBus
from python_core.core.interfaces import KernelContext, ToolSpec

from . import metrics

logger = logging.getLogger("echo.server_ops")


class ServerOpsPlugin:
    name = "server_ops"
    version = "0.1.0"
    capabilities = ["server:metrics", "server:service-status", "server:health-check"]

    def __init__(self, config: EchoConfig | None = None) -> None:
        self.config = config or EchoConfig.load()
        self._ctx: KernelContext | None = None
        self._bus: EventBus | None = None
        self._subs: list[Any] = []
        self.confirmations: dict[str, dict[str, Any]] = {}

    # ── lifecycle ─────────────────────────────────────────────────────────
    async def start(self, ctx: KernelContext) -> None:
        self._ctx = ctx
        self._bus = ctx.bus
        self.config = ctx.config
        ctx.services["server_ops"] = self

        if not bool(getattr(self.config, "server_ops_enabled", True)):
            ctx.logger.info("server_ops disabled by config")
            return

        tools = [
            ("sys_metrics", "CPU, RAM, disk, load average and uptime of this machine.", {}, self._tool_sys_metrics),
            ("service_status", "Read-only status of an allowlisted systemd unit.",
             {"unit": {"type": "string"}}, self._tool_service_status),
            ("service_restart", "Restart an allowlisted systemd unit (requires /confirm).",
             {"unit": {"type": "string"}, "confirm_id": {"type": "string"}}, self._tool_service_restart),
            ("journal_tail", "Tail journald logs for an allowlisted unit.",
             {"unit": {"type": "string"}, "lines": {"type": "integer"}}, self._tool_journal_tail),
            ("disk_usage", "Disk usage for a path.",
             {"path": {"type": "string"}}, self._tool_disk_usage),
            ("process_list", "Top processes by resident memory.",
             {"limit": {"type": "integer"}}, self._tool_process_list),
            ("docker_ps", "List running Docker containers (auto-detects docker).", {}, self._tool_docker_ps),
            ("server_health", "Run the configured health checks and alert on breaches.", {}, self._tool_health),
        ]
        for name, description, properties, handler in tools:
            await ctx.bus.emit("tool:register", {
                "spec": ToolSpec(
                    name=name,
                    description=description,
                    parameters={
                        "type": "object",
                        "properties": properties,
                        "required": ["unit"] if "unit" in properties else [],
                    },
                    source="plugin:server_ops",
                    tags=["server", "ops"],
                ),
                "handler": handler,
            })

        # Cron health checks via the existing scheduler: the job's start event
        # triggers the local check (no LLM call needed).
        scheduler = ctx.service("scheduler")
        if scheduler is None:
            pm = ctx.service("plugin_manager")
            if pm is not None and hasattr(pm, "get"):
                scheduler = pm.get("scheduler")
        if scheduler is not None and hasattr(scheduler, "add_job"):
            try:
                await scheduler.add_job(
                    "server-health",
                    getattr(self.config, "server_ops_health_interval", "every 5m"),
                    "Run the server health check and alert the owner on threshold breaches.",
                    session_id="server-health",
                    reply_channel=getattr(self.config, "server_ops_alerts_reply_channel", "") or None,
                )
            except Exception:  # health cron registration is best-effort
                logger.debug("could not register health cron", exc_info=True)

        async def _on_job_started(event: Any) -> None:
            payload = getattr(event, "payload", event) or {}
            if str(payload.get("job_name", "")) == "server-health":
                await self.check_health()

        self._subs.append(ctx.bus.subscribe("scheduler:job_started", _on_job_started, label="server_ops.cron"))

    async def stop(self) -> None:
        for sub in self._subs:
            if self._bus is not None:
                self._bus.unsubscribe(sub)
        self._subs.clear()

    # ── helpers ───────────────────────────────────────────────────────────
    @staticmethod
    async def _call(fn: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Support both sync and async metric backends (tests inject fakes)."""
        result = fn(*args, **kwargs)
        return await result if inspect.isawaitable(result) else result

    # ── safety ────────────────────────────────────────────────────────────
    def _unit_allowed(self, unit: str) -> bool:
        allowed = {str(u).strip() for u in getattr(self.config, "server_ops_units", []) if str(u).strip()}
        return str(unit).strip() in allowed

    def _preapproved(self, unit: str) -> bool:
        approved = {
            str(u).strip() for u in getattr(self.config, "server_ops_preapproved_units", []) if str(u).strip()
        }
        return str(unit).strip() in approved

    def _audit(self, action: str, outcome: dict[str, Any]) -> None:
        if self._ctx is None:
            return
        memory = self._ctx.service("memory")
        if memory is None or not hasattr(memory, "append_episode"):
            return
        try:
            memory.append_episode(
                f"server_ops:{action}",
                json.dumps(outcome, default=str)[:600],
                success=bool(outcome.get("ok", True)),
                metadata={"plugin": "server_ops", "action": action},
            )
        except Exception:  # audit failures never break ops
            logger.debug("server_ops audit failed", exc_info=True)

    # ── tool handlers ─────────────────────────────────────────────────────
    def _tool_sys_metrics(self, args: dict[str, Any]) -> dict[str, Any]:
        result = {"ok": True, **metrics.sys_metrics()}
        self._audit("sys_metrics", result)
        return result

    async def _tool_service_status(self, args: dict[str, Any]) -> dict[str, Any]:
        unit = str(args.get("unit", "")).strip()
        if not self._unit_allowed(unit):
            result = {"ok": False, "error": f"unit {unit!r} is not allowlisted"}
        else:
            result = await self._call(metrics.unit_status, unit)
        self._audit("service_status", result)
        return result

    async def _tool_service_restart(self, args: dict[str, Any]) -> dict[str, Any]:
        unit = str(args.get("unit", "")).strip()
        confirm_id = str(args.get("confirm_id") or "").strip()
        if not self._unit_allowed(unit):
            result = {"ok": False, "error": f"unit {unit!r} is not allowlisted"}
        elif self._preapproved(unit) or self._is_confirmed(confirm_id, unit):
            result = await self._call(metrics.unit_restart, unit)
        else:
            pend_id = f"sc-{uuid.uuid4().hex[:10]}"
            self.confirmations[pend_id] = {
                "unit": unit, "ts": time.time(),
                "expires": time.time() + float(getattr(self.config, "server_ops_confirm_ttl", 120.0)),
            }
            result = {
                "ok": True, "pending": True, "confirmation_id": pend_id,
                "message": f"Reply /confirm {pend_id} to restart {unit}.",
            }
        self._audit("service_restart", result)
        return result

    def _is_confirmed(self, confirm_id: str, unit: str) -> bool:
        entry = self.confirmations.pop(confirm_id, None)
        return bool(entry and entry["unit"] == unit and time.time() <= entry["expires"])

    async def confirm(self, confirm_id: str) -> dict[str, Any]:
        """Complete a pending restart after chat confirmation."""
        entry = self.confirmations.get(str(confirm_id).strip())
        if entry is None:
            return {"ok": False, "error": f"unknown or expired confirmation {confirm_id}"}
        if time.time() > entry["expires"]:
            self.confirmations.pop(confirm_id, None)
            return {"ok": False, "error": f"confirmation {confirm_id} expired"}
        unit = entry["unit"]
        if not self._unit_allowed(unit):
            self.confirmations.pop(confirm_id, None)
            return {"ok": False, "error": f"unit {unit!r} is not allowlisted"}
        self.confirmations.pop(confirm_id, None)
        result = await self._call(metrics.unit_restart, unit)
        self._audit("service_restart:confirmed", result)
        return {"ok": bool(result.get("ok")), **result}

    async def _tool_journal_tail(self, args: dict[str, Any]) -> dict[str, Any]:
        unit = str(args.get("unit", "")).strip()
        if not self._unit_allowed(unit):
            result = {"ok": False, "error": f"unit {unit!r} is not allowlisted"}
        else:
            result = await self._call(metrics.journal_tail, unit, int(args.get("lines", 50)))
        self._audit("journal_tail", result)
        return result

    def _tool_disk_usage(self, args: dict[str, Any]) -> dict[str, Any]:
        result = {"ok": True, **metrics.disk_usage(str(args.get("path", "/") or "/"))}
        self._audit("disk_usage", result)
        return result

    def _tool_process_list(self, args: dict[str, Any]) -> dict[str, Any]:
        result = {"ok": True, **metrics.process_list(int(args.get("limit", 20)))}
        self._audit("process_list", result)
        return result

    def _tool_docker_ps(self, args: dict[str, Any]) -> dict[str, Any]:
        result = metrics.docker_ps()
        self._audit("docker_ps", result)
        return result

    def _tool_health(self, args: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, **self.check_health()}

    # ── health checks / alerts ────────────────────────────────────────────
    def check_health(self) -> dict[str, Any]:
        """Check disk/RAM thresholds and allowlisted units; publish alerts."""
        alerts: list[dict[str, Any]] = []
        snap = metrics.sys_metrics()
        disk_pct = float(snap.get("disk", {}).get("percent", 0))
        ram_pct = float(snap.get("ram", {}).get("percent", 0))
        disk_limit = float(getattr(self.config, "server_ops_disk_threshold", 90.0))
        ram_limit = float(getattr(self.config, "server_ops_ram_threshold", 90.0))
        if disk_pct >= disk_limit:
            alerts.append({"severity": "warning", "kind": "disk", "message": f"Disk usage {disk_pct}% >= {disk_limit}%"})
        if ram_pct >= ram_limit:
            alerts.append({"severity": "warning", "kind": "ram", "message": f"RAM usage {ram_pct}% >= {ram_limit}%"})
        for unit in getattr(self.config, "server_ops_units", []):
            status = metrics.unit_status(str(unit))
            if status.get("active") not in ("active", "unknown"):
                alerts.append({
                    "severity": "critical", "kind": "service",
                    "message": f"systemd unit {unit} is {status.get('active')}",
                })
        report = {"checked": True, "alerts": alerts, "metrics": snap}
        self._audit("health_check", {"ok": not alerts, "alerts": alerts})
        for alert in alerts:
            payload = {
                **alert,
                "reply_channel": getattr(self.config, "server_ops_alerts_reply_channel", "") or None,
                "platform": str(getattr(self.config, "server_ops_alerts_reply_channel", "") or "").split(":")[0] or None,
            }
            if self._bus is not None:
                self._bus.publish("server:alert", payload, source="server_ops")
        return report


def create_plugin(config: EchoConfig | None = None) -> ServerOpsPlugin:
    return ServerOpsPlugin(config)
