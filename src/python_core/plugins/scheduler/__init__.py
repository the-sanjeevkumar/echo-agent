"""echo-agent Scheduler Plugin — architecture-level cron/scheduler system.

Provides:
- Recurring and one-shot scheduled tasks
- Event-driven scheduling via the bus (gateways, skills, other plugins can schedule)
- Persistence to data_dir/scheduler.json
- Tool exposure for CLI/gateway management (cron_add, cron_list, cron_remove, cron_run)
- Integration with Orchestrator for task execution
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from python_core.core.config import EchoConfig
from python_core.core.event_bus import EventBus
from python_core.core.interfaces import KernelContext, Plugin, ToolPlugin, ToolSpec
from python_core.orchestrator import Orchestrator

logger = logging.getLogger("echo.plugins.scheduler")


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class ScheduledJob:
    """A single scheduled job, persisted to disk and tracked in memory."""

    id: str
    name: str
    schedule: str          # cron syntax, "every 30m", "in 1h", ISO timestamp, etc.
    prompt: str            # self-contained task text for the orchestrator
    skills: list[str] = field(default_factory=list)  # skill names to load before run
    enabled: bool = True
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_run: str | None = None
    next_run: str | None = None
    run_count: int = 0
    max_runs: int | None = None  # None = unlimited
    session_id: str = "scheduler"  # session context for the orchestrator
    reply_channel: str | None = None  # chat_id to send response to (telegram/discord)
    platform: str | None = None  # "telegram" or "discord" - which gateway to use

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "schedule": self.schedule,
            "prompt": self.prompt,
            "skills": self.skills,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_run": self.last_run,
            "next_run": self.next_run,
            "run_count": self.run_count,
            "max_runs": self.max_runs,
            "session_id": self.session_id,
            "reply_channel": self.reply_channel,
            "platform": self.platform,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScheduledJob":
        return cls(
            id=data["id"],
            name=data["name"],
            schedule=data["schedule"],
            prompt=data["prompt"],
            skills=data.get("skills", []),
            enabled=data.get("enabled", True),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat()),
            last_run=data.get("last_run"),
            next_run=data.get("next_run"),
            run_count=data.get("run_count", 0),
            max_runs=data.get("max_runs"),
            session_id=data.get("session_id", "scheduler"),
            reply_channel=data.get("reply_channel"),
            platform=data.get("platform"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Schedule parsing (supports Hermes-like natural language + cron)
# ─────────────────────────────────────────────────────────────────────────────
import re
from datetime import timedelta


def parse_schedule(schedule: str, reference: datetime | None = None) -> datetime | None:
    """Parse a schedule string and return the NEXT run time (naive UTC).

    Supported formats:
      - cron: "0 9 * * *" (minute hour day month weekday)
      - interval: "every 30m", "every 2h", "every 1d", "every hour", "every day"
      - one-shot relative: "in 30m", "in 2h", "in 1d"
      - natural daily: "daily at 9am", "every day at 14:30"
      - natural weekly: "every monday 9am", "weekdays at 9am", "weekends at 10am"
      - ISO timestamp: "2026-06-01T09:00:00" (one-shot absolute)

    Returns None if the schedule is invalid or a one-shot that has already passed.
    """
    if reference is None:
        reference = datetime.now(timezone.utc).replace(tzinfo=None)

    schedule = schedule.strip().lower()

    # 1. ISO timestamp (absolute one-shot)
    try:
        dt = datetime.fromisoformat(schedule.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        if dt > reference:
            return dt
        return None  # already passed
    except ValueError:
        pass

    # 2. Cron syntax (5 or 6 fields)
    cron_match = re.match(r"^(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)(?:\s+(\S+))?$", schedule)
    if cron_match:
        return _next_cron(cron_match.groups(), reference)

    # 3. "every <interval>" — recurring interval
    interval_match = re.match(r"^every\s+(\d+)\s*([mhd])$", schedule)
    if interval_match:
        value, unit = int(interval_match.group(1)), interval_match.group(2)
        delta = {"m": timedelta(minutes=value), "h": timedelta(hours=value), "d": timedelta(days=value)}[unit]
        return reference + delta

    # 4. "every hour" / "every day" / "every week" / "every month"
    if schedule == "every hour":
        return reference.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    if schedule == "every day":
        return (reference + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    if schedule == "every week":
        # Next Monday 00:00
        days_ahead = (7 - reference.weekday()) % 7 or 7
        return (reference + timedelta(days=days_ahead)).replace(hour=0, minute=0, second=0, microsecond=0)
    if schedule == "every month":
        # First day of next month
        if reference.month == 12:
            return reference.replace(year=reference.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        return reference.replace(month=reference.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)

    # 5. "in <interval>" — one-shot relative
    in_match = re.match(r"^in\s+(\d+)\s*([mhd])$", schedule)
    if in_match:
        value, unit = int(in_match.group(1)), in_match.group(2)
        delta = {"m": timedelta(minutes=value), "h": timedelta(hours=value), "d": timedelta(days=value)}[unit]
        return reference + delta

    # 6. "daily at HH:MM" / "daily at H am/pm"
    daily_match = re.match(r"^daily\s+at\s+(\d{1,2}):?(\d{2})?\s*(am|pm)?$", schedule)
    if daily_match:
        hour = int(daily_match.group(1))
        minute = int(daily_match.group(2) or 0)
        ampm = daily_match.group(3)
        if ampm == "pm" and hour != 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        target = reference.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= reference:
            target += timedelta(days=1)
        return target

    # 7. "every <weekday> at HH:MM" / "weekdays at HH:MM" / "weekends at HH:MM"
    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    weekday_match = re.match(r"^every\s+(\w+)\s+at\s+(\d{1,2}):?(\d{2})?\s*(am|pm)?$", schedule)
    if weekday_match:
        day_name = weekday_match.group(1)
        if day_name in weekdays:
            target_weekday = weekdays.index(day_name)
            hour = int(weekday_match.group(2))
            minute = int(weekday_match.group(3) or 0)
            ampm = weekday_match.group(4)
            if ampm == "pm" and hour != 12:
                hour += 12
            elif ampm == "am" and hour == 12:
                hour = 0
            days_ahead = (target_weekday - reference.weekday()) % 7
            if days_ahead == 0:
                # Today — check time
                target = reference.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if target <= reference:
                    days_ahead = 7
            target = (reference + timedelta(days=days_ahead)).replace(hour=hour, minute=minute, second=0, microsecond=0)
            return target

    weekdays_match = re.match(r"^(weekdays|weekends)\s+at\s+(\d{1,2}):?(\d{2})?\s*(am|pm)?$", schedule)
    if weekdays_match:
        group = weekdays_match.group(1)
        hour = int(weekdays_match.group(2))
        minute = int(weekdays_match.group(3) or 0)
        ampm = weekdays_match.group(4)
        if ampm == "pm" and hour != 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        target = reference.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= reference:
            target += timedelta(days=1)
        # Find next matching weekday
        while True:
            wd = target.weekday()
            if (group == "weekdays" and wd < 5) or (group == "weekends" and wd >= 5):
                return target
            target += timedelta(days=1)

    return None  # Unknown format


def _next_cron(fields: tuple, reference: datetime) -> datetime | None:
    """Compute next cron match after reference. Simplified: only handles common patterns."""
    # This is a minimal cron matcher — for production, consider adding croniter as optional dep.
    # For now, support: * * * * * (every minute), 0 * * * * (hourly), 0 0 * * * (daily),
    # 0 0 * * 0 (weekly Sunday), 0 0 1 * * (monthly), and specific minute/hour/day/month/weekday.
    minute, hour, day, month, weekday = fields[:5]
    second = fields[5] if len(fields) > 5 else "0"

    # We'll do a simple forward search up to a reasonable limit (e.g., 1 year)
    current = reference.replace(second=0, microsecond=0)
    for _ in range(525600):  # max minutes in a year
        current += timedelta(minutes=1)
        if _cron_match(current, minute, hour, day, month, weekday):
            return current
    return None


def _cron_match(dt: datetime, minute: str, hour: str, day: str, month: str, weekday: str) -> bool:
    def match(field: str, value: int, max_val: int) -> bool:
        if field == "*":
            return True
        if "/" in field:
            base, step = field.split("/")
            if base != "*":
                return False
            return value % int(step) == 0
        # Comma-separated list
        for part in field.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                if start <= value <= end:
                    return True
            elif int(part) == value:
                return True
        return False

    return (
        match(minute, dt.minute, 59)
        and match(hour, dt.hour, 23)
        and match(day, dt.day, 31)
        and match(month, dt.month, 12)
        and match(weekday, dt.weekday(), 6)  # Monday=0 in cron? We'll treat 0=Sunday for compatibility
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler Plugin
# ─────────────────────────────────────────────────────────────────────────────
class SchedulerPlugin:
    """Cron/scheduler plugin for echo-agent.

    - Persists jobs to <data_dir>/scheduler.json
    - Runs a background task that wakes up every minute and fires due jobs
    - Exposes tools: cron_add, cron_list, cron_remove, cron_enable, cron_disable, cron_run
    - Listens for bus events: "scheduler:add", "scheduler:remove", "scheduler:list"
    """

    name = "scheduler"
    version = "0.1.0"
    capabilities = ["cron", "scheduling", "background-tasks", "tools"]

    def __init__(self, config: EchoConfig) -> None:
        self.config = config
        self._jobs: dict[str, ScheduledJob] = {}
        self._scheduler_task: asyncio.Task | None = None
        self._orchestrator: Orchestrator | None = None
        self._ctx: KernelContext | None = None
        self._data_file = config.data_dir / "scheduler.json"
        self._bus: EventBus | None = None
        self._registry: Any = None  # ToolRegistry for tool registration

    # ── Plugin lifecycle ────────────────────────────────────────────────────
    def start(self, ctx: KernelContext) -> None:
        self._ctx = ctx
        self._bus = ctx.bus
        self._registry = ctx.tools
        self._load_jobs()

        # Register scheduler tools with the tool registry
        for tool_spec in SCHEDULER_TOOLS:
            # Bind the tool handler to this instance
            handler_name = tool_spec.name
            handler = getattr(self, f"_tool_{handler_name}")
            tool_spec.handler = handler
            self._registry.register(tool_spec, handler)

        # Find the orchestrator (registered as a plugin)
        self._orchestrator = ctx.service("orchestrator")
        if self._orchestrator is None:
            # Fallback: scan plugins for one with agent-loop capability
            pm = ctx.service("plugin_manager")
            if pm:
                for name in pm.names():
                    plugin = pm.get(name)
                    if plugin and "agent-loop" in getattr(plugin, "capabilities", []):
                        self._orchestrator = plugin
                        break

        if self._orchestrator is None:
            logger.warning("scheduler: no orchestrator found — jobs will not execute")

        # Subscribe to bus events for external scheduling
        assert self._bus is not None
        self._bus.subscribe("scheduler:add", self._on_add, label="scheduler.add")
        self._bus.subscribe("scheduler:remove", self._on_remove, label="scheduler.remove")
        self._bus.subscribe("scheduler:list", self._on_list, label="scheduler.list")
        self._bus.subscribe("scheduler:run", self._on_run, label="scheduler.run")
        self._bus.subscribe("scheduler:enable", self._on_enable, label="scheduler.enable")
        self._bus.subscribe("scheduler:disable", self._on_disable, label="scheduler.disable")

        # Start background scheduler loop
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("scheduler plugin started with %d jobs", len(self._jobs))

    def stop(self) -> None:
        if self._scheduler_task:
            self._scheduler_task.cancel()
            self._scheduler_task = None
        # Unregister tools
        if self._registry:
            for tool_spec in SCHEDULER_TOOLS:
                self._registry.unregister(tool_spec.name)
        self._save_jobs()
        logger.info("scheduler plugin stopped")

    # ── Tool handlers ───────────────────────────────────────────────────────
    async def _tool_cron_add(self, args: dict[str, Any]) -> dict[str, Any]:
        return await self.add_job(
            name=args.get("name", "unnamed"),
            schedule=args.get("schedule", ""),
            prompt=args.get("prompt", ""),
            skills=args.get("skills", []),
            max_runs=args.get("max_runs"),
            session_id=args.get("session_id", "scheduler"),
            reply_channel=args.get("reply_channel"),
            platform=args.get("platform"),
        )

    async def _tool_cron_list(self, args: dict[str, Any]) -> dict[str, Any]:
        return await self.list_jobs()

    async def _tool_cron_remove(self, args: dict[str, Any]) -> dict[str, Any]:
        return await self.remove_job(args.get("job_id", ""))

    async def _tool_cron_enable(self, args: dict[str, Any]) -> dict[str, Any]:
        return await self.enable_job(args.get("job_id", ""))

    async def _tool_cron_disable(self, args: dict[str, Any]) -> dict[str, Any]:
        return await self.disable_job(args.get("job_id", ""))

    async def _tool_cron_run(self, args: dict[str, Any]) -> dict[str, Any]:
        return await self.run_job_now(args.get("job_id", ""))

    # ── Persistence ─────────────────────────────────────────────────────────
    def _load_jobs(self) -> None:
        if self._data_file.is_file():
            try:
                raw = json.loads(self._data_file.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    for job_data in raw:
                        job = ScheduledJob.from_dict(job_data)
                        self._jobs[job.id] = job
                logger.info("scheduler: loaded %d jobs from %s", len(self._jobs), self._data_file)
            except (OSError, ValueError) as exc:
                logger.warning("scheduler: failed to load jobs: %s", exc)

    def _save_jobs(self) -> None:
        try:
            self._data_file.parent.mkdir(parents=True, exist_ok=True)
            data = [job.to_dict() for job in self._jobs.values()]
            tmp = self._data_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._data_file)
        except OSError as exc:
            logger.error("scheduler: failed to save jobs: %s", exc)

    # ── Background scheduler loop ───────────────────────────────────────────
    async def _scheduler_loop(self) -> None:
        """Wake up every minute, check for due jobs, execute them."""
        try:
            while True:
                now = datetime.now(timezone.utc).replace(tzinfo=None)
                due_jobs = []

                for job in self._jobs.values():
                    if not job.enabled:
                        continue
                    if job.max_runs is not None and job.run_count >= job.max_runs:
                        continue
                    if job.next_run is None:
                        # First run — compute next_run
                        next_run = parse_schedule(job.schedule, now)
                        if next_run:
                            job.next_run = next_run.isoformat()
                            self._save_jobs()
                        continue
                    try:
                        next_run = datetime.fromisoformat(job.next_run)
                    except ValueError:
                        continue
                    if next_run <= now:
                        due_jobs.append(job)

                for job in due_jobs:
                    await self._execute_job(job, now)

                # Sleep until next minute boundary
                await asyncio.sleep(60 - now.second + 0.1)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.exception("scheduler loop crashed: %s", exc)

    async def _execute_job(self, job: ScheduledJob, now: datetime) -> None:
        """Execute a single scheduled job via the orchestrator."""
        if self._orchestrator is None:
            logger.error("scheduler: no orchestrator, skipping job %s", job.id)
            return

        logger.info("scheduler: executing job %s (%s)", job.id, job.name)

        # Update job state
        job.last_run = now.isoformat()
        job.run_count += 1

        # Compute next run
        next_run = parse_schedule(job.schedule, now)
        if next_run and (job.max_runs is None or job.run_count < job.max_runs):
            job.next_run = next_run.isoformat()
        else:
            job.next_run = None
            job.enabled = False  # auto-disable if max_runs reached

        self._save_jobs()

        # Publish job-started event so gateways can notify the originating chat
        if self._bus:
            self._bus.publish("scheduler:job_started", {
                "job_id": job.id,
                "job_name": job.name,
                "schedule": job.schedule,
                "reply_channel": job.reply_channel,
                "platform": job.platform,
            }, source="scheduler")

        # Execute via orchestrator with reply_channel and platform for gateway routing
        try:
            result = await self._orchestrator.handle_task(
                job.prompt,
                source="scheduler",
                session_id=job.session_id,
                reply_channel=job.reply_channel,
                platform=job.platform,
            )
            logger.info("scheduler: job %s completed (cached=%s)", job.id, result.get("cached", False))

            # Publish completion event
            if self._bus:
                self._bus.publish("scheduler:job_completed", {
                    "job_id": job.id,
                    "job_name": job.name,
                    "result": result,
                    "run_count": job.run_count,
                }, source="scheduler")

        except Exception as exc:  # noqa: BLE001
            logger.exception("scheduler: job %s failed: %s", job.id, exc)
            if self._bus:
                self._bus.publish("scheduler:job_failed", {
                    "job_id": job.id,
                    "job_name": job.name,
                    "error": str(exc),
                }, source="scheduler")

    # ── Bus event handlers ──────────────────────────────────────────────────
    async def _on_add(self, event) -> dict[str, Any]:
        payload = event.payload if hasattr(event, "payload") else event
        return await self.add_job(
            name=payload.get("name", "unnamed"),
            schedule=payload.get("schedule", ""),
            prompt=payload.get("prompt", ""),
            skills=payload.get("skills", []),
            max_runs=payload.get("max_runs"),
            session_id=payload.get("session_id", "scheduler"),
            reply_channel=payload.get("reply_channel"),
            platform=payload.get("platform"),
        )

    async def _on_remove(self, event) -> dict[str, Any]:
        payload = event.payload if hasattr(event, "payload") else event
        return await self.remove_job(payload.get("job_id", ""))

    async def _on_list(self, event) -> dict[str, Any]:
        return {"jobs": [job.to_dict() for job in self._jobs.values()]}

    async def _on_run(self, event) -> dict[str, Any]:
        payload = event.payload if hasattr(event, "payload") else event
        job_id = payload.get("job_id")
        job = self._jobs.get(job_id)
        if not job:
            return {"ok": False, "error": f"job not found: {job_id}"}
        await self._execute_job(job, datetime.now(timezone.utc).replace(tzinfo=None))
        return {"ok": True, "job": job.to_dict()}

    async def _on_enable(self, event) -> dict[str, Any]:
        payload = event.payload if hasattr(event, "payload") else event
        job = self._jobs.get(payload.get("job_id", ""))
        if job:
            job.enabled = True
            job.updated_at = datetime.now(timezone.utc).isoformat()
            next_run_dt = parse_schedule(job.schedule, datetime.now(timezone.utc).replace(tzinfo=None))
            if next_run_dt:
                job.next_run = next_run_dt.isoformat()
            self._save_jobs()
            return {"ok": True, "job": job.to_dict()}
        return {"ok": False, "error": "job not found"}

    async def _on_disable(self, event) -> dict[str, Any]:
        payload = event.payload if hasattr(event, "payload") else event
        job = self._jobs.get(payload.get("job_id", ""))
        if job:
            job.enabled = False
            job.updated_at = datetime.now(timezone.utc).isoformat()
            self._save_jobs()
            return {"ok": True, "job": job.to_dict()}
        return {"ok": False, "error": "job not found"}

    # ── Public API (used by tools) ──────────────────────────────────────────
    async def add_job(
        self,
        name: str,
        schedule: str,
        prompt: str,
        skills: list[str] | None = None,
        max_runs: int | None = None,
        session_id: str = "scheduler",
        reply_channel: str | None = None,
        platform: str | None = None,
    ) -> dict[str, Any]:
        """Add a new scheduled job."""
        if not schedule or not prompt:
            return {"ok": False, "error": "schedule and prompt are required"}

        # Validate schedule by parsing it
        test_next = parse_schedule(schedule, datetime.now(timezone.utc).replace(tzinfo=None))
        if test_next is None and not self._looks_like_cron(schedule):
            return {"ok": False, "error": f"invalid schedule format: {schedule}"}

        import uuid
        job_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        next_run = parse_schedule(schedule, now)
        next_run_str = next_run.isoformat() if next_run else None

        job = ScheduledJob(
            id=job_id,
            name=name,
            schedule=schedule,
            prompt=prompt,
            skills=skills or [],
            max_runs=max_runs,
            session_id=session_id,
            next_run=next_run_str,
            reply_channel=reply_channel,
            platform=platform,
        )
        self._jobs[job_id] = job
        self._save_jobs()
        logger.info("scheduler: added job %s (%s)", job_id, name)
        return {"ok": True, "job": job.to_dict()}

    def _looks_like_cron(self, schedule: str) -> bool:
        # Heuristic: 5 or 6 space-separated fields
        parts = schedule.split()
        return len(parts) in (5, 6) and all(p != "" for p in parts)

    async def remove_job(self, job_id: str) -> dict[str, Any]:
        if job_id in self._jobs:
            del self._jobs[job_id]
            self._save_jobs()
            return {"ok": True, "removed": job_id}
        return {"ok": False, "error": f"job not found: {job_id}"}

    async def list_jobs(self) -> dict[str, Any]:
        return {"jobs": [job.to_dict() for job in self._jobs.values()]}

    async def enable_job(self, job_id: str) -> dict[str, Any]:
        job = self._jobs.get(job_id)
        if job:
            job.enabled = True
            job.updated_at = datetime.now(timezone.utc).isoformat()
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            next_run = parse_schedule(job.schedule, now)
            if next_run:
                job.next_run = next_run.isoformat()
            self._save_jobs()
            return {"ok": True, "job": job.to_dict()}
        return {"ok": False, "error": "job not found"}

    async def disable_job(self, job_id: str) -> dict[str, Any]:
        job = self._jobs.get(job_id)
        if job:
            job.enabled = False
            job.updated_at = datetime.now(timezone.utc).isoformat()
            self._save_jobs()
            return {"ok": True, "job": job.to_dict()}
        return {"ok": False, "error": "job not found"}

    async def run_job_now(self, job_id: str) -> dict[str, Any]:
        job = self._jobs.get(job_id)
        if not job:
            return {"ok": False, "error": f"job not found: {job_id}"}
        await self._execute_job(job, datetime.now(timezone.utc).replace(tzinfo=None))
        return {"ok": True, "job": job.to_dict()}


# ─────────────────────────────────────────────────────────────────────────────
# Tool specifications (exposed to the orchestrator / LLM)
# ─────────────────────────────────────────────────────────────────────────────
SCHEDULER_TOOLS = [
    ToolSpec(
        name="cron_add",
        description=(
            "Add a scheduled cron job. Schedule formats: 'every 30m', 'every 2h', 'daily at 9am', "
            "'every monday 9am', 'weekdays at 9am', '0 9 * * *' (cron), 'in 1h' (one-shot), "
            "'2026-06-01T09:00:00' (absolute). Prompt must be self-contained. "
            "Optionally provide reply_channel (chat_id) and platform ('telegram' or 'discord') "
            "to route the response back to the originating chat."
        ),
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Human-friendly name"},
                "schedule": {"type": "string", "description": "Schedule expression (see description)"},
                "prompt": {"type": "string", "description": "Full task text for the agent"},
                "skills": {"type": "array", "items": {"type": "string"}, "description": "Skill names to load"},
                "max_runs": {"type": "integer", "description": "Max executions (null = unlimited)"},
                "session_id": {"type": "string", "description": "Session context for the task"},
                "reply_channel": {"type": "string", "description": "Chat ID to send response to (telegram chat_id or discord channel_id)"},
                "platform": {"type": "string", "enum": ["telegram", "discord"], "description": "Platform to route response through"},
            },
            "required": ["name", "schedule", "prompt"],
        },
    ),
    ToolSpec(
        name="cron_list",
        description="List all scheduled cron jobs with their status and next run time.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolSpec(
        name="cron_remove",
        description="Remove a scheduled cron job by ID.",
        parameters={
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
    ),
    ToolSpec(
        name="cron_enable",
        description="Enable a paused cron job.",
        parameters={
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
    ),
    ToolSpec(
        name="cron_disable",
        description="Disable a cron job (pauses it, keeps configuration).",
        parameters={
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
    ),
    ToolSpec(
        name="cron_run",
        description="Execute a cron job immediately (one-off, does not affect schedule).",
        parameters={
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Plugin factory
# ─────────────────────────────────────────────────────────────────────────────
def get_plugins(config: EchoConfig) -> list[SchedulerPlugin]:
    """Factory for PluginManager discovery."""
    return [SchedulerPlugin(config)]


# For direct programmatic registration
PLUGIN = None  # Set by get_plugins or create_plugin


def create_plugin(config: EchoConfig) -> SchedulerPlugin:
    global PLUGIN
    PLUGIN = SchedulerPlugin(config)
    return PLUGIN