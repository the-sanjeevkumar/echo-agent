"""Tests for the Scheduler Plugin."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from python_core.core.config import EchoConfig
from python_core.plugins.scheduler import SchedulerPlugin, ScheduledJob, parse_schedule
from datetime import datetime, timezone


class TestScheduleParsing:
    """Test schedule string parsing."""

    def test_cron_syntax(self):
        """Test basic cron syntax parsing."""
        ref = datetime(2026, 1, 1, 12, 0, 0)
        # Hourly: 0 * * * *
        next_run = parse_schedule("0 * * * *", ref)
        assert next_run == datetime(2026, 1, 1, 13, 0, 0)

        # Daily at midnight: 0 0 * * *
        next_run = parse_schedule("0 0 * * *", ref)
        assert next_run == datetime(2026, 1, 2, 0, 0, 0)

    def test_every_interval(self):
        """Test 'every Xm/h/d' format."""
        ref = datetime(2026, 1, 1, 12, 0, 0)
        assert parse_schedule("every 30m", ref) == datetime(2026, 1, 1, 12, 30, 0)
        assert parse_schedule("every 2h", ref) == datetime(2026, 1, 1, 14, 0, 0)
        assert parse_schedule("every 1d", ref) == datetime(2026, 1, 2, 12, 0, 0)

    def test_every_keywords(self):
        """Test 'every hour/day/week/month' keywords."""
        ref = datetime(2026, 1, 1, 12, 30, 45)
        assert parse_schedule("every hour", ref) == datetime(2026, 1, 1, 13, 0, 0)
        assert parse_schedule("every day", ref) == datetime(2026, 1, 2, 0, 0, 0)

    def test_in_interval(self):
        """Test 'in Xm/h/d' one-shot format."""
        ref = datetime(2026, 1, 1, 12, 0, 0)
        assert parse_schedule("in 30m", ref) == datetime(2026, 1, 1, 12, 30, 0)
        assert parse_schedule("in 2h", ref) == datetime(2026, 1, 1, 14, 0, 0)
        assert parse_schedule("in 1d", ref) == datetime(2026, 1, 2, 12, 0, 0)

    def test_daily_at(self):
        """Test 'daily at HH:MM' format."""
        ref = datetime(2026, 1, 1, 12, 0, 0)
        # 14:30 hasn't passed yet today -> should run today
        assert parse_schedule("daily at 14:30", ref) == datetime(2026, 1, 1, 14, 30, 0)
        # 9:15am has passed (it's 12:00) -> should run tomorrow
        assert parse_schedule("daily at 9:15am", ref) == datetime(2026, 1, 2, 9, 15, 0)
        # 12pm is exactly now -> target <= reference, so tomorrow (implementation detail)
        assert parse_schedule("daily at 12pm", ref) == datetime(2026, 1, 2, 12, 0, 0)
        assert parse_schedule("daily at 12am", ref) == datetime(2026, 1, 2, 0, 0, 0)  # midnight passed -> tomorrow

        # If time already passed today, should be tomorrow
        ref_pm = datetime(2026, 1, 1, 14, 0, 0)
        assert parse_schedule("daily at 9am", ref_pm) == datetime(2026, 1, 2, 9, 0, 0)

    def test_weekly_at(self):
        """Test 'every weekday at HH:MM' format."""
        # Monday 2026-01-05
        ref = datetime(2026, 1, 5, 12, 0, 0)
        # Next Monday
        next_monday = parse_schedule("every monday at 9am", ref)
        assert next_monday == datetime(2026, 1, 12, 9, 0, 0)

        # If today is Monday and time hasn't passed
        ref_morning = datetime(2026, 1, 5, 8, 0, 0)
        next_monday = parse_schedule("every monday at 9am", ref_morning)
        assert next_monday == datetime(2026, 1, 5, 9, 0, 0)

    def test_weekdays_weekends(self):
        """Test 'weekdays at' and 'weekends at'."""
        # Monday
        ref = datetime(2026, 1, 5, 12, 0, 0)
        next_wd = parse_schedule("weekdays at 9am", ref)
        assert next_wd == datetime(2026, 1, 6, 9, 0, 0)  # Tuesday

        # Friday
        ref_fri = datetime(2026, 1, 9, 12, 0, 0)
        next_wd = parse_schedule("weekdays at 9am", ref_fri)
        assert next_wd == datetime(2026, 1, 12, 9, 0, 0)  # Monday

        # Saturday 12:00 - 9am already passed, should go to Sunday
        ref_sat = datetime(2026, 1, 10, 12, 0, 0)
        next_we = parse_schedule("weekends at 9am", ref_sat)
        assert next_we == datetime(2026, 1, 11, 9, 0, 0)  # Sunday

        # Saturday 8:00 - 9am hasn't passed, should run today
        ref_sat_morning = datetime(2026, 1, 10, 8, 0, 0)
        next_we = parse_schedule("weekends at 9am", ref_sat_morning)
        assert next_we == datetime(2026, 1, 10, 9, 0, 0)  # Today (Saturday)

    def test_iso_timestamp(self):
        """Test ISO timestamp parsing."""
        ref = datetime(2026, 1, 1, 12, 0, 0)
        future = datetime(2026, 6, 1, 9, 0, 0)
        result = parse_schedule("2026-06-01T09:00:00", ref)
        assert result == future

        # Past timestamp should return None
        past = datetime(2025, 1, 1, 0, 0, 0)
        result = parse_schedule("2025-01-01T00:00:00", ref)
        assert result is None


class TestScheduledJob:
    """Test ScheduledJob dataclass."""

    def test_to_from_dict(self):
        """Test round-trip serialization."""
        job = ScheduledJob(
            id="test123",
            name="Test Job",
            schedule="every 1h",
            prompt="Do something",
            skills=["web"],
            enabled=True,
            created_at="2026-01-01T00:00:00",
            updated_at="2026-01-01T00:00:00",
            last_run=None,
            next_run="2026-01-01T01:00:00",
            run_count=5,
            max_runs=10,
            session_id="test-session",
        )
        data = job.to_dict()
        restored = ScheduledJob.from_dict(data)
        assert restored.id == job.id
        assert restored.name == job.name
        assert restored.schedule == job.schedule
        assert restored.prompt == job.prompt
        assert restored.skills == job.skills
        assert restored.enabled == job.enabled
        assert restored.run_count == job.run_count
        assert restored.max_runs == job.max_runs
        assert restored.session_id == job.session_id


class TestSchedulerPlugin:
    """Test SchedulerPlugin functionality."""

    @pytest.fixture
    async def scheduler(self):
        """Create a scheduler plugin with a temp data dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EchoConfig(data_dir=Path(tmpdir))
            config.ensure_dirs()
            plugin = SchedulerPlugin(config)
            
            # Mock orchestrator
            class MockOrchestrator:
                async def handle_task(self, text, *, source="scheduler", session_id="scheduler", on_event=None, reply_channel=None, platform=None):
                    return {"response": f"Executed: {text}", "steps": [], "cached": False, "elapsed_ms": 10}
            
            # Mock kernel context with minimal services
            class MockBus:
                def __init__(self):
                    self.subs = []
                def subscribe(self, pattern, handler, *, priority=0, once=False, label=""):
                    from python_core.core.event_bus import Subscription
                    from itertools import count
                    _serial = count(1)
                    sub = Subscription(
                        token=next(_serial),
                        pattern=pattern,
                        handler=handler,
                        priority=priority,
                        once=once,
                        label=label,
                    )
                    self.subs.append(sub)
                    return sub
                def unsubscribe(self, sub):
                    if sub in self.subs:
                        self.subs.remove(sub)
                        return True
                    return False
                def publish(self, topic, payload, *, source="test"):
                    # Fire and forget for tests
                    pass
            
            class MockTools:
                def __init__(self):
                    self.tools = {}
                def register(self, spec, handler):
                    self.tools[spec.name] = (spec, handler)
                    return spec
                def unregister(self, name):
                    return self.tools.pop(name, None) is not None
            
            class MockCtx:
                def __init__(self):
                    self.bus = MockBus()
                    self.tools = MockTools()
                    self._orchestrator = MockOrchestrator()
                def service(self, name):
                    if name == "orchestrator":
                        return self._orchestrator
                    return None
            
            ctx = MockCtx()
            plugin.start(ctx)
            
            yield plugin
            
            plugin.stop()

    @pytest.mark.asyncio
    async def test_add_job(self, scheduler):
        """Test adding a job."""
        result = await scheduler.add_job(
            name="Test Job",
            schedule="every 1h",
            prompt="Do something",
        )
        assert result["ok"] is True
        assert result["job"]["name"] == "Test Job"
        assert result["job"]["schedule"] == "every 1h"
        assert result["job"]["prompt"] == "Do something"
        assert result["job"]["enabled"] is True
        assert result["job"]["run_count"] == 0
        assert "id" in result["job"]

    @pytest.mark.asyncio
    async def test_add_job_validation(self, scheduler):
        """Test job validation."""
        # Missing schedule
        result = await scheduler.add_job(name="Test", schedule="", prompt="Do something")
        assert result["ok"] is False
        assert "required" in result["error"].lower()

        # Missing prompt
        result = await scheduler.add_job(name="Test", schedule="every 1h", prompt="")
        assert result["ok"] is False

        # Invalid schedule
        result = await scheduler.add_job(name="Test", schedule="invalid", prompt="Do something")
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_list_jobs(self, scheduler):
        """Test listing jobs."""
        # Initially empty
        result = await scheduler.list_jobs()
        assert result["jobs"] == []

        # Add a job
        await scheduler.add_job(name="Job1", schedule="every 1h", prompt="Task 1")
        await scheduler.add_job(name="Job2", schedule="daily at 9am", prompt="Task 2")

        result = await scheduler.list_jobs()
        assert len(result["jobs"]) == 2
        names = {j["name"] for j in result["jobs"]}
        assert names == {"Job1", "Job2"}

    @pytest.mark.asyncio
    async def test_remove_job(self, scheduler):
        """Test removing a job."""
        add_result = await scheduler.add_job(name="To Remove", schedule="every 1h", prompt="Task")
        job_id = add_result["job"]["id"]

        # Remove it
        result = await scheduler.remove_job(job_id)
        assert result["ok"] is True
        assert result["removed"] == job_id

        # Verify gone
        list_result = await scheduler.list_jobs()
        assert len(list_result["jobs"]) == 0

        # Remove non-existent
        result = await scheduler.remove_job("nonexistent")
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_enable_disable_job(self, scheduler):
        """Test enabling/disabling a job."""
        add_result = await scheduler.add_job(name="Toggle", schedule="every 1h", prompt="Task")
        job_id = add_result["job"]["id"]

        # Disable
        result = await scheduler.disable_job(job_id)
        assert result["ok"] is True
        assert result["job"]["enabled"] is False

        # Enable
        result = await scheduler.enable_job(job_id)
        assert result["ok"] is True
        assert result["job"]["enabled"] is True

    @pytest.mark.asyncio
    async def test_run_job_now(self, scheduler):
        """Test running a job immediately."""
        add_result = await scheduler.add_job(name="Run Now", schedule="every 1h", prompt="Task")
        job_id = add_result["job"]["id"]

        result = await scheduler.run_job_now(job_id)
        assert result["ok"] is True
        assert result["job"]["run_count"] == 1
        assert result["job"]["last_run"] is not None

        # Non-existent
        result = await scheduler.run_job_now("nonexistent")
        assert result["ok"] is False

    @pytest.mark.asyncio
    async def test_persistence(self, scheduler):
        """Test job persistence to disk."""
        # Add jobs
        await scheduler.add_job(name="Persist1", schedule="every 1h", prompt="Task 1")
        await scheduler.add_job(name="Persist2", schedule="daily at 9am", prompt="Task 2")

        # Save happens automatically, but verify file exists
        assert scheduler._data_file.exists()

        # Load raw JSON
        with open(scheduler._data_file) as f:
            data = json.load(f)
        assert len(data) == 2
        names = {j["name"] for j in data}
        assert names == {"Persist1", "Persist2"}

    @pytest.mark.asyncio
    async def test_persistence_reload(self):
        """Test jobs are loaded on plugin restart."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EchoConfig(data_dir=Path(tmpdir))
            config.ensure_dirs()
            
            # First plugin instance
            plugin1 = SchedulerPlugin(config)
            
            class MockBus:
                def __init__(self):
                    self.subs = []
                def subscribe(self, pattern, handler, *, priority=0, once=False, label=""):
                    from python_core.core.event_bus import Subscription
                    from itertools import count
                    _serial = count(1)
                    sub = Subscription(
                        token=next(_serial),
                        pattern=pattern,
                        handler=handler,
                        priority=priority,
                        once=once,
                        label=label,
                    )
                    self.subs.append(sub)
                    return sub
                def unsubscribe(self, sub):
                    if sub in self.subs:
                        self.subs.remove(sub)
                        return True
                    return False
                def publish(self, topic, payload, *, source="test"):
                    # Fire and forget for tests
                    pass
            
            class MockTools:
                def __init__(self):
                    self.tools = {}
                def register(self, spec, handler):
                    self.tools[spec.name] = (spec, handler)
                    return spec
                def unregister(self, name):
                    return self.tools.pop(name, None) is not None
            
            class MockCtx:
                def __init__(self):
                    self.bus = MockBus()
                    self.tools = MockTools()
                def service(self, name):
                    return None
            
            ctx = MockCtx()
            plugin1.start(ctx)
            
            await plugin1.add_job(name="ReloadTest", schedule="every 1h", prompt="Task")
            plugin1.stop()
            
            # Second plugin instance - should load persisted jobs
            plugin2 = SchedulerPlugin(config)
            ctx2 = MockCtx()
            plugin2.start(ctx2)
            
            result = await plugin2.list_jobs()
            assert len(result["jobs"]) == 1
            assert result["jobs"][0]["name"] == "ReloadTest"
            
            plugin2.stop()


class TestSchedulerTools:
    """Test scheduler tools via tool registry interface."""

    @pytest.fixture
    async def scheduler_with_tools(self):
        """Create a scheduler plugin with tools registered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = EchoConfig(data_dir=Path(tmpdir))
            config.ensure_dirs()
            plugin = SchedulerPlugin(config)
            
            # Mock orchestrator
            class MockOrchestrator:
                async def handle_task(self, text, *, source="scheduler", session_id="scheduler", on_event=None, reply_channel=None, platform=None):
                    return {"response": f"Executed: {text}", "steps": [], "cached": False, "elapsed_ms": 10}
            
            class MockBus:
                def __init__(self):
                    self.subs = []
                def subscribe(self, pattern, handler, *, priority=0, once=False, label=""):
                    from python_core.core.event_bus import Subscription
                    from itertools import count
                    _serial = count(1)
                    sub = Subscription(
                        token=next(_serial),
                        pattern=pattern,
                        handler=handler,
                        priority=priority,
                        once=once,
                        label=label,
                    )
                    self.subs.append(sub)
                    return sub
                def unsubscribe(self, sub):
                    if sub in self.subs:
                        self.subs.remove(sub)
                        return True
                    return False
                def publish(self, topic, payload, *, source="test"):
                    # Fire and forget for tests
                    pass
            
            class MockTools:
                def __init__(self):
                    self.tools = {}
                def register(self, spec, handler):
                    self.tools[spec.name] = (spec, handler)
                    return spec
                def unregister(self, name):
                    return self.tools.pop(name, None) is not None
                async def execute(self, name, args):
                    spec, handler = self.tools.get(name, (None, None))
                    if spec is None:
                        return {"ok": False, "error": f"unknown tool: {name}", "tool": name}
                    if handler is None:
                        return {"ok": False, "error": f"tool {name} has no local handler", "tool": name}
                    try:
                        out = handler(args or {})
                        if asyncio.iscoroutine(out):
                            out = await out
                        return {"ok": True, "result": out, "tool": name}
                    except Exception as exc:
                        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "tool": name}
            
            class MockCtx:
                def __init__(self):
                    self.bus = MockBus()
                    self.tools = MockTools()
                    self._orchestrator = MockOrchestrator()
                def service(self, name):
                    if name == "orchestrator":
                        return self._orchestrator
                    return None
            
            ctx = MockCtx()
            plugin.start(ctx)
            
            yield plugin, ctx.tools
            
            plugin.stop()

    @pytest.mark.asyncio
    async def test_cron_add_tool(self, scheduler_with_tools):
        """Test cron_add tool."""
        plugin, tools = scheduler_with_tools
        result = await tools.execute("cron_add", {
            "name": "Tool Job",
            "schedule": "every 30m",
            "prompt": "Tool test",
            "skills": ["web"],
            "max_runs": 5,
        })
        assert result["ok"] is True
        assert result["result"]["ok"] is True
        assert result["result"]["job"]["name"] == "Tool Job"
        assert result["result"]["job"]["skills"] == ["web"]
        assert result["result"]["job"]["max_runs"] == 5

    @pytest.mark.asyncio
    async def test_cron_list_tool(self, scheduler_with_tools):
        """Test cron_list tool."""
        plugin, tools = scheduler_with_tools
        await tools.execute("cron_add", {"name": "Job1", "schedule": "every 1h", "prompt": "Task"})
        await tools.execute("cron_add", {"name": "Job2", "schedule": "daily at 9am", "prompt": "Task"})
        
        result = await tools.execute("cron_list", {})
        assert result["ok"] is True
        assert len(result["result"]["jobs"]) == 2

    @pytest.mark.asyncio
    async def test_cron_remove_tool(self, scheduler_with_tools):
        """Test cron_remove tool."""
        plugin, tools = scheduler_with_tools
        add_result = await tools.execute("cron_add", {"name": "To Remove", "schedule": "every 1h", "prompt": "Task"})
        job_id = add_result["result"]["job"]["id"]
        
        result = await tools.execute("cron_remove", {"job_id": job_id})
        assert result["ok"] is True
        assert result["result"]["ok"] is True
        assert result["result"]["removed"] == job_id

    @pytest.mark.asyncio
    async def test_cron_enable_disable_tool(self, scheduler_with_tools):
        """Test cron_enable and cron_disable tools."""
        plugin, tools = scheduler_with_tools
        add_result = await tools.execute("cron_add", {"name": "Toggle", "schedule": "every 1h", "prompt": "Task"})
        job_id = add_result["result"]["job"]["id"]
        
        # Disable
        result = await tools.execute("cron_disable", {"job_id": job_id})
        assert result["ok"] is True
        assert result["result"]["job"]["enabled"] is False
        
        # Enable
        result = await tools.execute("cron_enable", {"job_id": job_id})
        assert result["ok"] is True
        assert result["result"]["job"]["enabled"] is True

    @pytest.mark.asyncio
    async def test_cron_run_tool(self, scheduler_with_tools):
        """Test cron_run tool."""
        plugin, tools = scheduler_with_tools
        add_result = await tools.execute("cron_add", {"name": "Run Now", "schedule": "every 1h", "prompt": "Task"})
        job_id = add_result["result"]["job"]["id"]
        
        result = await tools.execute("cron_run", {"job_id": job_id})
        assert result["ok"] is True
        assert result["result"]["ok"] is True
        assert result["result"]["job"]["run_count"] == 1