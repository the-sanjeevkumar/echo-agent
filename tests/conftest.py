"""Shared fixtures: sys.path bootstrap + a fully-wired test kernel.

The kernel fixture assembles the REAL system (memory, self-healing, skills,
universal client, orchestrator) with only the LLM provider swapped for a
deterministic scripted fake — so tests exercise the actual event-bus wiring,
not mocks of it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from python_core.core.config import EchoConfig
from python_core.core.event_bus import EventBus
from python_core.core.interfaces import KernelContext
from python_core.core.plugin_manager import PluginManager
from python_core.core.tool_registry import ToolRegistry
from python_core.inter_agent.universal_client import UniversalClientPlugin
from python_core.orchestrator import Orchestrator
from python_core.plugins.memory.learning import REVIEW_SCHEMA
from python_core.plugins.memory.plugin import (
    EXTRACTION_SCHEMA,
    CognitiveMemoryPlugin,
)
from python_core.plugins.self_healing.plugin import SelfHealingPlugin
from python_core.plugins.server_ops import ServerOpsPlugin
from python_core.plugins.skills.ingester import SkillIngester
from python_core.plugins.supervisor import SupervisorPlugin


class FakeProvider:
    """Scripted LLM provider: pops one action dict per complete() call.

    Falls back to a terminal 'respond' action when the script is exhausted —
    tests never hang on an infinite tool loop.

    Extraction probes (``schema`` is the profile ``EXTRACTION_SCHEMA``) return
    an empty extraction without touching the script or the call log, so the
    per-turn profile extraction added in Phase 1 stays invisible to existing
    action-loop tests.
    """

    name = "fake_llm"
    version = "0.1.0"
    capabilities = ["llm", "structured-output"]

    def __init__(self, script: list[dict[str, Any]] | None = None) -> None:
        self.script = list(script or [])
        self.calls: list[list[dict[str, str]]] = []
        # Optional overrides for the structured probes (Phase 1/4). They never
        # consume the action script or pollute the action call log.
        self.extraction_return: dict[str, Any] | None = None
        self.review_return: dict[str, Any] | None = None
        self.review_calls: list[list[dict[str, str]]] = []

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        if schema is EXTRACTION_SCHEMA:
            return {
                "content": dict(self.extraction_return)
                if self.extraction_return is not None
                else {"name": None, "preferences": [], "facts": []}
            }
        if schema is REVIEW_SCHEMA:
            self.review_calls.append([dict(m) for m in messages])
            return {
                "content": dict(self.review_return)
                if self.review_return is not None
                else {"facts": [], "preferences": [], "lessons": [], "skill_candidates": []}
            }
        self.calls.append([dict(m) for m in messages])
        action = self.script.pop(0) if self.script else {"thought": "done", "action": "respond", "response": "ok"}
        return {"content": action}

    # plugin lifecycle (kernel registers it like a real provider)
    def start(self, ctx: KernelContext) -> None:
        ctx.services["llm"] = self

    def stop(self) -> None:
        pass


@dataclass
class Kernel:
    """Bundle of everything a test may need."""

    config: EchoConfig
    bus: EventBus
    ctx: KernelContext
    pm: PluginManager
    orchestrator: Orchestrator
    provider: FakeProvider
    captured: dict[str, list[Any]] = field(default_factory=dict)

    def record(self, pattern: str) -> None:
        """Start capturing payloads for a topic pattern into self.captured."""
        self.captured.setdefault(pattern, [])
        pattern_key = pattern
        async def _cap(event: Any) -> None:
            self.captured[pattern_key].append(getattr(event, "payload", event))
        self.bus.subscribe(pattern, _cap, label=f"test.capture:{pattern_key}")


@pytest.fixture
async def kernel(tmp_path: Path) -> Kernel:
    config = EchoConfig(
        data_dir=tmp_path / "data",
        enable_semantic_cache=False,  # tests opt in explicitly where relevant
        consolidation_idle_seconds=9999.0,  # background loop must not race tests
    )
    config.ensure_dirs()

    bus = EventBus()
    registry = ToolRegistry(bus)
    ctx = KernelContext(bus=bus, config=config, tools=registry, data_dir=config.data_dir)

    from python_core.main import _register_builtin_tools

    _register_builtin_tools(registry, config)

    pm = PluginManager(bus, config)
    ctx.services["plugin_manager"] = pm

    provider = FakeProvider()
    orchestrator = Orchestrator(bus=bus, config=config, tools=registry, provider=provider)

    pm.register(CognitiveMemoryPlugin(config))
    pm.register(provider)
    pm.register(SelfHealingPlugin(config))
    pm.register(ServerOpsPlugin(config))
    pm.register(SupervisorPlugin(config))
    pm.register(SkillIngester(config))
    pm.register(UniversalClientPlugin(config))
    pm.register(orchestrator)
    await pm.start_all(ctx)

    return Kernel(config=config, bus=bus, ctx=ctx, pm=pm, orchestrator=orchestrator, provider=provider)


@pytest.fixture
def bus() -> EventBus:
    return EventBus()
