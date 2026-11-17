"""Strict interface contracts (PEP 544 Protocols) for every plugin kind.

The microkernel (event bus + plugin manager) knows nothing about concrete
features. Any object satisfying one of these Protocols can be hot-plugged.
``runtime_checkable`` lets the plugin manager validate instances at load time,
so a malformed plugin fails fast with a precise error instead of corrupting
the running system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

__all__ = [
    "ToolSpec",
    "KernelContext",
    "Plugin",
    "MemoryPlugin",
    "ToolPlugin",
    "GatewayPlugin",
    "LLMProvider",
    "SelfHealingPlugin",
    "SkillPlugin",
    "ToolHandler",
]


# ── Tool schema (MODULE 3/5: shared by skills, manifest server, LLM calls) ──
@dataclass(slots=True)
class ToolSpec:
    """A JSON-Schema-described tool that the agent (or any external agent
    via the manifest protocol) can invoke."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    source: str = "builtin"          # builtin | skill:<repo> | plugin:<name>
    version: str = "0.1.0"
    tags: list[str] = field(default_factory=list)
    # Optional callable; when None the tool is advertised but must be executed
    # remotely (used by the universal client proxying external agents).
    handler: Callable[..., Any] | None = None

    def to_json_schema(self) -> dict[str, Any]:
        """Manifest/LLM-facing projection (handlers are never serialized)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "source": self.source,
            "version": self.version,
            "tags": self.tags,
        }


ToolHandler = Callable[[dict[str, Any]], Any]


# ── Kernel context handed to every plugin at start-up ────────────────────────
@dataclass(slots=True)
class KernelContext:
    """Everything a plugin may legitimately touch. Deliberately small:
    the event bus for communication, config for settings, the tool registry
    for capability exposure, and a free-form services dict for cross-plugin
    lookups (e.g. memory plugin registers itself under 'memory')."""

    bus: Any                                   # core.event_bus.EventBus
    config: Any                                # core.config.EchoConfig
    tools: Any                                 # core.tool_registry.ToolRegistry
    services: dict[str, Any] = field(default_factory=dict)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("echo.kernel"))
    data_dir: Path | None = None

    # Convenience accessors so plugins don't need casting gymnastics.
    def service(self, name: str, default: Any = None) -> Any:
        return self.services.get(name, default)


# ── Base plugin contract ─────────────────────────────────────────────────────
@runtime_checkable
class Plugin(Protocol):
    """Minimal lifecycle contract. All other plugin Protocols extend this."""

    name: str
    version: str
    capabilities: list[str]

    def start(self, ctx: KernelContext) -> None | Awaitable[None]:
        """Idempotent start-up (subscribe to bus topics, spawn tasks...)."""
        ...

    def stop(self) -> None | Awaitable[None]:
        """Release resources (cancel tasks, close sockets...)."""
        ...


@runtime_checkable
class MemoryPlugin(Plugin, Protocol):
    """Dual-process cognitive memory (MODULE 1)."""

    def write_working(self, content: str, *, role: str = "system", salience: float = 0.5) -> dict:
        """Push an item into volatile Working Memory."""
        ...
    def append_episode(self, action: str, outcome: str, *, metadata: dict | None = None) -> str:
        """Record an experience (action→outcome) into the episodic graph."""
        ...
    async def consolidate(self) -> dict:
        """Run one consolidation pass: WM → high-value EM/SM patterns."""
        ...
    def recall(self, query: str, *, k: int = 5) -> list[dict]:
        """Hybrid semantic recall from SM (vector) + EM (graph expansion)."""
        ...


@runtime_checkable
class ToolPlugin(Plugin, Protocol):
    """Exposes executable tools to the registry (skills, self-healing...)."""

    def tools(self) -> list[ToolSpec]: ...
    async def execute_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]: ...


@runtime_checkable
class GatewayPlugin(Plugin, Protocol):
    """Platform gateway (MODULE 6): translates external chat traffic into
    internal bus events and delivers agent replies back to the platform."""

    async def send_reply(self, channel_id: str, text: str) -> None: ...


@runtime_checkable
class LLMProvider(Protocol):
    """Any chat-completion backend (local ollama, vLLM, OpenAI-compatible...).

    ``schema`` forces structured JSON output (MODULE 4: zero filler) using the
    provider's native response-format mechanism when available."""

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Return at minimum {"content": str}; structured mode returns parsed JSON."""
        ...


@runtime_checkable
class SelfHealingPlugin(Plugin, Protocol):
    """Safe self-modification pipeline (MODULE 2)."""

    async def propose_patch(self, target_file: str, patch: str) -> dict[str, Any]:
        """Validate a unified diff in the sandbox; apply atomically or roll back."""
        ...


@runtime_checkable
class SkillPlugin(Plugin, Protocol):
    """Autonomous skill acquisition (MODULE 3)."""

    async def ingest(self, source: str) -> dict[str, Any]:
        """Clone/analyze a repo (or description) and hot-register its tools."""
        ...
