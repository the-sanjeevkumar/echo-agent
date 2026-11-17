"""Universal client — talk to ANY agent speaking the manifest protocol
(MODULE 5, client side).

Zero-friction interop in three steps, no shared SDK required:
    1. fetch_manifest(base_url)   → discover identity + tool schemas
    2. negotiate(manifest, caps)  → capability intersection
    3. execute(base_url, tool, args) → JSON-RPC POST /execute

The plugin also exposes local *proxy tools* (``agent_call``, ``agent_discover``)
and can mirror a remote agent's tools into the local registry with prefixed
names, so the orchestrator can call remote capabilities exactly like local ones.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from python_core.core.config import EchoConfig
from python_core.core.event_bus import EventBus
from python_core.core.interfaces import KernelContext, ToolSpec
from python_core.inter_agent.manifest_server import new_request_id

logger = logging.getLogger("echo.agent_client")

DEFAULT_TIMEOUT = 30.0


class AgentError(RuntimeError):
    """Raised (and also returned in envelopes) for inter-agent failures."""


class UniversalAgentClient:
    """Stateless HTTP/JSON-RPC client for the manifest protocol.

    ``transport`` optionally injects an httpx transport (used by the test
    suite to run the full protocol against an in-process ASGI app).
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT, transport: Any | None = None) -> None:
        self.timeout = timeout
        self._transport = transport
        self._manifest_cache: dict[str, dict[str, Any]] = {}

    def _make_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=self.timeout, transport=self._transport)

    # ── discovery ──────────────────────────────────────────────────────────
    async def fetch_manifest(self, base_url: str, *, force: bool = False) -> dict[str, Any]:
        url = base_url.rstrip("/") + "/manifest.json"
        if not force and base_url in self._manifest_cache:
            return self._manifest_cache[base_url]
        try:
            async with self._make_client() as client:
                resp = await client.get(url)
                resp.raise_for_status()
                manifest = resp.json()
        except httpx.HTTPError as exc:
            raise AgentError(f"manifest fetch failed for {url}: {exc}") from exc
        if not isinstance(manifest, dict) or "tools" not in manifest:
            raise AgentError(f"{url} did not return a valid manifest (missing 'tools')")
        self._manifest_cache[base_url] = manifest
        return manifest

    def negotiate(self, manifest: dict[str, Any], required_capabilities: list[str] | None = None) -> dict[str, Any]:
        """Capability negotiation: what can we actually do together?

        Returns {"compatible": bool, "shared_capabilities": [...],
                 "missing": [...], "tools": [...]}.
        """
        their_caps = set(manifest.get("capabilities", []))
        required = set(required_capabilities or [])
        missing = sorted(required - their_caps)
        return {
            "compatible": not missing,
            "shared_capabilities": sorted(their_caps & required) if required else sorted(their_caps),
            "missing": missing,
            "tools": [t.get("name") for t in manifest.get("tools", [])],
        }

    # ── execution ──────────────────────────────────────────────────────────
    async def execute(
        self,
        base_url: str,
        tool: str,
        args: dict[str, Any] | None = None,
        *,
        token: str | None = None,
    ) -> dict[str, Any]:
        """POST /execute with JSON-RPC 2.0; normalized result envelope."""
        url = base_url.rstrip("/") + "/execute"
        headers = {"Content-Type": "application/json"}
        bearer = token or _env_token()
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        body = {"jsonrpc": "2.0", "id": new_request_id(), "method": tool, "params": args or {}}
        try:
            async with self._make_client() as client:
                resp = await client.post(url, json=body, headers=headers)
        except httpx.HTTPError as exc:
            return {"ok": False, "error": f"transport error: {exc}", "agent": base_url, "tool": tool}
        try:
            data = resp.json()
        except ValueError:
            return {"ok": False, "error": f"non-JSON response (HTTP {resp.status_code})", "agent": base_url, "tool": tool}
        if "error" in data:
            err = data["error"] or {}
            return {"ok": False, "error": err.get("message", "rpc error"), "code": err.get("code"), "agent": base_url, "tool": tool}
        return {"ok": True, "result": data.get("result"), "agent": base_url, "tool": tool}

    # ── proxy tools (mirror a remote agent into the local registry) ────────
    async def register_remote_tools(
        self,
        bus: EventBus,
        base_url: str,
        *,
        alias: str | None = None,
        required_capabilities: list[str] | None = None,
    ) -> dict[str, Any]:
        """Fetch the remote manifest and hot-register local proxy tools for
        every remote tool. Proxy handlers forward over /execute. This is how
        echo-agent 'absorbs' another agent without installing anything."""
        manifest = await self.fetch_manifest(base_url)
        negotiation = self.negotiate(manifest, required_capabilities)
        alias = _sanitize(alias or manifest.get("name", "agent"))
        client = self
        registered: list[str] = []
        for tool in manifest.get("tools", []):
            remote_name = tool.get("name", "")
            if not remote_name:
                continue
            local_name = f"agent_{alias}_{_sanitize(remote_name)}"

            def _proxy(args: dict[str, Any], _url=base_url, _tool=remote_name) -> Any:
                return client.execute(_url, _tool, args)

            spec = ToolSpec(
                name=local_name,
                description=f"[agent:{alias}] {tool.get('description', '')}"[:500],
                parameters=tool.get("parameters") or {"type": "object", "properties": {}},
                source=f"agent:{base_url}",
                tags=["remote"],
            )
            # emit (awaited, not publish) so every listed proxy tool is
            # already callable when this method returns.
            await bus.emit("tool:register", {"spec": spec, "handler": _proxy}, source="universal_client")
            registered.append(local_name)
        return {"agent": manifest.get("name"), "base_url": base_url, "negotiation": negotiation, "proxy_tools": registered}


def _env_token() -> str:
    import os

    return os.environ.get("ECHO_AGENT_TOKEN", "")


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name).strip("_").lower()[:40] or "agent"


class UniversalClientPlugin:
    """Plugin surface: services['agent_client'] + two user-facing tools."""

    name = "universal_client"
    version = "0.1.0"
    capabilities = ["inter-agent-client", "http"]

    def __init__(self, config: EchoConfig | None = None) -> None:
        self.config = config or EchoConfig.load()
        self.client = UniversalAgentClient()
        self._ctx: KernelContext | None = None

    async def start(self, ctx: KernelContext) -> None:
        self._ctx = ctx
        ctx.services["agent_client"] = self.client
        await ctx.bus.emit("tool:register", {
            "spec": ToolSpec(
                name="agent_discover",
                description="Fetch and inspect another agent's /manifest.json (tools, capabilities, auth).",
                parameters={
                    "type": "object",
                    "properties": {"base_url": {"type": "string"}, "required_capabilities": {"type": "array", "items": {"type": "string"}}},
                    "required": ["base_url"],
                },
                source="plugin:universal_client",
                tags=["inter-agent"],
            ),
            "handler": self._tool_discover,
        })
        await ctx.bus.emit("tool:register", {
            "spec": ToolSpec(
                name="agent_call",
                description="Execute a tool on ANY manifest-speaking agent via JSON-RPC POST /execute.",
                parameters={
                    "type": "object",
                    "properties": {
                        "base_url": {"type": "string"},
                        "tool": {"type": "string"},
                        "args": {"type": "object"},
                        "token": {"type": "string"},
                    },
                    "required": ["base_url", "tool"],
                },
                source="plugin:universal_client",
                tags=["inter-agent"],
            ),
            "handler": self._tool_call,
        })
        await ctx.bus.emit("tool:register", {
            "spec": ToolSpec(
                name="agent_absorb",
                description="Mirror every tool of a remote agent into the local registry as proxy tools (no restart).",
                parameters={
                    "type": "object",
                    "properties": {"base_url": {"type": "string"}, "alias": {"type": "string"}},
                    "required": ["base_url"],
                },
                source="plugin:universal_client",
                tags=["inter-agent"],
            ),
            "handler": self._tool_absorb,
        })
        ctx.logger.info("universal client plugin online")

    def stop(self) -> None:
        self._ctx = None

    def _tool_discover(self, args: dict[str, Any]) -> Any:
        return self.client.fetch_manifest(str(args["base_url"]))

    def _tool_call(self, args: dict[str, Any]) -> Any:
        return self.client.execute(
            str(args["base_url"]),
            str(args["tool"]),
            args.get("args") or {},
            token=args.get("token"),
        )

    def _tool_absorb(self, args: dict[str, Any]) -> Any:
        if self._ctx is None:
            return {"ok": False, "error": "plugin not started"}
        return self.client.register_remote_tools(
            self._ctx.bus,
            str(args["base_url"]),
            alias=args.get("alias"),
        )


def create_plugin() -> UniversalClientPlugin:
    return UniversalClientPlugin()
