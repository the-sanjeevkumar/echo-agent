"""Manifest server — the agent's public face (MODULE 5, server side).

Exposes the Universal HTTP/JSON-RPC Agent Manifest Protocol:

    GET  /manifest.json   → identity, capabilities, tool schemas, auth hints
    POST /execute         → JSON-RPC 2.0 single-tool execution
    GET  /health          → liveness

Any future agent, in any language, can interoperate with echo-agent by
fetching this manifest and POSTing standard JSON-RPC — no shared SDK, no
mutual installation, no codegen step.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import Any

from python_core import __version__
from python_core.core.interfaces import KernelContext

logger = logging.getLogger("echo.manifest")

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
except Exception:  # noqa: BLE001 — core deps, but keep the module importable in bare envs
    FastAPI = None  # type: ignore[assignment]


def build_manifest(ctx: KernelContext) -> dict[str, Any]:
    """The manifest document. Tool schemas come LIVE from the registry, so a
    skill ingested moments ago is immediately visible to external agents."""
    pm = ctx.service("plugin_manager")
    capabilities = sorted({c for p in (pm.manifest() if pm else []) for c in p.get("capabilities", [])})
    auth_type = "bearer" if os.environ.get("ECHO_AGENT_TOKEN") else "none"
    return {
        "protocol": "echo-agent-manifest/1",
        "name": "echo-agent",
        "version": __version__,
        "description": "Local-first, self-improving, event-driven polyglot AI agent.",
        "endpoints": {
            "manifest": "/manifest.json",
            "execute": "/execute",
            "health": "/health",
        },
        "auth": {"type": auth_type, "header": "Authorization", "scheme": "Bearer"},
        "capabilities": capabilities,
        "tools": ctx.tools.schemas(),
        "features": {
            "memory": "dual-process (working/episodic/semantic) with idle consolidation",
            "self_modification": "sandboxed patch validation with atomic rollback",
            "token_efficiency": "structured outputs, semantic caching, delta diffs",
        },
    }


def create_app(ctx: KernelContext) -> Any:
    """FastAPI application factory. ``ctx`` closes over the live kernel."""
    if FastAPI is None:
        raise RuntimeError("fastapi is required for the manifest server")

    app = FastAPI(title="echo-agent", version=__version__, docs_url=None, redoc_url=None)

    def _authorized(request: Request) -> None:
        expected = os.environ.get("ECHO_AGENT_TOKEN")
        if expected:
            provided = request.headers.get("Authorization", "")
            if provided != f"Bearer {expected}":
                raise HTTPException(status_code=401, detail="unauthorized")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "ts": time.time()}

    @app.get("/manifest.json")
    async def manifest(request: Request) -> dict[str, Any]:
        _authorized(request)
        return build_manifest(ctx)

    @app.post("/execute")
    async def execute(request: Request) -> JSONResponse:
        """JSON-RPC 2.0: method = tool name, params = tool args object."""
        _authorized(request)
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            return _rpc_error(None, -32700, "invalid JSON body")

        rpc_id = payload.get("id")
        if payload.get("jsonrpc") != "2.0" or not isinstance(payload.get("method"), str):
            return _rpc_error(rpc_id, -32600, "invalid JSON-RPC request")

        method = payload["method"]
        params = payload.get("params") or {}
        if method == "rpc.manifest":
            return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": build_manifest(ctx)})
        if not isinstance(params, dict):
            return _rpc_error(rpc_id, -32602, "params must be an object")

        result = await ctx.tools.execute(method, params)
        if result.get("ok"):
            return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": result.get("result")})
        code = -32601 if "unknown tool" in str(result.get("error", "")) else -32000
        return _rpc_error(rpc_id, code, str(result.get("error", "execution failed")))

    return app


def _rpc_error(rpc_id: Any, code: int, message: str) -> JSONResponse:
    return JSONResponse(
        {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}},
        status_code=200,  # JSON-RPC errors ride on HTTP 200
    )


class ManifestServer:
    """Plugin-style lifecycle wrapper around uvicorn (background task)."""

    name = "manifest_server"
    version = "0.1.0"
    capabilities = ["inter-agent-server", "http"]

    def __init__(self, ctx: KernelContext) -> None:
        self.ctx = ctx
        self._server: Any = None
        self._task: Any = None

    async def start(self) -> None:
        import uvicorn

        app = create_app(self.ctx)
        config = uvicorn.Config(
            app,
            host=self.ctx.config.manifest_host,
            port=self.ctx.config.manifest_port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._serve())
        self.ctx.logger.info(
            "manifest server on http://%s:%d/manifest.json",
            self.ctx.config.manifest_host, self.ctx.config.manifest_port,
        )

    async def _serve(self) -> None:
        try:
            await self._server.serve()
        except (Exception, SystemExit) as exc:  # noqa: BLE001
            logger.warning("manifest server crashed: %s", exc)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (Exception, SystemExit, asyncio.CancelledError):  # noqa: BLE001
                pass
            self._task = None


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]
