"""Universal agent protocol: manifest server + universal client, end-to-end.

Runs the full HTTP round trip against the REAL FastAPI app using httpx's
ASGI transport — server-side manifest generation, JSON-RPC execution,
capability negotiation, and remote tool absorption.
"""

from __future__ import annotations

import httpx
import pytest

from python_core.inter_agent.manifest_server import build_manifest, create_app
from python_core.inter_agent.universal_client import UniversalAgentClient


@pytest.fixture
async def asgi(kernel):
    """httpx AsyncClient bound to the kernel's live manifest server app."""
    app = create_app(kernel.ctx)
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://echo.test")
    yield client
    await client.aclose()


async def test_manifest_exposes_live_tools(asgi) -> None:
    resp = await asgi.get("/manifest.json")
    assert resp.status_code == 200
    manifest = resp.json()
    assert manifest["protocol"] == "echo-agent-manifest/1"
    assert manifest["name"] == "echo-agent"
    tool_names = [t["name"] for t in manifest["tools"]]
    assert "fs_read" in tool_names and "memory_recall" in tool_names
    assert manifest["endpoints"]["execute"] == "/execute"
    assert manifest["auth"]["type"] == "none"


async def test_manifest_reflects_runtime_tool_registration(asgi, kernel) -> None:
    """A tool registered mid-flight appears in the manifest instantly."""
    await kernel.bus.emit("tool:register", {
        "spec": {"name": "fresh_tool", "description": "just arrived", "parameters": {"type": "object", "properties": {}}, "source": "test"},
        "handler": lambda a: {"fresh": True},
    })
    manifest = (await asgi.get("/manifest.json")).json()
    assert any(t["name"] == "fresh_tool" for t in manifest["tools"])


async def test_execute_jsonrpc_success(asgi) -> None:
    payload = {
        "jsonrpc": "2.0", "id": 7, "method": "fs_read",
        "params": {"path": "pyproject.toml", "max_chars": 40},
    }
    resp = await asgi.post("/execute", json=payload)
    body = resp.json()
    assert body["jsonrpc"] == "2.0" and body["id"] == 7
    result = body["result"]  # /execute unwraps the tool envelope into result
    assert len(result["content"]) <= 40 and result["truncated"] is True


async def test_execute_jsonrpc_unknown_tool(asgi) -> None:
    resp = await asgi.post("/execute", json={"jsonrpc": "2.0", "id": 1, "method": "no_such_tool", "params": {}})
    body = resp.json()
    assert body["error"]["code"] == -32601


async def test_execute_invalid_request(asgi) -> None:
    resp = await asgi.post("/execute", json={"id": 1})  # no jsonrpc, no method
    body = resp.json()
    assert body["error"]["code"] == -32600


async def test_health(asgi) -> None:
    body = (await asgi.get("/health")).json()
    assert body["status"] == "ok"


# ── universal client against the SAME app (true interop loop) ────────────────
@pytest.fixture
async def agent_pair(kernel, asgi):
    app = create_app(kernel.ctx)
    client = UniversalAgentClient(transport=httpx.ASGITransport(app=app))
    return kernel, client


async def test_fetch_manifest_and_negotiate(agent_pair) -> None:
    kernel, client = agent_pair
    manifest = await client.fetch_manifest("http://echo.test")
    assert manifest["name"] == "echo-agent"
    negotiation = client.negotiate(manifest, required_capabilities=["llm", "skill-acquisition", "teleportation"])
    assert "llm" in negotiation["shared_capabilities"]
    assert "skill-acquisition" in negotiation["shared_capabilities"]
    assert negotiation["missing"] == ["teleportation"]
    assert negotiation["compatible"] is False


async def test_execute_via_universal_client(agent_pair) -> None:
    kernel, client = agent_pair
    out = await client.execute("http://echo.test", "memory_stats", {})
    assert out["ok"] is True and out["result"]["wm_items"] >= 0


async def test_absorb_remote_tools(agent_pair) -> None:
    """MODULE 5 payoff: absorb another agent's tools as local proxies."""
    kernel, client = agent_pair
    report = await client.register_remote_tools(kernel.bus, "http://echo.test", alias="echo2")
    assert report["agent"] == "echo-agent"
    assert any("fs_read" in name for name in report["proxy_tools"])

    # the proxy tool actually round-trips to the "remote" agent
    proxy_name = next(n for n in report["proxy_tools"] if n.endswith("fs_read"))
    r = await kernel.ctx.tools.execute(proxy_name, {"path": "README.md", "max_chars": 10})
    assert r["ok"] is True


async def test_agent_call_tool(agent_pair) -> None:
    kernel, client = agent_pair
    # Route the plugin's internal client through the ASGI transport too.
    plugin = kernel.pm.get("universal_client")
    from python_core.inter_agent.universal_client import UniversalAgentClient

    plugin.client = UniversalAgentClient(transport=httpx.ASGITransport(app=create_app(kernel.ctx)))
    r = await kernel.ctx.tools.execute(
        "agent_call",
        {"base_url": "http://echo.test", "tool": "memory_stats", "args": {}},
    )
    assert r["ok"] is True
    # two envelopes: registry wraps the client's JSON-RPC envelope again
    assert r["result"]["ok"] is True
    assert "episodes" in r["result"]["result"]["episodic"]


async def test_agent_error_envelope(agent_pair) -> None:
    kernel, client = agent_pair
    out = await client.execute("http://echo.test", "definitely_missing", {})
    assert out["ok"] is False and out["code"] == -32601


def test_build_manifest_structure(kernel) -> None:
    manifest = build_manifest(kernel.ctx)
    assert set(manifest) >= {"protocol", "name", "version", "endpoints", "auth", "capabilities", "tools"}
    assert any("memory" in c for c in manifest["capabilities"])  # from memory plugin
