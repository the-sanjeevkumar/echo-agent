"""Plugin manager: validation, discovery contract, hot-reload, gateways."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from python_core.core.event_bus import EventBus
from python_core.core.interfaces import KernelContext
from python_core.core.plugin_manager import PluginManager
from python_core.plugins.gateways.base import GatewayInterface


class MiniPlugin:
    name = "mini"
    version = "1.0"
    capabilities = ["mini"]

    def __init__(self) -> None:
        self.started = False

    def start(self, ctx) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False


def test_validate_rejects_bad_plugins(bus: EventBus) -> None:
    pm = PluginManager(bus, config=None)
    with pytest.raises(TypeError):
        pm.register(object())  # no name/version/start/stop


def test_register_collision(bus: EventBus) -> None:
    pm = PluginManager(bus, config=None)
    pm.register(MiniPlugin())
    with pytest.raises(ValueError):
        pm.register(MiniPlugin())


async def test_lifecycle_start_stop(bus: EventBus, tmp_path: Path) -> None:
    from python_core.core.config import EchoConfig
    from python_core.core.tool_registry import ToolRegistry

    config = EchoConfig(data_dir=tmp_path / "cfg")
    config.ensure_dirs()
    ctx = KernelContext(bus=bus, config=config, tools=ToolRegistry(bus), data_dir=config.data_dir)
    pm = PluginManager(bus, config)
    plugin = MiniPlugin()

    lifecycle: list = []

    async def _cap(e) -> None:
        lifecycle.append((getattr(e, "topic", "?"), getattr(e, "payload", e)))

    bus.subscribe("plugin:*", _cap, label="cap")
    pm.register(plugin)
    started = await pm.start_all(ctx)
    assert started == ["mini"] and plugin.started is True
    await asyncio.sleep(0.05)  # let the fire-and-forget lifecycle events run
    assert ("plugin:started", {"plugin": "mini", "capabilities": ["mini"]}) in lifecycle

    await pm.stop_all()
    assert plugin.started is False
    await asyncio.sleep(0.05)
    assert any(t == "plugin:stopped" for t, _ in lifecycle)


async def test_discovery_contract_and_hot_reload(bus: EventBus, tmp_path: Path) -> None:
    """Write a plugin module on disk, discover it, mutate it, hot-reload."""
    plugin_dir = tmp_path / "external_plugins"
    plugin_dir.mkdir()
    module = plugin_dir / "reloadable_plugin.py"
    module.write_text(
        "VERSION = '1.0'\n"
        "class Reloadable:\n"
        "    name = 'reloadable'\n"
        "    version = VERSION\n"
        "    capabilities = ['demo']\n"
        "    def start(self, ctx):\n"
        "        ctx.services['reloadable'] = self\n"
        "    def stop(self):\n"
        "        pass\n"
        "def get_plugins():\n"
        "    return [Reloadable()]\n"
    )

    from python_core.core.config import EchoConfig
    from python_core.core.tool_registry import ToolRegistry

    config = EchoConfig(data_dir=tmp_path / "cfg2")
    config.ensure_dirs()
    ctx = KernelContext(bus=bus, config=config, tools=ToolRegistry(bus), data_dir=config.data_dir)
    pm = PluginManager(bus, config)

    assert pm.discover(plugin_dir) == ["reloadable_plugin"]
    assert pm.get("reloadable") is not None and pm.get("reloadable").version == "1.0"

    # Mutate the module on disk (as the self-healing patcher would), then reload.
    module.write_text(module.read_text().replace("VERSION = '1.0'", "VERSION = '2.0'"))
    ok = await pm.hot_reload("reloadable", ctx)
    assert ok is True
    assert pm.get("reloadable").version == "2.0"


async def test_hot_reload_unknown_plugin(bus: EventBus) -> None:
    pm = PluginManager(bus, config=None)
    assert await pm.hot_reload("ghost", ctx=None) is False


# ── Gateway interface contract ───────────────────────────────────────────────
class DummyGateway(GatewayInterface):
    name = "dummy"
    capabilities = ["gateway", "dummy"]

    def __init__(self) -> None:
        super().__init__()
        self.replies: list[tuple[str, str]] = []

    async def start(self, ctx) -> None:
        self.bind_bus(ctx.bus)

    async def stop(self) -> None:
        self.unbind_bus()

    async def send_reply(self, channel_id: str, text: str) -> None:
        self.replies.append((channel_id, text))


async def test_gateway_reply_routing(kernel) -> None:
    """Gateway receives ONLY replies addressed to its own channels."""
    gw = DummyGateway()
    gw.bind_bus(kernel.bus)

    # addressed to dummy → delivered
    kernel.bus.publish("task:completed", {
        "response": "here you go",
        "reply_channel": "dummy:chan-42",
    })
    await asyncio.sleep(0.1)
    assert gw.replies == [("chan-42", "here you go")]

    # addressed to another gateway → ignored
    gw.replies.clear()
    kernel.bus.publish("task:completed", {"response": "x", "reply_channel": "telegram:123"})
    await asyncio.sleep(0.1)
    assert gw.replies == []


async def test_gateway_inbound_event_shape(kernel) -> None:
    """emit_task produces exactly the contract payload the orchestrator consumes."""
    kernel.record("task:incoming")
    gw = DummyGateway()
    gw.emit_task(kernel.bus, "do the thing", "chan-7", source_user="alice")
    await asyncio.sleep(0.1)
    payload = kernel.captured["task:incoming"][0]
    assert payload["text"] == "do the thing"
    assert payload["source"] == "dummy"
    assert payload["reply_channel"] == "dummy:chan-7"
    assert payload["user"] == "alice"


async def test_end_to_end_gateway_to_agent(kernel) -> None:
    """Full loop: gateway inbound → orchestrator → reply routed back."""
    kernel.provider.script = [{"thought": "t", "action": "respond", "response": "gateway says hi back"}]
    gw = DummyGateway()
    gw.bind_bus(kernel.bus)
    gw.emit_task(kernel.bus, "hello from platform", "chan-99")
    await asyncio.sleep(0.3)  # allow the full async chain
    assert gw.replies == [("chan-99", "gateway says hi back")]


def test_gateway_is_abstract() -> None:
    with pytest.raises(TypeError):
        GatewayInterface()  # type: ignore[abstract]


async def test_builtin_tools_via_kernel(kernel) -> None:
    """fs tools present with real file behavior (delta-first policy)."""
    r = await kernel.ctx.tools.execute("fs_read", {"path": "pyproject.toml", "max_chars": 2000})
    assert r["ok"] and "[project]" in r["result"]["content"]

    r2 = await kernel.ctx.tools.execute("fs_list", {"dir": ".", "pattern": "*.toml"})
    assert r2["ok"] and any(e["path"].endswith("pyproject.toml") for e in r2["result"]["entries"])

    # shell disabled by default (safety); the tool "ran" but returned an error
    r3 = await kernel.ctx.tools.execute("shell_exec", {"command": ["echo", "hi"]})
    assert r3["result"]["ok"] is False and "ECHO_ALLOW_SHELL" in r3["result"]["error"]
