"""Dynamic plugin lifecycle manager with hot-reload.

Discovery contract — a plugin module/package must expose ONE of:

    def get_plugins() -> Sequence[object]      # preferred (explicit list)
    PLUGIN = <instance>                        # single instance
    def create_plugin() -> object              # factory

Any object is a plugin if it satisfies the :class:`~python_core.core.interfaces.Plugin`
Protocol: ``name``, ``version``, ``capabilities``, ``start(ctx)``, ``stop()``.
The manager validates shape at load time (``runtime_checkable`` Protocol) so a
broken plugin fails loudly at the boundary instead of mid-flight.

Hot-reload (used by the self-healing patch manager after code changes):
unload the module from ``sys.modules`` (including submodules), re-import,
re-instantiate, swap instances, and emit lifecycle events so the rest of the
system reacts without restarts.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

from .config import EchoConfig
from .event_bus import EventBus
from .interfaces import KernelContext, Plugin

logger = logging.getLogger("echo.plugins")


@dataclass(slots=True)
class PluginInstance:
    """A live plugin plus its provenance (needed for hot-reload)."""

    instance: Plugin
    module_name: str | None = None      # None → registered programmatically
    source_file: str | None = None


class PluginManager:
    """Owns the plugin registry: discovery, validation, lifecycle, hot-reload."""

    def __init__(self, bus: EventBus, config: EchoConfig) -> None:
        self.bus = bus
        self.config = config
        self._plugins: dict[str, PluginInstance] = {}

    # ── programmatic registration ──────────────────────────────────────────
    def register(self, plugin: Any, *, module_name: str | None = None) -> Plugin:
        """Attach a validated plugin instance. Raises TypeError on bad shape."""
        self._validate(plugin)
        name = plugin.name
        if name in self._plugins:
            raise ValueError(f"plugin name collision: {name!r}")
        self._plugins[name] = PluginInstance(instance=plugin, module_name=module_name)
        return plugin

    @staticmethod
    def _validate(plugin: Any) -> None:
        missing = [
            attr for attr in ("name", "version", "capabilities", "start", "stop")
            if not hasattr(plugin, attr)
        ]
        if missing:
            raise TypeError(f"{plugin!r} is not a valid plugin; missing: {missing}")
        if not isinstance(plugin.capabilities, (list, tuple, set)):
            raise TypeError(f"plugin {plugin.name}: capabilities must be a sequence")

    # ── filesystem discovery ───────────────────────────────────────────────
    def discover(self, directory: str | Path) -> list[str]:
        """Import every ``*.py`` module directly inside *directory* that
        satisfies the discovery contract. Returns module names loaded."""
        loaded: list[str] = []
        directory = Path(directory)
        if not directory.is_dir():
            logger.warning("plugin dir missing: %s", directory)
            return loaded
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
        importlib.invalidate_caches()  # see files dropped after process start
        for py in sorted(directory.glob("*.py")):
            if py.name.startswith("_"):
                continue
            module_name = py.stem
            try:
                module = importlib.import_module(module_name)
                self.load_from_module(module, module_name)
                loaded.append(module_name)
            except Exception:  # noqa: BLE001 — a broken plugin must not kill the kernel
                logger.exception("failed to load plugin module %s", py)
        return loaded

    def load_from_module(self, module: ModuleType, module_name: str | None = None) -> list[Plugin]:
        """Extract plugin instances from an imported module per the contract."""
        candidates: Sequence[Any] | None = None
        if hasattr(module, "get_plugins"):
            candidates = module.get_plugins()
        elif hasattr(module, "create_plugin"):
            candidates = [module.create_plugin()]
        elif hasattr(module, "PLUGIN"):
            candidates = [module.PLUGIN]
        if not candidates:
            logger.debug("module %s exposes no plugins", getattr(module, "__name__", module))
            return []
        plugins: list[Plugin] = []
        for cand in candidates:
            try:
                self.register(cand, module_name=module_name or getattr(module, "__name__", None))
                plugins.append(cand)
            except (TypeError, ValueError) as exc:
                logger.error("rejected candidate %r: %s", cand, exc)
        return plugins

    # ── lifecycle ──────────────────────────────────────────────────────────
    async def start_all(self, ctx: KernelContext) -> list[str]:
        """Start every registered plugin; a failing plugin is skipped, not fatal."""
        started: list[str] = []
        for name, inst in list(self._plugins.items()):
            try:
                out = inst.instance.start(ctx)
                if inspect.isawaitable(out):
                    await out
                started.append(name)
                self.bus.publish("plugin:started", {"plugin": name, "capabilities": list(inst.instance.capabilities)}, source="plugin_manager")
                logger.info("plugin started: %s v%s %s", name, inst.instance.version, list(inst.instance.capabilities))
            except Exception:  # noqa: BLE001
                logger.exception("plugin %s failed to start — skipping", name)
        return started

    async def stop_all(self) -> None:
        for name, inst in list(self._plugins.items()):
            try:
                out = inst.instance.stop()
                if inspect.isawaitable(out):
                    await out
                self.bus.publish("plugin:stopped", {"plugin": name}, source="plugin_manager")
            except Exception:  # noqa: BLE001
                logger.exception("plugin %s failed to stop cleanly", name)

    # ── hot-reload (MODULE 2 / MODULE 3 interplay) ─────────────────────────
    async def hot_reload(self, name: str, ctx: KernelContext) -> bool:
        """Unload + reimport + restart one plugin without touching the rest.

        Steps: stop old instance → purge its module(s) from sys.modules →
        re-import → rebuild via the discovery contract → validate → start.
        Returns True if a healthy replacement is now running.
        """
        inst = self._plugins.get(name)
        if inst is None:
            logger.warning("hot_reload: unknown plugin %s", name)
            return False
        module_name = inst.module_name

        # 0. Drop the old registration FIRST so the replacement can register
        #    under the same name (load_from_module enforces name uniqueness).
        del self._plugins[name]

        # 1. Graceful stop of the old instance.
        try:
            out = inst.instance.stop()
            if inspect.isawaitable(out):
                await out
        except Exception:  # noqa: BLE001
            logger.exception("stop failed during hot-reload of %s", name)

        # 2. Purge the module (and submodules) from sys.modules, DELETE the
        #    on-disk bytecode cache, and invalidate import finders.
        #    Why delete __pycache__: a rewritten file with identical size and
        #    the same mtime second revalidates against a stale .pyc — the
        #    classic "my hot-patch did nothing" bug that would silently break
        #    the self-healing loop.
        stale_files: list[Path] = []
        if module_name:
            old = sys.modules.get(module_name)
            if old is not None and getattr(old, "__file__", None):
                stale_files.append(Path(old.__file__))
        if module_name:
            for mod_name in [m for m in list(sys.modules) if m == module_name or m.startswith(module_name + ".")]:
                sys.modules.pop(mod_name, None)
        for src in stale_files:
            pyc_dir = src.parent / "__pycache__"
            if pyc_dir.is_dir():
                for pyc in pyc_dir.glob(f"{src.stem}.cpython-*.pyc"):
                    try:
                        pyc.unlink()
                    except OSError:
                        pass
        importlib.invalidate_caches()

        # 3. Restart path: programmatic plugins are re-imported by module name;
        #    discovered plugins are re-discovered from their source directory.
        try:
            if module_name:
                module = importlib.import_module(module_name)
                if not self.load_from_module(module, module_name):
                    raise RuntimeError(f"module {module_name} yielded no plugins after reload")
            elif inst.source_file:
                self.discover(Path(inst.source_file).parent)
            replacement = self._plugins.get(name)
            if replacement is None:
                raise RuntimeError(f"plugin {name} missing after reload")

            out = replacement.instance.start(ctx)
            if inspect.isawaitable(out):
                await out
            self.bus.publish("plugin:reloaded", {"plugin": name}, source="plugin_manager")
            logger.info("plugin hot-reloaded: %s", name)
            return True
        except Exception:  # noqa: BLE001
            logger.exception("hot-reload of %s failed; system keeps running on remaining plugins", name)
            return False

    # ── introspection ──────────────────────────────────────────────────────
    def names(self) -> list[str]:
        return sorted(self._plugins)

    def get(self, name: str) -> Plugin | None:
        inst = self._plugins.get(name)
        return inst.instance if inst else None

    def find_by_module(self, module_name: str) -> str | None:
        """Map a python module name to a registered plugin name (used by the
        self-healing plugin to hot-reload after a successful patch)."""
        for name, inst in self._plugins.items():
            if inst.module_name and (inst.module_name == module_name or module_name.startswith(inst.module_name + ".")):
                return name
        return None

    def manifest(self) -> list[dict[str, Any]]:
        return [
            {"name": i.instance.name, "version": i.instance.version, "capabilities": list(i.instance.capabilities), "module": i.module_name}
            for i in self._plugins.values()
        ]
