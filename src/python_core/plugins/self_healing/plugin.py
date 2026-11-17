"""Self-healing plugin glue (MODULE 2).

Exposes the safe self-modification pipeline as event-bus tools so the
orchestrator (or any gateway, or another agent via the manifest protocol)
can request code changes without importing anything:

    self_patch(target_file, patch, test_path?) → validate in sandbox, apply or reject
    self_diff(original, modified)              → unified diff
    self_rollback(target_file)                 → restore the last backup

Learning loop: every rejection is written to Episodic Memory with the test
evidence; every acceptance records the diff. The consolidator later mines
these episodes into cause-effect rules ("diffs touching X tend to break Y").
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from python_core.core.config import EchoConfig
from python_core.core.event_bus import EventBus
from python_core.core.interfaces import KernelContext, ToolSpec

from .patch_manager import PatchManager, create_diff
from .sandbox import Sandbox

logger = logging.getLogger("echo.selfheal")


def module_name_for(project_root: Path, file_path: Path) -> str | None:
    """Derive the importable module name of a file inside *project_root*.

    src/python_core/plugins/x.py → python_core.plugins.x
    src/python_core/plugins/__init__.py → python_core.plugins
    """
    try:
        rel = file_path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return None
    parts = list(rel.with_suffix("").parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return None
    return ".".join(parts)


class SelfHealingPlugin:
    name = "self_healing"
    version = "0.1.0"
    capabilities = ["self-modification", "sandbox-validation", "atomic-rollback"]

    def __init__(self, config: EchoConfig | None = None) -> None:
        self.config = config or EchoConfig.load()
        self.pm = PatchManager(Sandbox())
        self._ctx: KernelContext | None = None
        self._bus: EventBus | None = None
        self._backups: dict[str, str] = {}  # target_file → original content (session-scoped)

    # ── lifecycle ──────────────────────────────────────────────────────────
    async def start(self, ctx: KernelContext) -> None:
        self._ctx = ctx
        self._bus = ctx.bus
        ctx.config.ensure_dirs()
        ctx.services["self_healing"] = self

        tools = [
            (
                ToolSpec(
                    name="self_patch",
                    description=(
                        "Safely modify this agent's own code. Provide a unified diff for target_file; "
                        "it is validated by the full pytest suite in an isolated sandbox and applied "
                        "atomically only if all tests pass. Rejected patches are recorded to memory."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "target_file": {"type": "string", "description": "path relative to project root, e.g. src/python_core/foo.py"},
                            "patch": {"type": "string", "description": "unified diff (diff --git or plain ---/+++ hunks)"},
                            "test_path": {"type": "string", "description": "optional pytest path to scope validation"},
                        },
                        "required": ["target_file", "patch"],
                    },
                    source="plugin:self_healing",
                    tags=["self-modification"],
                ),
                self._tool_self_patch,
            ),
            (
                ToolSpec(
                    name="self_diff",
                    description="Generate a unified diff from original and modified file contents.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "original": {"type": "string"},
                            "modified": {"type": "string"},
                        },
                        "required": ["original", "modified"],
                    },
                    source="plugin:self_healing",
                ),
                self._tool_self_diff,
            ),
            (
                ToolSpec(
                    name="self_rollback",
                    description="Restore a previously patched file to its last known-good content (atomic).",
                    parameters={
                        "type": "object",
                        "properties": {"target_file": {"type": "string"}},
                        "required": ["target_file"],
                    },
                    source="plugin:self_healing",
                    tags=["self-modification", "rollback"],
                ),
                self._tool_self_rollback,
            ),
        ]
        for spec, handler in tools:
            await ctx.bus.emit("tool:register", {"spec": spec, "handler": handler})

        ctx.logger.info("self-healing plugin online")

    def stop(self) -> None:
        self._ctx = None

    # ── pipeline ───────────────────────────────────────────────────────────
    async def propose_patch(
        self,
        target_file: str,
        patch: str,
        *,
        test_path: str | None = None,
        project_root: Path | None = None,
    ) -> dict[str, Any]:
        root = project_root or self.config.resolved_project_root
        outcome = await self.pm.propose_patch(
            Path(root), target_file, patch, test_path=test_path, timeout=self.config.sandbox_test_timeout
        )
        self._record(target_file, outcome)
        return {
            "applied": outcome.applied,
            "target_file": outcome.target_file,
            "reason": outcome.reason,
            "diff": outcome.diff[:4000],
            "evidence": (outcome.meta.get("evidence", "") if outcome.meta else "")[:1500],
        }

    def _record(self, target_file: str, outcome: Any) -> None:
        """Episodic learning + bus notification. Never raises."""
        if self._bus is None:
            return
        memory = self._ctx.service("memory") if self._ctx else None
        success = bool(getattr(outcome, "applied", False))
        try:
            if success and getattr(outcome, "original_content", ""):
                self._backups[target_file] = outcome.original_content
                self._persist_backup(target_file, outcome.original_content)
            if memory is not None:
                memory.append_episode(
                    action=f"self_patch({target_file})",
                    outcome=(outcome.reason or ("applied" if success else "rejected"))[:300],
                    success=success,
                    metadata={
                        "diff": (getattr(outcome, "diff", "") or "")[:2000],
                        "evidence": (getattr(outcome, "meta", {}).get("evidence", "") if getattr(outcome, "meta", {}) else "")[:2000],
                    },
                )
        except Exception:  # noqa: BLE001
            logger.exception("failed to record patch outcome")
        self._bus.publish(
            "selfheal:applied" if success else "selfheal:rejected",
            {"target_file": target_file, "reason": getattr(outcome, "reason", "")},
            source="self_healing",
        )
        if success:
            self._hot_reload_if_plugin(target_file)

    def _hot_reload_if_plugin(self, target_file: str) -> None:
        """If the patched file belongs to a registered plugin, hot-reload it."""
        if self._ctx is None:
            return
        pm = self._ctx.service("plugin_manager")
        if pm is None:
            return
        module_name = module_name_for(Path(self.config.resolved_project_root), Path(target_file))
        if not module_name:
            return
        plugin_name = pm.find_by_module(module_name)
        if plugin_name:
            self._bus.publish("selfheal:hotreload", {"plugin": plugin_name, "module": module_name}, source="self_healing")
            # Fire-and-forget: hot_reload is async; run it safely.
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(pm.hot_reload(plugin_name, self._ctx))
            except RuntimeError:
                logger.debug("no running loop; plugin %s reloads on next start", plugin_name)

    # ── backups / rollback ─────────────────────────────────────────────────
    def _persist_backup(self, target_file: str, content: str) -> None:
        backup_dir = self.config.data_dir / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        safe = target_file.replace("/", "__").replace("\\", "__")
        (backup_dir / f"{int(time.time())}__{safe}").write_text(content, encoding="utf-8")

    def _tool_self_patch(self, args: dict[str, Any]) -> dict[str, Any]:
        # Tool handlers may be async; the registry awaits coroutines. We need
        # an async invocation, so return the coroutine for the registry.
        return self.propose_patch(
            target_file=str(args["target_file"]),
            patch=str(args["patch"]),
            test_path=args.get("test_path"),
        )

    def _tool_self_diff(self, args: dict[str, Any]) -> dict[str, Any]:
        return {"diff": create_diff(str(args["original"]), str(args["modified"]))}

    def _tool_self_rollback(self, args: dict[str, Any]) -> dict[str, Any]:
        from .patch_manager import atomic_write

        target_file = str(args["target_file"])
        content = self._backups.get(target_file)
        if content is None:
            return {"ok": False, "error": f"no session backup for {target_file}"}
        path = Path(self.config.resolved_project_root) / target_file
        atomic_write(path, content)
        self._backups.pop(target_file, None)
        if self._bus is not None:
            self._bus.publish("selfheal:rolledback", {"target_file": target_file}, source="self_healing")
        return {"ok": True, "target_file": target_file}


def create_plugin() -> SelfHealingPlugin:
    """Discovery-contract entry point."""
    return SelfHealingPlugin()
