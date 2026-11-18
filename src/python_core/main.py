"""echo-agent CLI entry point & daemon manager.

Entry points (installed via pyproject [project.scripts]):
    echo-agent run "task text"     one-shot task through the full agent loop
    echo-agent start [--headless]  daemon: plugins + manifest server + REPL
    echo-agent skills add <url>    ingest a skill repository immediately
    echo-agent tools list          show the live tool registry
    echo-agent plugins list        show loaded plugins and capabilities
    echo-agent pairing list        show pending/approved DM senders
    echo-agent pairing approve ... approve a DM pairing code
    echo-agent bus                 launch the Rust echo-bus IPC server
    echo-agent version             print version

The kernel bootstrap (:func:`build_kernel`) is deliberately importable so
tests can construct the full system without the CLI.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import httpx
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path
from typing import Any

from python_core import __version__
from python_core.core.config import EchoConfig
from python_core.core.event_bus import EventBus
from python_core.core.interfaces import KernelContext, ToolSpec
from python_core.core.plugin_manager import PluginManager
from python_core.core.tool_registry import ToolRegistry
from python_core.lifecycle import is_setup_complete
from python_core.inter_agent.manifest_server import ManifestServer
from python_core.orchestrator import Orchestrator
from python_core.plugins.gateways.discord import DiscordGateway
from python_core.plugins.gateways.pairing import PairingStore
from python_core.plugins.gateways.telegram import TelegramGateway
from python_core.plugins.llm.openai_compat import OpenAICompatProvider
from python_core.plugins.memory.plugin import CognitiveMemoryPlugin
from python_core.plugins.self_healing.plugin import SelfHealingPlugin
from python_core.plugins.skills.ingester import SkillIngester
from python_core.plugins.web.plugin import WebPlugin
from python_core.setup_wizard import SetupOptions, run_setup
from python_core.ui import (
    Spinner,
    banner,
    color_enabled,
    paint,
    render_response,
    render_steps,
)

logger = logging.getLogger("echo.cli")

# Third-party loggers that are chatty at INFO and add nothing for CLI users.
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "uvicorn",
    "uvicorn.access",
    "uvicorn.error",
    "asyncio",
    "watchfiles",
    "scrapling",
)


def _configure_logging(verbose: bool) -> None:
    """Warnings/errors only by default — no INFO request spam in the REPL.

    ``-v`` restores timestamped DEBUG output for troubleshooting. The ``echo.*``
    namespace follows the same switch, so routine startup/consolidation notices
    stay quiet unless explicitly requested.
    """
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format=(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
            if verbose
            else "%(levelname)s %(name)s: %(message)s"
        ),
        force=True,
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.INFO if verbose else logging.WARNING)
    logging.getLogger("echo").setLevel(level)



# ─────────────────────────────────────────────────────────────────────────────
# Kernel bootstrap
# ─────────────────────────────────────────────────────────────────────────────
def _allowed_roots(config: EchoConfig) -> list[Path]:
    """Roots the filesystem tools may touch. Configured via fs_allowed_roots;
    empty ⇒ safe defaults (project root, data dir, cwd, system temp)."""
    if config.fs_allowed_roots:
        return [Path(p).expanduser().resolve() for p in config.fs_allowed_roots]
    return [
        config.resolved_project_root.resolve(),
        config.data_dir.resolve(),
        Path.cwd().resolve(),
        Path(tempfile.gettempdir()).resolve(),
    ]


def _path_allowed(path: Path, roots: list[Path]) -> bool:
    """True iff *path* resolves to a location under one of *roots*."""
    try:
        resolved = path.expanduser().resolve()
    except (OSError, RuntimeError):
        return False
    for root in roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False

def _register_builtin_tools(registry: ToolRegistry, config: EchoConfig) -> None:
    """Core filesystem/shell tools. File mutation is DELTA-FIRST (MODULE 4):
    agents send unified diffs, never whole-file rewrites, via fs_apply_diff."""
    from python_core.plugins.self_healing.patch_manager import apply_unified_diff, atomic_write

    roots = _allowed_roots(config)

    def _blocked(path: Path) -> dict[str, Any] | None:
        if _path_allowed(path, roots):
            return None
        return {"ok": False, "error": f"path outside allowed roots ({path}); configure fs_allowed_roots to extend"}

    def fs_read(args: dict[str, Any]) -> dict[str, Any]:
        path = Path(args["path"]).expanduser()
        denied = _blocked(path)
        if denied is not None:
            return denied
        if not path.exists():
            parent = path.parent if path.parent.is_dir() else Path(".")
            nearby = sorted(
                f"{p.name}{'/' if p.is_dir() else ''}" for p in parent.iterdir()
            )[:12]
            return {"ok": False, "error": f"file not found: {path}", "nearby": nearby}
        if not path.is_file():
            return {"ok": False, "error": f"not a regular file (use fs_list): {path}"}
        text = path.read_text(encoding="utf-8", errors="replace")
        max_chars = int(args.get("max_chars", 8000))
        return {"path": str(path), "content": text[:max_chars], "truncated": len(text) > max_chars}

    def fs_list(args: dict[str, Any]) -> dict[str, Any]:
        base = Path(args.get("dir", ".")).expanduser()
        denied = _blocked(base)
        if denied is not None:
            return denied
        if not base.is_dir():
            return {"entries": [], "error": f"directory not found: {base}"}
        pattern = args.get("pattern", "*")
        entries = []
        for p in sorted(base.glob(pattern))[:200]:
            entries.append({"path": str(p), "dir": p.is_dir(), "size": p.stat().st_size if p.is_file() else None})
        return {"entries": entries}

    def fs_apply_diff(args: dict[str, Any]) -> dict[str, Any]:
        path = Path(args["path"]).expanduser()
        denied = _blocked(path)
        if denied is not None:
            return denied
        if not path.is_file():
            return {"ok": False, "error": f"file not found: {path}"}
        original = path.read_text(encoding="utf-8")
        try:
            patched = apply_unified_diff(original, str(args.get("diff", "")))
        except Exception as exc:  # noqa: BLE001 — PatchApplyError is data for the LLM
            return {"ok": False, "error": f"diff rejected: {exc}"}
        atomic_write(path, patched)
        return {"ok": True, "path": str(path), "bytes_changed": len(patched) - len(original)}

    def fs_write(args: dict[str, Any]) -> dict[str, Any]:
        path = Path(args["path"]).expanduser()
        denied = _blocked(path)
        if denied is not None:
            return denied
        atomic_write(path, str(args.get("content", "")))
        return {"ok": True, "path": str(path)}

    def shell_exec(args: dict[str, Any]) -> dict[str, Any]:
        if os.environ.get("ECHO_ALLOW_SHELL") != "1":
            return {"ok": False, "error": "shell disabled; set ECHO_ALLOW_SHELL=1 to enable"}
        import time as _time

        cmd = args.get("command") or []
        if isinstance(cmd, str):
            cmd = cmd.split()
        timeout = float(args.get("timeout", 60.0))
        started = _time.monotonic()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return {
                "ok": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": proc.stdout[-8000:],
                "stderr": proc.stderr[-3000:],
                "elapsed_ms": int((_time.monotonic() - started) * 1000),
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timeout after {timeout:.0f}s"}
        except FileNotFoundError as exc:
            return {"ok": False, "error": f"executable not found: {exc}"}

    specs = [
        (ToolSpec(name="fs_read", description="Read a local file (bounded excerpt).",
                  parameters={"type": "object", "properties": {"path": {"type": "string"}, "max_chars": {"type": "integer"}}, "required": ["path"]}), fs_read),
        (ToolSpec(name="fs_list", description="List files matching a glob under a directory.",
                  parameters={"type": "object", "properties": {"dir": {"type": "string"}, "pattern": {"type": "string"}}}), fs_list),
        (ToolSpec(name="fs_apply_diff", description="Apply a unified diff to a file atomically (PREFERRED edit mechanism).",
                  parameters={"type": "object", "properties": {"path": {"type": "string"}, "diff": {"type": "string"}}, "required": ["path", "diff"]}, tags=["delta"]), fs_apply_diff),
        (ToolSpec(name="fs_write", description="Atomically write file content (use only when a diff is impractical).",
                  parameters={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}), fs_write),
        (ToolSpec(name="shell_exec", description="Run a local command (requires ECHO_ALLOW_SHELL=1).",
                  parameters={"type": "object", "properties": {"command": {"type": "array", "items": {"type": "string"}}, "timeout": {"type": "number"}}, "required": ["command"]}), shell_exec),
    ]
    for spec, handler in specs:
        registry.register(spec, handler)


async def build_kernel(
    config: EchoConfig,
    *,
    include_gateways: bool = True,
) -> tuple[KernelContext, PluginManager, Orchestrator]:
    """Assemble the microkernel: bus → registry → context → plugins → loop."""
    config.ensure_dirs()
    bus = EventBus()
    registry = ToolRegistry(bus)
    ctx = KernelContext(bus=bus, config=config, tools=registry, data_dir=config.data_dir)
    _register_builtin_tools(registry, config)

    pm = PluginManager(bus, config)
    ctx.services["plugin_manager"] = pm  # self-healing hot-reload + manifests

    provider = OpenAICompatProvider(config)
    orchestrator = Orchestrator(bus=bus, config=config, tools=registry, provider=provider)

    # Built-in plugin order matters: memory + llm first (services others use).
    pm.register(CognitiveMemoryPlugin(config))
    pm.register(provider)
    pm.register(SelfHealingPlugin(config))
    pm.register(SkillIngester(config))
    pm.register(WebPlugin(config))
    from python_core.inter_agent.universal_client import UniversalClientPlugin

    pm.register(UniversalClientPlugin(config))
    if include_gateways:
        if config.discord_token:
            pm.register(DiscordGateway(config.discord_token))
        if config.telegram_token:
            pm.register(TelegramGateway(config.telegram_token))
    pm.register(orchestrator)  # last: the loop sees the fully wired world

    # External plugin discovery (config.plugin_dirs) — infinite expandability.
    for d in config.plugin_dirs:
        pm.discover(Path(d).expanduser())

    await pm.start_all(ctx)
    return ctx, pm, orchestrator


# ─────────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────────
async def _cmd_run(task_text: str, config: EchoConfig, *, as_json: bool = False) -> int:
    interactive = (not as_json) and color_enabled(sys.stdout)
    _ctx, pm, orchestrator = await build_kernel(config, include_gateways=False)
    try:
        if interactive:
            spinner = Spinner("thinking")
            await spinner.start()
            try:
                result = await orchestrator.handle_task(task_text, source="cli", on_event=spinner.on_event)
            finally:
                await spinner.stop()
            render_response(result.get("response", ""))
            render_steps(result.get("steps", []))
            if result.get("cached"):
                print(paint("  (cached)", "2"))
            return 0 if result.get("response") is not None else 1

        result = await orchestrator.handle_task(task_text, source="cli")
        print(json.dumps(
            {
                "response": result.get("response", ""),
                "steps": result.get("steps", []),
                "cached": result.get("cached", False),
                "elapsed_ms": result.get("elapsed_ms"),
            },
            indent=2,
            ensure_ascii=False,
        ))
        return 0 if result.get("response") is not None else 1
    finally:
        await pm.stop_all()


async def _cmd_start(args: argparse.Namespace, config: EchoConfig) -> int:
    ctx, pm, orchestrator = await build_kernel(config, include_gateways=True)
    server: ManifestServer | None = None
    bridge = None
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):
            pass  # e.g. Windows proactor loop

    try:
        # Universal manifest protocol (MODULE 5): be discoverable.
        server = ManifestServer(ctx)
        await server.start()
    except (Exception, SystemExit) as exc:  # noqa: BLE001 — manifest server is optional
        ctx.logger.warning("manifest server unavailable (%s); continuing without it", exc)
        server = None

    # Optional high-throughput Rust bus bridge (when echo-bus is running).
    if os.environ.get("ECHO_ENABLE_BUS") == "1":
        from python_core.inter_agent.rust_bridge import RustBusBridge

        bridge = RustBusBridge(ctx.bus, config.resolved_bus_socket)
        bridge.start()
        bridge.forward(("task:*", "tool:executed"))
        ctx.logger.info("rust bus bridge enabled → %s", config.resolved_bus_socket)

    try:
        if args.headless:
            print("echo-agent daemon running (headless). Ctrl+C to stop.")
            await stop.wait()
        else:
            await _repl(orchestrator, stop)
    finally:
        if server is not None:
            await server.stop()
        await pm.stop_all()
    return 0


async def _repl(orchestrator: Orchestrator, stop: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    print(banner(__version__))
    print()
    while not stop.is_set():
        prompt = paint("echo> ", "36", "1")
        line_task = loop.run_in_executor(None, input, prompt)
        stop_task = asyncio.create_task(stop.wait())
        done, _pending = await asyncio.wait({line_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        if stop_task in done:
            line_task.cancel()
            print()
            break
        stop_task.cancel()
        try:
            line = line_task.result().strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        try:
            piped_stdin = not sys.stdin.isatty()
        except (AttributeError, ValueError):
            piped_stdin = True
        if piped_stdin:
            print()  # piped stdin has no terminal echo; keep output on its own line
        if line in ("exit", "quit"):
            print()
            break
        if not line:
            continue

        spinner = Spinner("thinking")
        await spinner.start()
        try:
            result = await orchestrator.handle_task(line, source="repl", on_event=spinner.on_event)
        finally:
            await spinner.stop()

        render_response(result.get("response", ""))
        render_steps(result.get("steps", []))
        if result.get("cached"):
            print(paint("  (cached)", "2"))
        print()


async def _cmd_skills_add(source: str, config: EchoConfig) -> int:
    ctx, pm, _ = await build_kernel(config, include_gateways=False)
    try:
        ingester = ctx.service("skills")
        if ingester is None:
            print("skills plugin unavailable", file=sys.stderr)
            return 1
        report = await ingester.ingest(source)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report.get("ok") else 1
    finally:
        await pm.stop_all()


def _cmd_tools(config: EchoConfig) -> int:
    """Static view: builtin tools + any installed skill manifests."""
    registry = ToolRegistry()
    _register_builtin_tools(registry, config)
    for spec in registry.list():
        print(f"- {spec.name:28s} [{spec.source}] {spec.description[:80]}")
    return 0


def _cmd_plugins(config: EchoConfig) -> int:
    """Dry manifest of the built-in plugin set (no kernel started)."""
    builtins = [
        {"name": "memory", "capabilities": CognitiveMemoryPlugin.capabilities},
        {"name": "llm_openai_compat", "capabilities": OpenAICompatProvider.capabilities},
        {"name": "self_healing", "capabilities": SelfHealingPlugin.capabilities},
        {"name": "skills", "capabilities": SkillIngester.capabilities},
        {"name": "web", "capabilities": WebPlugin.capabilities},
        {"name": "universal_client", "capabilities": ["inter-agent-client", "http"]},
        {"name": "orchestrator", "capabilities": ["agent-loop", "structured-output", "sessions"]},
    ]
    if config.discord_token:
        builtins.append({"name": "gateway_discord", "capabilities": DiscordGateway.capabilities})
    if config.telegram_token:
        builtins.append({"name": "gateway_telegram", "capabilities": TelegramGateway.capabilities})
    for p in builtins:
        print(f"- {p['name']:22s} {', '.join(p['capabilities'])}")
    return 0


def _pairing_store(config: EchoConfig) -> PairingStore:
    return PairingStore(
        config.data_dir / "gateways" / "pairing.json",
        ttl=config.gateway_pairing_ttl,
        max_pending=config.gateway_pairing_max_pending,
    )


def _cmd_pairing(args: argparse.Namespace, config: EchoConfig) -> int:
    """List/approve/dismiss/revoke DM pairing requests (OpenClaw-style)."""
    store = _pairing_store(config)
    action = args.pairing_command
    if action == "list":
        pending = store.list_pending(args.channel)
        approved = store.list_approved(args.channel)
        if args.as_json:
            # ``requests`` mirrors OpenClaw's pairing JSON; ``pending``/``approved``
            # are kept for older scripts.
            print(json.dumps(
                {
                    "channel": args.channel or "",
                    "requests": pending,
                    "pending": pending,
                    "approved": approved,
                },
                indent=2,
                ensure_ascii=False,
            ))
            return 0
        if not pending and not approved:
            print("no pending or approved pairings")
            return 0
        if pending:
            print("PENDING")
            for item in pending:
                who = f"@{item['username']}" if item["username"] else item["sender_id"]
                print(
                    f"  {item['channel']:9s} {item['code']}  {who:24s} "
                    f"user={item['sender_id']}  expires_in={item['expires_in']}s"
                )
        if approved:
            print("APPROVED")
            for item in approved:
                who = f"@{item['username']}" if item["username"] else item["sender_id"]
                print(f"  {item['channel']:9s} {who:24s} user={item['sender_id']}")
        return 0

    if action == "approve":
        result = store.approve(args.channel, args.code, notify=bool(getattr(args, "notify", False)))
        if result is None:
            print(
                f"no pending pairing with code {args.code!r} on {args.channel}. "
                f"Run `echo-agent pairing list {args.channel}` to see pending codes.",
                file=sys.stderr,
            )
            return 1
        who = f"@{result['username']}" if result["username"] else result["sender_id"]
        note = (
            " (requester notified)" if result.get("notified")
            else " (use --notify to message the requester)"
        )
        print(f"approved {args.channel} user {who} ({result['sender_id']}){note}")
        return 0

    if action == "dismiss":
        removed = store.dismiss(args.channel, args.code)
        if removed is None:
            print(
                f"no pending pairing with code {args.code!r} on {args.channel}. "
                f"Run `echo-agent pairing list {args.channel}` to see pending codes.",
                file=sys.stderr,
            )
            return 1
        print(f"dismissed {args.channel} pairing {args.code} for user {removed['sender_id']}")
        return 0

    if action == "revoke":
        removed = store.revoke(args.channel, args.sender_id)
        if not removed:
            print(f"no pairing found for {args.channel} user {args.sender_id}", file=sys.stderr)
            return 1
        print(f"revoked {args.channel} user {args.sender_id}")
        return 0

    print(f"unknown pairing command: {action}", file=sys.stderr)
    return 1


def _config_path(args: argparse.Namespace, config: EchoConfig) -> Path:
    if getattr(args, "config", None):
        return Path(args.config)
    return config.data_dir / "config.toml"


def _load_raw_toml(path: Path) -> dict[str, Any]:
    if path.is_file():
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    return {}


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_scalar(item) for item in value) + "]"
    return json.dumps(str(value), ensure_ascii=False)


def _dump_raw_toml(path: Path, raw: dict[str, Any]) -> None:
    lines: list[str] = []
    for section, value in raw.items():
        if isinstance(value, dict):
            lines.append(f"[{section}]")
            for key, item in value.items():
                lines.append(f"{key} = {_toml_scalar(item)}")
            lines.append("")
        else:
            lines.append(f"{section} = {_toml_scalar(value)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(("\n".join(lines).rstrip() + "\n"), encoding="utf-8")


def _prompt(label: str, default: Any = "") -> str:
    suffix = f" [{default}]" if default not in ("", None) else ""
    try:
        value = input(f"{label}{suffix}: ").strip()
    except EOFError:
        value = ""
    return value if value else ("" if default is None else str(default))


def _prompt_bool(label: str, default: bool) -> bool:
    value = _prompt(label + " (y/n)", "y" if default else "n").lower()
    if value in {"y", "yes", "true", "1"}:
        return True
    if value in {"n", "no", "false", "0"}:
        return False
    return bool(default)


def _prompt_list(label: str, current: Any) -> list[str]:
    default = ", ".join(current) if isinstance(current, list) else str(current or "")
    value = _prompt(label, default)
    if value == default:
        return list(current) if isinstance(current, list) else ([str(current)] if current else [])
    return [item.strip() for item in value.split(",") if item.strip()]


def _pid_path(config: EchoConfig) -> Path:
    return config.data_dir / "echo-agent.pid"


def _log_path(config: EchoConfig) -> Path:
    return config.data_dir / "echo-agent.log"


def _read_pid(config: EchoConfig) -> int | None:
    path = _pid_path(config)
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _cmd_status(config: EchoConfig) -> int:
    pid = _read_pid(config)
    if pid is not None and _pid_alive(pid):
        print(f"echo-agent running (pid {pid})")
        print(f"  log: {_log_path(config)}")
        return 0
    if pid is not None:
        try:
            _pid_path(config).unlink()
        except OSError:
            pass
    print("echo-agent not running")
    return 1


def _cmd_stop(config: EchoConfig) -> int:
    pid = _read_pid(config)
    if pid is None or not _pid_alive(pid):
        try:
            _pid_path(config).unlink()
        except OSError:
            pass
        print("echo-agent not running")
        return 0
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        print(f"could not stop pid {pid}: {exc}", file=sys.stderr)
        return 1
    for _ in range(50):
        if not _pid_alive(pid):
            break
        time.sleep(0.1)
    if _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        time.sleep(0.2)
    try:
        _pid_path(config).unlink()
    except OSError:
        pass
    print(f"stopped echo-agent (pid {pid})")
    return 0


def _cmd_start_background(config: EchoConfig) -> int:
    pid = _read_pid(config)
    if pid is not None and _pid_alive(pid):
        print(f"echo-agent already running (pid {pid})")
        return 0
    config.ensure_dirs()
    exe = shutil.which("echo-agent") or sys.argv[0] or "echo-agent"
    log = _log_path(config)
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log, "ab")
    try:
        proc = subprocess.Popen(
            [exe, "start", "--headless"],
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )
    finally:
        handle.close()
    time.sleep(1.2)
    if not _pid_alive(proc.pid):
        print("echo-agent failed to start; see " + str(log), file=sys.stderr)
        return 1
    _pid_path(config).write_text(str(proc.pid), encoding="utf-8")
    print(f"started echo-agent (pid {proc.pid})")
    print(f"  log: {log}")
    return 0


def _cmd_restart(config: EchoConfig) -> int:
    _cmd_stop(config)
    return _cmd_start_background(config)


def _cmd_gateway_list(config: EchoConfig) -> int:
    print("Gateway status")
    for platform in ("telegram", "discord"):
        token = config.telegram_token if platform == "telegram" else config.discord_token
        dm_policy = config.telegram_dm_policy if platform == "telegram" else config.discord_dm_policy
        group_policy = config.telegram_group_policy if platform == "telegram" else config.discord_group_policy
        groups = config.telegram_groups if platform == "telegram" else config.discord_guilds
        print(
            f"  {platform:9s} token={'set' if token else 'unset':5s} "
            f"dm={dm_policy:8s} groups={group_policy:8s} allowed={len(groups)}"
        )
    print(f"  pairing ttl={config.gateway_pairing_ttl:.0f}s reply_to={config.gateway_reply_to_mode}")
    return 0


async def _cmd_gateway_test(platform: str, config: EchoConfig) -> int:
    token = config.telegram_token if platform == "telegram" else config.discord_token
    if not token:
        print(f"no {platform} token configured", file=sys.stderr)
        return 1
    try:
        if platform == "telegram":
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"https://api.telegram.org/bot{token}/getMe")
            data = response.json()
            if not data.get("ok"):
                print(f"telegram token rejected: {data.get('description', 'unknown error')}", file=sys.stderr)
                return 1
            result = data.get("result") or {}
            print(f"telegram ok: @{result.get('username')} (id {result.get('id')})")
        else:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    "https://discord.com/api/v10/users/@me",
                    headers={"Authorization": f"Bot {token}"},
                )
            if response.status_code != 200:
                print(f"discord token rejected: HTTP {response.status_code}", file=sys.stderr)
                return 1
            data = response.json()
            print(f"discord ok: {data.get('username')} (id {data.get('id')})")
    except Exception as exc:  # noqa: BLE001 — network failure should be readable
        print(f"{platform} test failed: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_gateway(args: argparse.Namespace, config: EchoConfig) -> int:
    action = args.gateway_command
    if action == "list":
        return _cmd_gateway_list(config)

    if action == "test":
        return asyncio.run(_cmd_gateway_test(args.platform, config))

    path = _config_path(args, config)

    if action == "remove":
        platform = args.platform
        raw = _load_raw_toml(path)
        section = raw.get(platform)
        if isinstance(section, dict):
            section.pop("token", None)
            _dump_raw_toml(path, raw)
        print(f"removed {platform} token from {path}")
        return 0

    if action != "configure":
        print(f"unknown gateway command: {action}", file=sys.stderr)
        return 1

    platform = args.platform
    raw = _load_raw_toml(path)
    section = raw.get(platform)
    if not isinstance(section, dict):
        section = {}
        raw[platform] = section

    current_token = str(section.get("token", "") or (config.telegram_token if platform == "telegram" else config.discord_token))
    if getattr(args, "token", None) is not None:
        token = args.token
    else:
        if current_token:
            print(f"Current token: {current_token[:6]}...{current_token[-4:]}")
        try:
            token = getpass.getpass("Bot token (blank to keep current): ").strip()
        except EOFError:
            token = ""
        if not token:
            token = current_token
    if not token:
        print("a bot token is required", file=sys.stderr)
        return 1
    section["token"] = token

    dm_default = str(section.get("dm_policy", config.telegram_dm_policy if platform == "telegram" else config.discord_dm_policy))
    group_default = str(section.get("group_policy", config.telegram_group_policy if platform == "telegram" else config.discord_group_policy))
    dm_policy = getattr(args, "dm_policy", None)
    if dm_policy is None:
        dm_policy = _prompt("DM policy (pairing/allowlist/open/disabled)", dm_default)
    if dm_policy not in {"pairing", "allowlist", "open", "disabled"}:
        dm_policy = dm_default
    section["dm_policy"] = dm_policy

    allow_default = section.get("allow_from", config.telegram_allow_from if platform == "telegram" else config.discord_allow_from)
    allow_from = getattr(args, "allow_from", None)
    if allow_from is None:
        allow_from = _prompt_list("DM allow_from (comma-separated)", allow_default)
    section["allow_from"] = allow_from

    group_policy = getattr(args, "group_policy", None)
    if group_policy is None:
        group_policy = _prompt("Group policy (allowlist/open/disabled)", group_default)
    if group_policy not in {"allowlist", "open", "disabled"}:
        group_policy = group_default
    section["group_policy"] = group_policy

    groups_default = section.get("groups" if platform == "telegram" else "guilds", config.telegram_groups if platform == "telegram" else config.discord_guilds)
    groups_value = getattr(args, "groups", None)
    if groups_value is None and platform == "discord":
        groups_value = getattr(args, "guilds", None)
    if groups_value is None:
        label = "Allowed chat IDs" if platform == "telegram" else "Allowed guild IDs"
        groups_value = _prompt_list(label, groups_default)
    section["groups" if platform == "telegram" else "guilds"] = groups_value

    channels_default = section.get("channels", config.discord_channels)
    channels_value = getattr(args, "channels", None)
    if platform == "discord":
        if channels_value is None:
            channels_value = _prompt_list("Allowed channel IDs (empty = all)", channels_default)
        section["channels"] = channels_value

    group_allow_default = section.get("group_allow_from", config.telegram_group_allow_from if platform == "telegram" else config.discord_group_allow_from)
    group_allow_value = getattr(args, "group_allow_from", None)
    if group_allow_value is None:
        group_allow_value = _prompt_list("Group sender allowlist (empty = any member)", group_allow_default)
    section["group_allow_from"] = group_allow_value

    require_default = bool(section.get("require_mention", config.telegram_require_mention if platform == "telegram" else config.discord_require_mention))
    require_value = getattr(args, "require_mention", None)
    if require_value is None:
        require_mention = _prompt_bool("Require @mention in groups", require_default)
    else:
        require_mention = require_value == "true"
    section["require_mention"] = require_mention

    if platform == "telegram":
        link_default = bool(section.get("link_preview", config.telegram_link_preview))
        link_value = getattr(args, "link_preview", None)
        if link_value is None:
            link_preview = _prompt_bool("Enable link previews", link_default)
        else:
            link_preview = link_value == "true"
        section["link_preview"] = link_preview
        section.setdefault("ignore_other_mentions", True)
    else:
        prefix_default = str(section.get("command_prefix", config.discord_command_prefix))
        prefix_value = getattr(args, "command_prefix", None)
        if prefix_value is None:
            prefix_value = _prompt("Command prefix", prefix_default)
        section["command_prefix"] = prefix_value or prefix_default
        slash_default = bool(section.get("slash_commands", config.discord_slash_commands))
        slash_value = getattr(args, "slash_commands", None)
        if slash_value is None:
            slash_commands = _prompt_bool("Enable slash commands", slash_default)
        else:
            slash_commands = slash_value == "true"
        section["slash_commands"] = slash_commands
        section.setdefault("allow_bots", False)
        section.setdefault("group_dm_enabled", False)

    gateway_section = raw.setdefault("gateway", {})
    if not isinstance(gateway_section, dict):
        gateway_section = {}
        raw["gateway"] = gateway_section
    reply_default = str(gateway_section.get("reply_to_mode", config.gateway_reply_to_mode))
    reply_value = getattr(args, "reply_to_mode", None)
    if reply_value is None:
        reply_value = _prompt("Reply threading (off/first/all)", reply_default)
    if reply_value not in {"off", "first", "all"}:
        reply_value = reply_default
    gateway_section["reply_to_mode"] = reply_value

    typing_default = bool(gateway_section.get("typing", config.gateway_typing))
    typing_value = getattr(args, "typing", None)
    if typing_value is None:
        typing_enabled = _prompt_bool("Typing indicators", typing_default)
    else:
        typing_enabled = typing_value == "true"
    gateway_section["typing"] = typing_enabled

    _dump_raw_toml(path, raw)
    print(f"configured {platform} in {path}")
    print("run `echo-agent start` (or restart the daemon) to use it")
    return 0


def _cmd_bus(config: EchoConfig) -> int:
    """Launch the Rust echo-bus (built binary preferred, cargo fallback)."""
    socket = config.resolved_bus_socket
    binary = Path(os.environ.get("ECHO_BUS_BIN", Path(__file__).resolve().parents[2] / "target" / "release" / "echo-bus"))
    cmd = [str(binary), "--socket", socket] if binary.is_file() else ["cargo", "run", "--release", "--", "--socket", socket]
    print(f"launching: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
    return proc.returncode or 0


def _first_run_hook(config: EchoConfig, args: argparse.Namespace) -> None:
    """Launch the wizard on first interactive run; stay silent for scripts.

    The kernel boots without an LLM, so a non-TTY run only gets one hint line
    and otherwise continues with defaults.
    """
    if getattr(args, "no_setup", False) or is_setup_complete(config.data_dir):
        return
    try:
        interactive = sys.stdin.isatty()
    except (AttributeError, ValueError, OSError):
        interactive = False
    if not interactive:
        print("hint: run `echo-agent setup` to configure; continuing with defaults")
        return
    print("first run — launching setup wizard…")
    code = run_setup(config, SetupOptions())
    if code != 0:
        raise SystemExit(code)


# ─────────────────────────────────────────────────────────────────────────────
# Argparse wiring
# ─────────────────────────────────────────────────────────────────────────────
def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="echo-agent", description="Local-first self-improving AI agent")
    parser.add_argument("--config", default=None, help="path to config.toml")
    parser.add_argument("--verbose", "-v", action="store_true")
    sub = parser.add_subparsers(dest="command", required=False)

    setup_p = sub.add_parser("setup", aliases=["onboard"], help="interactive first-run setup wizard")
    setup_p.add_argument("--reset", action="store_true", help="ignore existing configuration")
    setup_p.add_argument("--quick", action="store_true", help="accept all defaults, no prompts")
    setup_p.add_argument("--llm-base-url")
    setup_p.add_argument("--llm-model")
    setup_p.add_argument("--llm-api-key")
    setup_p.add_argument("--telegram-token")
    setup_p.add_argument("--discord-token")
    setup_p.add_argument("--data-dir")
    setup_p.add_argument("--port", type=int)
    setup_p.add_argument("--yes", "-y", action="store_true", help="non-interactive")

    run_p = sub.add_parser("run", help="execute one task")
    run_p.add_argument("task", help="task text")
    run_p.add_argument("--json", action="store_true", dest="as_json", help="machine-readable JSON output (also implied when stdout is not a TTY)")
    run_p.add_argument("--no-setup", action="store_true", help="skip the first-run wizard")

    start_p = sub.add_parser("start", help="daemon mode (manifest server + gateways + REPL)")
    start_p.add_argument("--headless", action="store_true", help="no REPL; run until signaled")
    start_p.add_argument("--no-setup", action="store_true", help="skip the first-run wizard")

    agent_p = sub.add_parser("agent", help="alias for `start`")
    agent_p.add_argument("--headless", action="store_true", help="no REPL; run until signaled")
    agent_p.add_argument("--no-setup", action="store_true", help="skip the first-run wizard")

    doctor_p = sub.add_parser("doctor", help="diagnose the installation")
    doctor_p.add_argument("--probe", action="store_true", help="probe LLM endpoint and gateway tokens")

    skills_p = sub.add_parser("skills", help="skill management")
    skills_sub = skills_p.add_subparsers(dest="skills_command", required=True)
    add_p = skills_sub.add_parser("add", help="ingest a repo URL or local path")
    add_p.add_argument("source")

    sub.add_parser("tools", help="list registered tools")
    sub.add_parser("plugins", help="list built-in plugins")

    sub.add_parser("status", help="show whether the background daemon is running")
    sub.add_parser("stop", help="stop the background daemon")
    sub.add_parser("restart", help="restart the background daemon")

    pairing_p = sub.add_parser("pairing", help="list/approve/revoke DM pairing codes")
    pairing_sub = pairing_p.add_subparsers(dest="pairing_command", required=True)
    pairing_list = pairing_sub.add_parser("list", help="show pending and approved senders")
    pairing_list.add_argument("channel", nargs="?", help="telegram or discord (default: all)")
    pairing_list.add_argument("--json", action="store_true", dest="as_json")
    pairing_approve = pairing_sub.add_parser("approve", help="approve a pending code")
    pairing_approve.add_argument("channel", choices=["telegram", "discord"])
    pairing_approve.add_argument("code")
    pairing_approve.add_argument(
        "--notify", action="store_true", help="send a confirmation to the requester's DM"
    )
    pairing_dismiss = pairing_sub.add_parser("dismiss", help="dismiss a pending code without approving")
    pairing_dismiss.add_argument("channel", choices=["telegram", "discord"])
    pairing_dismiss.add_argument("code")
    pairing_revoke = pairing_sub.add_parser("revoke", help="revoke an approved sender")
    pairing_revoke.add_argument("channel", choices=["telegram", "discord"])
    pairing_revoke.add_argument("sender_id")

    gateway_p = sub.add_parser("gateway", help="configure/test/list/start/stop chat gateways")
    gateway_sub = gateway_p.add_subparsers(dest="gateway_command", required=True)
    gateway_sub.add_parser("list", help="show gateway status")
    gateway_sub.add_parser("status", help="show daemon status")
    gateway_sub.add_parser("start", help="start the daemon in the background")
    gateway_sub.add_parser("stop", help="stop the background daemon")
    gateway_sub.add_parser("restart", help="restart the background daemon")
    gateway_test = gateway_sub.add_parser("test", help="validate a bot token")
    gateway_test.add_argument("platform", choices=["telegram", "discord"])
    gateway_configure = gateway_sub.add_parser("configure", help="interactively configure a gateway")
    gateway_configure.add_argument("platform", choices=["telegram", "discord"])
    gateway_configure.add_argument("--token")
    gateway_configure.add_argument("--dm-policy", dest="dm_policy", choices=["pairing", "allowlist", "open", "disabled"])
    gateway_configure.add_argument("--allow-from", dest="allow_from")
    gateway_configure.add_argument("--group-policy", dest="group_policy", choices=["allowlist", "open", "disabled"])
    gateway_configure.add_argument("--groups", dest="groups")
    gateway_configure.add_argument("--guilds", dest="guilds")
    gateway_configure.add_argument("--channels", dest="channels")
    gateway_configure.add_argument("--group-allow-from", dest="group_allow_from")
    gateway_configure.add_argument("--require-mention", dest="require_mention", choices=["true", "false"])
    gateway_configure.add_argument("--command-prefix", dest="command_prefix")
    gateway_configure.add_argument("--slash-commands", dest="slash_commands", choices=["true", "false"])
    gateway_configure.add_argument("--link-preview", dest="link_preview", choices=["true", "false"])
    gateway_configure.add_argument("--reply-to-mode", dest="reply_to_mode", choices=["off", "first", "all"])
    gateway_configure.add_argument("--typing", dest="typing", choices=["true", "false"])
    gateway_remove = gateway_sub.add_parser("remove", help="remove a gateway token")
    gateway_remove.add_argument("platform", choices=["telegram", "discord"])

    channels_p = sub.add_parser("channels", help="manage chat channels (alias for gateway)")
    channels_sub = channels_p.add_subparsers(dest="channels_command", required=True)
    channels_sub.add_parser("list", help="list channel status")
    channels_status = channels_sub.add_parser("status", help="show channel/daemon status")
    channels_status.add_argument("--probe", action="store_true", help="validate configured tokens")
    channels_add = channels_sub.add_parser("add", help="add/configure a channel")
    channels_add.add_argument("--channel", required=True, choices=["telegram", "discord"])
    channels_add.add_argument("--token")
    channels_add.add_argument("--dm-policy", dest="dm_policy", choices=["pairing", "allowlist", "open", "disabled"])
    channels_add.add_argument("--allow-from", dest="allow_from")
    channels_add.add_argument("--group-policy", dest="group_policy", choices=["allowlist", "open", "disabled"])
    channels_add.add_argument("--groups", dest="groups")
    channels_add.add_argument("--guilds", dest="guilds")
    channels_add.add_argument("--channels", dest="channels")
    channels_add.add_argument("--group-allow-from", dest="group_allow_from")
    channels_add.add_argument("--require-mention", dest="require_mention", choices=["true", "false"])
    channels_add.add_argument("--command-prefix", dest="command_prefix")
    channels_add.add_argument("--slash-commands", dest="slash_commands", choices=["true", "false"])
    channels_add.add_argument("--link-preview", dest="link_preview", choices=["true", "false"])
    channels_add.add_argument("--reply-to-mode", dest="reply_to_mode", choices=["off", "first", "all"])
    channels_add.add_argument("--typing", dest="typing", choices=["true", "false"])
    channels_remove = channels_sub.add_parser("remove", help="remove a channel token")
    channels_remove.add_argument("--channel", required=True, choices=["telegram", "discord"])
    channels_test = channels_sub.add_parser("test", help="validate a channel token")
    channels_test.add_argument("--channel", required=True, choices=["telegram", "discord"])

    sub.add_parser("bus", help="launch the Rust echo-bus IPC server")
    sub.add_parser("version", help="print version")

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    config = EchoConfig.load(args.config)

    if args.command is None:
        if not is_setup_complete(config.data_dir):
            return run_setup(config, SetupOptions())
        parser.print_help()
        print()
        _cmd_status(config)
        return 0
    if args.command in {"setup", "onboard"}:
        return run_setup(config, SetupOptions(
            reset=args.reset,
            quick=args.quick,
            llm_base_url=args.llm_base_url,
            llm_model=args.llm_model,
            llm_api_key=args.llm_api_key,
            telegram_token=args.telegram_token,
            discord_token=args.discord_token,
            data_dir=args.data_dir,
            port=args.port,
            yes=args.yes,
            interactive=False if args.yes else None,
        ))
    if args.command == "doctor":
        from python_core.doctor import run_doctor

        return run_doctor(config, probe=args.probe)
    if args.command == "run":
        _first_run_hook(config, args)
        return asyncio.run(_cmd_run(args.task, config, as_json=args.as_json))
    if args.command in {"start", "agent"}:
        _first_run_hook(config, args)
        return asyncio.run(_cmd_start(args, config))
    if args.command == "skills":
        if args.skills_command == "add":
            return asyncio.run(_cmd_skills_add(args.source, config))
    if args.command == "tools":
        return _cmd_tools(config)
    if args.command == "plugins":
        return _cmd_plugins(config)
    if args.command == "pairing":
        return _cmd_pairing(args, config)
    if args.command == "gateway":
        # map gateway service commands
        if args.gateway_command in {"status"}:
            return _cmd_status(config)
        if args.gateway_command == "start":
            return _cmd_start_background(config)
        if args.gateway_command == "stop":
            return _cmd_stop(config)
        if args.gateway_command == "restart":
            return _cmd_restart(config)
        return _cmd_gateway(args, config)
    if args.command == "channels":
        # Normalize OpenClaw-style channel commands onto the gateway implementation.
        cmd = args.channels_command
        if cmd == "list":
            args.gateway_command = "list"
            return _cmd_gateway(args, config)
        if cmd == "status":
            code = _cmd_gateway_list(config)
            if getattr(args, "probe", False):
                for platform in ("telegram", "discord"):
                    token = config.telegram_token if platform == "telegram" else config.discord_token
                    if token:
                        asyncio.run(_cmd_gateway_test(platform, config))
            return code
        if cmd == "add":
            args.gateway_command = "configure"
            args.platform = args.channel
            # carry --channel-only fields onto the expected attribute names
            if getattr(args, "guilds", None) is not None and getattr(args, "groups", None) is None and args.platform == "discord":
                args.groups = args.guilds
            return _cmd_gateway(args, config)
        if cmd == "remove":
            args.gateway_command = "remove"
            args.platform = args.channel
            return _cmd_gateway(args, config)
        if cmd == "test":
            args.gateway_command = "test"
            args.platform = args.channel
            return _cmd_gateway(args, config)
        print(f"unknown channels command: {cmd}", file=sys.stderr)
        return 1
    if args.command == "status":
        return _cmd_status(config)
    if args.command == "stop":
        return _cmd_stop(config)
    if args.command == "restart":
        return _cmd_restart(config)
    if args.command == "bus":
        return _cmd_bus(config)
    if args.command == "version":
        print(f"echo-agent {__version__}")
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(cli())
