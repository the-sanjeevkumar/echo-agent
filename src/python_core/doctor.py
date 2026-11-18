"""Read-only installation health checks for ``echo-agent doctor``.

Prints ✓/✗/! per check and returns non-zero only for hard failures (things
that stop the agent from running). Optional integrations degrade to warnings.
"""

from __future__ import annotations

import os
import shutil
import socket
import sys
from pathlib import Path
from typing import Any, TextIO

from python_core import __version__
from python_core.core.config import EchoConfig
from python_core.lifecycle import read_install_metadata


def _line(stream: TextIO, mark: str, label: str, detail: str = "") -> None:
    text = f"  {mark} {label}"
    if detail:
        text += f": {detail}"
    print(text, file=stream, flush=True)


def _check_port(host: str, port: int) -> tuple[bool, str]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            sock.bind((host, port))
        return True, "free"
    except OSError as exc:
        return False, str(exc)


def _browser_dirs() -> list[Path]:
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return [cache / "ms-playwright", cache / "camoufox", cache / "scrapling"]


def run_doctor(
    config: EchoConfig,
    *,
    probe: bool = False,
    stream: TextIO | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    out = stream or sys.stdout
    hard_failures = 0
    print(f"echo-agent doctor · v{__version__}", file=out, flush=True)

    # 1. install metadata
    meta = metadata if metadata is not None else read_install_metadata(config.data_dir)
    if meta:
        version = str(meta.get("version") or "?")
        _line(out, "✓" if version == __version__ else "!", "install",
              f"method={meta.get('method', '?')} version={version}")
        if version != __version__:
            _line(out, "!", "hint", "rerun install.sh to upgrade to the current version")
    else:
        _line(out, "!", "install", "no install.json (development checkout?)")

    # 2. entry point on PATH
    exe = shutil.which("echo-agent")
    if exe:
        _line(out, "✓", "entry point", exe)
    else:
        hard_failures += 1
        _line(out, "✗", "entry point", "echo-agent not on PATH")
        _line(out, "!", "hint", "add ~/.local/bin to PATH or rerun install.sh")

    # 3. config parses (env wins, same path the daemon uses)
    try:
        loaded = EchoConfig.load()
        _line(out, "✓", "config", str(config.data_dir / "config.toml"))
    except Exception as exc:  # noqa: BLE001 — config errors are user-facing
        hard_failures += 1
        _line(out, "✗", "config", str(exc))
        loaded = config
        _line(out, "!", "hint", "fix config.toml or rerun `echo-agent setup`")

    # 4. data dir writable
    data_dir = loaded.data_dir
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        probe_file = data_dir / ".doctor-write"
        probe_file.write_text("ok", encoding="utf-8")
        probe_file.unlink()
        _line(out, "✓", "data dir", str(data_dir))
    except OSError as exc:
        hard_failures += 1
        _line(out, "✗", "data dir", str(exc))
        _line(out, "!", "hint", f"check permissions on {data_dir}")

    # 5. daemon pidfile
    pidfile = data_dir / "echo-agent.pid"
    try:
        pid = int(pidfile.read_text(encoding="utf-8").strip())
        try:
            os.kill(pid, 0)
            _line(out, "✓", "daemon", f"running (pid {pid})")
        except OSError:
            _line(out, "!", "daemon", f"stale pidfile ({pid})")
    except (OSError, ValueError):
        _line(out, "!", "daemon", "not running (no pidfile)")

    # 6. manifest port
    free, detail = _check_port(loaded.manifest_host, loaded.manifest_port)
    if free:
        _line(out, "✓", "manifest port", f"{loaded.manifest_host}:{loaded.manifest_port} free")
    else:
        _line(out, "!", "manifest port", f"{loaded.manifest_host}:{loaded.manifest_port} in use ({detail})")

    # 7. LLM endpoint
    if probe:
        from python_core.setup_wizard import probe_llm

        models = probe_llm(loaded.llm_base_url, timeout=2.0)
        if models is None:
            _line(out, "!", "LLM endpoint", f"unreachable: {loaded.llm_base_url}")
            _line(out, "!", "hint", "start Ollama/LM Studio or set [llm] base_url in config.toml")
        else:
            _line(out, "✓", "LLM endpoint", f"{loaded.llm_base_url} ({len(models)} models)")
    else:
        _line(out, "!", "LLM endpoint", f"skipped (use --probe): {loaded.llm_base_url}")

    # 8. gateway tokens
    tokens = {
        "telegram": loaded.telegram_token,
        "discord": loaded.discord_token,
    }
    if probe:
        from python_core.setup_wizard import validate_discord, validate_telegram

        for name, token in tokens.items():
            if not token:
                _line(out, "!", f"{name} token", "not configured")
                continue
            validator = validate_telegram if name == "telegram" else validate_discord
            ok, detail = validator(token)
            _line(out, "✓" if ok else "!", f"{name} token", detail)
    else:
        for name, token in tokens.items():
            _line(out, "!" if not token else "✓", f"{name} token", "set" if token else "not configured")

    # 9. scrapling browser binaries (shared caches — never auto-delete)
    installed = [p for p in _browser_dirs() if p.is_dir() and any(p.iterdir())]
    if installed:
        _line(out, "✓", "browser binaries", ", ".join(str(p) for p in installed))
    else:
        _line(out, "!", "browser binaries", "not found")
        _line(out, "!", "hint", "optional: uv run scrapling install")

    if hard_failures:
        print(f"{hard_failures} hard failure(s) — see ✗ lines above", file=out)
    else:
        print("no hard failures", file=out)
    return 1 if hard_failures else 0
