"""Read-only installation health checks for ``echo-agent doctor``.

Prints ✓/✗/! per check and returns non-zero only for hard failures (things
that stop the agent from running). Optional integrations degrade to warnings.
"""

from __future__ import annotations

import importlib.util
import json
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


def _optional(mod: str) -> bool:
    return bool(importlib.util.find_spec(mod))


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

    # 10. installed extras / optional capabilities
    extras = []
    if _optional("tiktoken"):
        extras.append("native(tiktoken)")
    if _optional("chromadb"):
        extras.append("native(chromadb)")
    if _optional("discord"):
        extras.append("discord")
    if _optional("onnxruntime") and _optional("tokenizers"):
        extras.append("embeddings(onnxruntime+tokenizers)")
    if _optional("scrapling"):
        extras.append("scrapling")
    if extras:
        _line(out, "✓", "extras", ", ".join(extras))
    else:
        _line(out, "!", "extras", "none (rerun install.sh --extras all for full capability)")

    # 11. embeddings backend (PHASE 3)
    embeddings_on = bool(loaded.enable_embeddings)
    backend = str(loaded.embeddings_backend or "auto")
    onnx_ok = _optional("onnxruntime") and _optional("tokenizers")
    model_dir = Path(loaded.embedding_model_dir or loaded.data_dir / "models" / "all-MiniLM-L6-v2")
    model_ok = (model_dir / "model.onnx").is_file() and (model_dir / "tokenizer.json").is_file()
    active = "onnx" if (embeddings_on and onnx_ok and model_ok) else "hash (offline fallback)"
    if embeddings_on and not onnx_ok:
        _line(out, "!", "embeddings", f"backend={backend} → {active} (onnxruntime/tokenizers not installed)")
    elif embeddings_on and not model_ok:
        _line(out, "!", "embeddings", f"backend={backend} → {active} (model files missing in {model_dir})")
    else:
        _line(out, "✓", "embeddings", f"backend={backend} → {active}")

    # 12. data-dir health (PHASE 1/2 memory files)
    memory_dir = loaded.memory_dir
    profile = memory_dir.parent / "profiles" / "owner.json"
    user_md = memory_dir / "USER.md"
    memory_md = memory_dir / "MEMORY.md"
    files_ok = True
    if not profile.is_file():
        _line(out, "!", "profile", f"{profile} not yet created (tell the agent your name!)")
        files_ok = False
    if not user_md.is_file():
        _line(out, "!", "USER.md", f"{user_md} not yet created (first save will write it)")
    if not memory_md.is_file():
        _line(out, "!", "MEMORY.md", f"{memory_md} not yet created (first save will write it)")
    if files_ok:
        _line(out, "✓", "data dir health", f"profiles + {memory_dir.name} present")

    # 13. registered agents (PHASE 7 supervisor)
    agents_path = loaded.data_dir / "agents.json"
    try:
        agents = json.loads(agents_path.read_text(encoding="utf-8"))
        if isinstance(agents, dict) and agents:
            _line(out, "✓", "registered agents", ", ".join(str(k) for k in agents))
        elif agents_path.is_file():
            _line(out, "!", "registered agents", "none registered")
        else:
            _line(out, "!", "registered agents", "no agents.json yet")
    except (OSError, ValueError):
        _line(out, "!", "registered agents", f"{agents_path} unreadable")

    if hard_failures:
        print(f"{hard_failures} hard failure(s) — see ✗ lines above", file=out)
    else:
        print("no hard failures", file=out)
    return 1 if hard_failures else 0
