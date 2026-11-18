"""OpenClaw-style first-run setup wizard.

Standard library + the existing ``ui`` helpers only. Prompts degrade to plain
text on dumb terminals; every value is staged in memory and committed to
config.toml atomically at the very end, so Ctrl-C never leaves a partial setup.
Tokens are read with getpass, never echoed, and never written to install.json.
"""

from __future__ import annotations

import asyncio
import getpass
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from python_core import __version__
from python_core.core.config import EchoConfig
from python_core.lifecycle import (
    atomic_write_text,
    mark_setup_complete,
    merge_config_text,
)
from python_core.ui import Spinner, banner, color_enabled, paint

LLM_CANDIDATES = (
    ("Ollama", "http://127.0.0.1:11434/v1"),
    ("LM Studio", "http://127.0.0.1:1234/v1"),
    ("llama.cpp", "http://127.0.0.1:8080/v1"),
)


class SetupAborted(Exception):
    """Raised on Ctrl-C / EOF so no config is written."""


@dataclass
class SetupOptions:
    """CLI flags. ``None`` means 'not supplied' (prompt in interactive mode)."""

    reset: bool = False
    quick: bool = False
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: str | None = None
    telegram_token: str | None = None
    discord_token: str | None = None
    data_dir: str | None = None
    port: int | None = None
    yes: bool = False
    interactive: bool | None = None
    stream: TextIO | None = None


def _is_interactive(options: SetupOptions) -> bool:
    if options.interactive is not None:
        return options.interactive
    try:
        return bool(sys.stdin.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def _out(options: SetupOptions) -> TextIO:
    return options.stream or sys.stdout


def _say(options: SetupOptions, text: str = "") -> None:
    print(text, file=_out(options), flush=True)


def _ask(prompt: str, default: str = "", *, hidden: bool = False) -> str:
    try:
        raw = getpass.getpass(prompt) if hidden else input(prompt)
    except (EOFError, KeyboardInterrupt) as exc:
        raise SetupAborted from exc
    return (raw or "").strip() or default


def _choose(options: SetupOptions, title: str, choices: list[str], default: int = 1) -> int:
    _say(options, f"\n{title}")
    for index, label in enumerate(choices, 1):
        marker = " (default)" if index == default else ""
        _say(options, f"  {index}. {label}{marker}")
    raw = _ask("choice: ", str(default))
    if raw.isdigit() and 1 <= int(raw) <= len(choices):
        return int(raw)
    return default


# ── probes (monkeypatched in tests; no network at import time) ───────────────
def probe_llm(base_url: str, *, timeout: float = 1.0) -> list[str] | None:
    """Return model ids from an OpenAI-compatible endpoint, or None if down."""
    import httpx  # core dependency; imported lazily to keep wizard import light

    url = base_url.rstrip("/") + "/models"
    try:
        resp = httpx.get(url, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        return [str(item.get("id")) for item in payload.get("data", []) if item.get("id")]
    except Exception:  # noqa: BLE001 — an endpoint may be absent or non-OpenAI
        return None


def validate_telegram(token: str, *, timeout: float = 5.0) -> tuple[bool, str]:
    import httpx

    try:
        resp = httpx.post(
            f"https://api.telegram.org/bot{token}/getMe", timeout=timeout
        )
        data = resp.json()
        if resp.status_code == 200 and data.get("ok"):
            user = data.get("result") or {}
            return True, str(user.get("username") or user.get("id") or "bot")
        return False, str(data.get("description") or f"HTTP {resp.status_code}")
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def validate_discord(token: str, *, timeout: float = 5.0) -> tuple[bool, str]:
    import httpx

    try:
        resp = httpx.get(
            "https://discord.com/api/v10/users/@me",
            headers={"Authorization": f"Bot {token}"},
            timeout=timeout,
        )
        if resp.status_code == 200:
            return True, str(resp.json().get("username") or "bot")
        return False, f"HTTP {resp.status_code}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def _mask(token: str) -> str:
    return "set" if token else "not set"


# ── wizard flow ──────────────────────────────────────────────────────────────
def _select_llm(options: SetupOptions, interactive: bool, values: dict[str, Any]) -> None:
    _say(options, "\nStep 2 · LLM provider")
    if not interactive:
        values["llm"]["base_url"] = options.llm_base_url or values["llm"]["base_url"]
        values["llm"]["model"] = options.llm_model or values["llm"]["model"]
        if options.llm_api_key is not None:
            values["llm"]["api_key"] = options.llm_api_key
        return

    def _discover() -> list[tuple[str, str, list[str]]]:
        found: list[tuple[str, str, list[str]]] = []
        for name, url in LLM_CANDIDATES:
            models = probe_llm(url)
            if models is not None:
                found.append((name, url, models))
        return found

    _say(options, "  checking local endpoints (Ollama / LM Studio / llama.cpp)…")
    reachable = _with_spinner("probing local LLM endpoints", options, _discover)

    choices: list[str] = []
    for name, url, models in reachable:
        label = f"{name} — {url}"
        if models:
            label += f"  [{', '.join(models[:3])}]"
        choices.append(label)
    choices.append("Custom OpenAI-compatible endpoint")
    choices.append("Skip for now (agent still boots)")

    index = _choose(options, "Choose an LLM source:", choices, default=1 if reachable else len(choices))
    if index <= len(reachable):
        _name, url, models = reachable[index - 1]
        values["llm"]["base_url"] = url
        values["llm"]["model"] = models[0] if models else _ask("model name: ", values["llm"]["model"])
    elif index == len(choices) - 1:
        values["llm"]["base_url"] = _ask("base URL: ", values["llm"]["base_url"])
        values["llm"]["model"] = _ask("model name: ", values["llm"]["model"])
        values["llm"]["api_key"] = _ask("API key (optional): ", values["llm"]["api_key"], hidden=True)
    else:
        _say(options, "  skipping LLM setup — you can configure it later in config.toml")


def _with_spinner(label: str, options: SetupOptions, fn: Any) -> Any:
    """Run a blocking probe in a thread while the Spinner animates (TTY only)."""

    async def _run() -> Any:
        spinner = Spinner(label, stream=_out(options))
        await spinner.start()
        try:
            return await asyncio.to_thread(fn)
        finally:
            await spinner.stop()

    return asyncio.run(_run())


def _validate_token(options: SetupOptions, platform: str, token: str) -> bool:
    if not token:
        return False
    validator = validate_telegram if platform == "telegram" else validate_discord
    ok, detail = _with_spinner(f"validating {platform} token", options, lambda: validator(token))
    if ok:
        _say(options, f"  ✓ {platform} bot: @{detail}")
    else:
        _say(options, f"  ! {platform} token could not be verified: {detail}")
        _say(options, "    it will be saved anyway; check `echo-agent gateway test " + platform + "` later")
    return ok


def _select_channels(options: SetupOptions, interactive: bool, values: dict[str, Any]) -> None:
    _say(options, "\nStep 3 · Chat channels (optional)")
    for platform in ("telegram", "discord"):
        attr = f"{platform}_token"
        flag_value = getattr(options, attr)
        token = flag_value if flag_value is not None else (
            _ask(f"  {platform.title()} bot token (Enter to skip): ", "", hidden=True)
            if interactive else ""
        )
        if token:
            values[platform]["token"] = token
            _validate_token(options, platform, token)
        if platform == "discord" and token:
            _say(options, "  note: Discord needs the `discord` extra and Message Content intent")
            _say(options, "        install: uv tool install --force 'echo-agent[discord] @ git+…'")


def _security(options: SetupOptions, interactive: bool, values: dict[str, Any]) -> None:
    _say(options, "\nStep 4 · Security")
    _say(options, "  DMs: pairing (unknown senders get a code you approve from the CLI)")
    _say(options, "  Groups: allowlist (empty list = closed)")
    if not interactive or options.quick:
        return
    if _choose(options, "Change access rules?", ["Keep safe defaults", "Set explicit allowlists"], 1) == 2:
        for platform in ("telegram", "discord"):
            ids = _ask(f"  {platform.title()} allowFrom IDs (comma-separated, Enter to skip): ")
            if ids:
                values[platform]["allow_from"] = [x.strip() for x in ids.split(",") if x.strip()]
                values[platform]["dm_policy"] = "allowlist"


def _advanced(options: SetupOptions, interactive: bool, values: dict[str, Any], config: EchoConfig) -> None:
    _say(options, "\nStep 5 · Advanced")
    if not interactive or options.quick:
        if options.port is not None:
            values["manifest_port"] = options.port
        if options.data_dir:
            values["data_dir"] = options.data_dir
        return
    if _choose(options, "Configure advanced settings?", ["Use defaults", "Customize"], 1) == 1:
        return
    data_dir = _ask("data directory: ", str(config.data_dir))
    values["data_dir"] = data_dir
    port = _ask("manifest port: ", str(options.port or config.manifest_port))
    values["manifest_port"] = int(port) if port.isdigit() else config.manifest_port
    budget = _ask("context token budget: ", str(config.max_context_tokens))
    values["tokens"]["max_context_tokens"] = int(budget) if budget.isdigit() else config.max_context_tokens
    allowed = _ask("web allowed domains (comma-separated, Enter = all): ")
    blocked = _ask("web blocked domains (comma-separated): ")
    if allowed:
        values["web"]["allowed_domains"] = [x.strip() for x in allowed.split(",") if x.strip()]
    if blocked:
        values["web"]["blocked_domains"] = [x.strip() for x in blocked.split(",") if x.strip()]


def _summary(options: SetupOptions, values: dict[str, Any], config_path: Path) -> None:
    _say(options, "\n" + paint("✓ Setup complete", "32", enabled=color_enabled(_out(options))))
    _say(options, f"  config:      {config_path}")
    _say(options, f"  LLM:         {values['llm']['base_url']} ({_mask(values['llm']['api_key'])})")
    _say(options, f"  Telegram:    {_mask(values['telegram']['token'])}")
    _say(options, f"  Discord:     {_mask(values['discord']['token'])}")
    _say(options, "  DM policy:   pairing   ·   group policy: allowlist")
    _say(options, "\nNext steps")
    _say(options, "  echo-agent agent                 start the agent")
    _say(options, "  echo-agent gateway start         run it in the background")
    _say(options, "  echo-agent pairing list telegram approve a DM code")


def run_setup(config: EchoConfig, options: SetupOptions | None = None) -> int:
    """Run the wizard. Returns a process exit code; never raises on user input."""
    options = options or SetupOptions()
    interactive = _is_interactive(options)
    stream = _out(options)
    try:
        _say(options, banner(__version__, stream=stream))
        _say(options, "\nStep 1 · Welcome")

        config_path = Path(options.data_dir).expanduser() / "config.toml" if options.data_dir else (
            Path(os.environ.get("ECHO_DATA_DIR", config.data_dir)).expanduser() / "config.toml"
        )
        existing = config_path.read_text(encoding="utf-8") if config_path.is_file() else ""
        if existing and not options.reset and interactive and not options.quick:
            _say(options, "  existing configuration found: " + str(config_path))
            choice = _choose(
                options,
                "What would you like to do?",
                ["Keep existing setup", "Reconfigure (merge)", "Reset to defaults"],
                1,
            )
            if choice == 1:
                mark_setup_complete(config.data_dir, __version__)
                _say(options, "  keeping existing configuration")
                return 0
            if choice == 3:
                options.reset = True
                existing = ""

        values: dict[str, Any] = {
            "llm": {
                "base_url": options.llm_base_url or ("http://127.0.0.1:11434/v1" if options.reset else config.llm_base_url),
                "model": options.llm_model or ("llama3.1" if options.reset else config.llm_model),
                "api_key": options.llm_api_key if options.llm_api_key is not None else ("" if options.reset else config.llm_api_key),
            },
            "telegram": {
                "token": options.telegram_token if options.telegram_token is not None else ("" if options.reset else config.telegram_token),
                "dm_policy": "pairing",
                "group_policy": "allowlist",
            },
            "discord": {
                "token": options.discord_token if options.discord_token is not None else ("" if options.reset else config.discord_token),
                "dm_policy": "pairing",
                "group_policy": "allowlist",
            },
            "tokens": {"max_context_tokens": config.max_context_tokens},
            "web": {"allowed_domains": list(config.web_allowed_domains), "blocked_domains": list(config.web_blocked_domains)},
        }

        _say(options, "  mode: " + ("QuickStart (safe defaults)" if options.quick or not interactive else "Advanced"))
        _select_llm(options, interactive and not options.quick, values)
        _select_channels(options, interactive, values)
        _security(options, interactive, values)
        _advanced(options, interactive, values, config)

        # Stage every field, then commit once.
        updates = dict(values)
        if options.data_dir:
            updates["data_dir"] = options.data_dir
        merged = merge_config_text("" if options.reset else existing, updates)
        target = Path(options.data_dir).expanduser() / "config.toml" if options.data_dir else config_path
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(target, merged, mode=0o600)
        config.data_dir = target.parent
        config.ensure_dirs()
        marker = mark_setup_complete(config.data_dir, __version__)
        _summary(options, values, target)
        _say(options, f"  marker:      {marker}")
        return 0
    except SetupAborted:
        _say(options, "\nsetup cancelled — no changes were written")
        return 130
    except KeyboardInterrupt:
        _say(options, "\nsetup cancelled — no changes were written")
        return 130
