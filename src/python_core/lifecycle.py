"""Shared lifecycle paths, metadata, and config helpers.

Used by the setup wizard, doctor, and uninstaller. Standard library only, so
these helpers stay importable even when optional runtime deps are missing.

On-disk contract (all under one data root, default ``~/.echo_agent``)::

    install.json        written by install.sh: method/extras/version/repo
    .setup-complete     written by the wizard: version/timestamp
    config.toml         merged by the wizard, read by EchoConfig.load
    browsers-owned.json written when the user opts into `scrapling install`
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tomllib

REPO_URL = "https://github.com/the-sanjeevkumar/echo-agent.git"

RC_MARKER_START = "# >>> echo-agent >>>"
RC_MARKER_END = "# <<< echo-agent <<<"

INSTALL_META = "install.json"
SETUP_MARKER = ".setup-complete"
BROWSERS_OWNED = "browsers-owned.json"


# ── paths ────────────────────────────────────────────────────────────────────
def default_data_dir() -> Path:
    """Data root from ``ECHO_DATA_DIR`` or ``~/.echo_agent``."""
    env = os.environ.get("ECHO_DATA_DIR")
    return Path(env).expanduser() if env else Path.home() / ".echo_agent"


def home_dir() -> Path:
    return Path(os.environ.get("HOME") or Path.home()).expanduser()


def setup_marker_path(data_dir: Path | None = None) -> Path:
    return (data_dir or default_data_dir()) / SETUP_MARKER


def install_metadata_path(data_dir: Path | None = None) -> Path:
    return (data_dir or default_data_dir()) / INSTALL_META


def browsers_owned_path(data_dir: Path | None = None) -> Path:
    return (data_dir or default_data_dir()) / BROWSERS_OWNED


def resolve_data_dir(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the data root used by an uninstall/doctor run.

    Priority: explicit argument > ``ECHO_DATA_DIR`` > a top-level ``data_dir``
    in the default ``config.toml`` > ``data_dir`` in ``install.json`` > default.
    """
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("ECHO_DATA_DIR")
    if env:
        return Path(env).expanduser()
    default = default_data_dir()
    for reader in (_data_dir_from_config, _data_dir_from_metadata):
        found = reader(default)
        if found:
            return found
    return default


def _data_dir_from_config(default: Path) -> Path | None:
    try:
        with open(default / "config.toml", "rb") as fh:
            raw = tomllib.load(fh)
    except (OSError, ValueError):
        return None
    value = raw.get("data_dir")
    return Path(str(value)).expanduser() if value else None


def _data_dir_from_metadata(default: Path) -> Path | None:
    meta = read_json(default / INSTALL_META)
    value = meta.get("data_dir") if meta else None
    return Path(str(value)).expanduser() if value else None


# ── small JSON helpers ───────────────────────────────────────────────────────
def read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def atomic_write_text(path: Path, text: str, *, mode: int = 0o600) -> None:
    """Write *text* atomically with restrictive permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def write_json(path: Path, payload: dict[str, Any], *, mode: int = 0o600) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n", mode=mode)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── setup / install metadata ─────────────────────────────────────────────────
def is_setup_complete(data_dir: Path | None = None) -> bool:
    marker = read_json(setup_marker_path(data_dir))
    return bool(marker and marker.get("version"))


def mark_setup_complete(data_dir: Path | None = None, version: str = "0.2.0") -> Path:
    path = setup_marker_path(data_dir)
    write_json(path, {"version": version, "completed_at": utc_now()})
    return path


def read_install_metadata(data_dir: Path | None = None) -> dict[str, Any]:
    return read_json(install_metadata_path(data_dir)) or {}


# ── TOML merge (stdlib only; keeps unknown keys) ─────────────────────────────
def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return json.dumps(str(value), ensure_ascii=False)


def dump_toml(data: dict[str, Any]) -> str:
    """Serialize a flat/two-level config mapping to readable TOML."""
    lines: list[str] = []
    scalars = {k: v for k, v in data.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in data.items() if isinstance(v, dict)}
    for key, value in scalars.items():
        lines.append(f"{key} = {_toml_value(value)}")
    for name, table in tables.items():
        if lines:
            lines.append("")
        lines.append(f"[{name}]")
        for key, value in table.items():
            if isinstance(value, dict):
                continue  # deeper nesting is not part of echo-agent's schema
            lines.append(f"{key} = {_toml_value(value)}")
        for key, value in table.items():
            if isinstance(value, dict):
                lines.append("")
                lines.append(f"[{name}.{key}]")
                for sub_key, sub_value in value.items():
                    lines.append(f"{sub_key} = {_toml_value(sub_value)}")
    return "\n".join(lines).rstrip() + "\n"


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def merge_config_text(existing: str, updates: dict[str, Any]) -> str:
    """Merge *updates* into existing TOML text, preserving unknown keys."""
    try:
        base = tomllib.loads(existing) if existing.strip() else {}
    except ValueError:
        base = {}
    return dump_toml(_deep_merge(base, updates))


# ── rc marker blocks ─────────────────────────────────────────────────────────
def strip_marker_block(text: str) -> str:
    """Remove exactly one ``# >>> echo-agent >>>`` block, keeping other lines."""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    inside = False
    for line in lines:
        stripped = line.strip()
        if stripped == RC_MARKER_START:
            inside = True
            continue
        if stripped == RC_MARKER_END:
            inside = False
            continue
        if not inside:
            out.append(line)
    return "".join(out)


def has_marker_block(text: str) -> bool:
    return RC_MARKER_START in text and RC_MARKER_END in text
