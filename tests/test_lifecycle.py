"""Lifecycle helpers: config merge, markers, atomic writes, data-dir resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
import tomllib

from python_core.lifecycle import (
    atomic_write_text,
    default_data_dir,
    has_marker_block,
    is_setup_complete,
    mark_setup_complete,
    merge_config_text,
    read_json,
    resolve_data_dir,
    strip_marker_block,
    write_json,
)


def test_merge_preserves_unknown_keys(tmp_path: Path) -> None:
    existing = (
        'manifest_port = 9999\n\n[llm]\nbase_url = "old"\ncustom_key = "keep"\n\n'
        '[mystery]\nfoo = 42\n'
    )
    merged = merge_config_text(existing, {"llm": {"base_url": "new", "model": "m"}})
    data = tomllib.loads(merged)
    assert data["manifest_port"] == 9999
    assert data["mystery"]["foo"] == 42
    assert data["llm"]["custom_key"] == "keep"
    assert data["llm"]["base_url"] == "new"
    assert data["llm"]["model"] == "m"


def test_merge_handles_empty_and_invalid_text() -> None:
    assert tomllib.loads(merge_config_text("", {"llm": {"model": "x"}}))["llm"]["model"] == "x"
    assert tomllib.loads(merge_config_text("not valid toml {", {"llm": {"model": "y"}}))["llm"]["model"] == "y"


def test_atomic_write_and_json_modes(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    atomic_write_text(path, "x = 1\n")
    assert path.read_text() == "x = 1\n"
    assert path.stat().st_mode & 0o777 == 0o600
    meta = tmp_path / "meta.json"
    write_json(meta, {"method": "uv-tool"})
    assert read_json(meta) == {"method": "uv-tool"}
    assert meta.stat().st_mode & 0o777 == 0o600


def test_setup_marker_roundtrip(tmp_path: Path) -> None:
    assert is_setup_complete(tmp_path) is False
    mark_setup_complete(tmp_path, "9.9.9")
    assert is_setup_complete(tmp_path) is True
    marker = (tmp_path / ".setup-complete").read_text()
    assert '"version": "9.9.9"' in marker


def test_strip_marker_block_keeps_surroundings() -> None:
    text = (
        "export FOO=1\n"
        "# >>> echo-agent >>>\n"
        'export PATH="$HOME/.local/bin:$PATH"\n'
        "# <<< echo-agent <<<\n"
        "export BAR=2\n"
    )
    cleaned = strip_marker_block(text)
    assert has_marker_block(text)
    assert not has_marker_block(cleaned)
    assert "export FOO=1" in cleaned
    assert "export BAR=2" in cleaned
    assert "echo-agent" not in cleaned
    assert strip_marker_block(cleaned) == cleaned


def test_resolve_data_dir_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ECHO_DATA_DIR", str(tmp_path / "from-env"))
    assert resolve_data_dir() == tmp_path / "from-env"
    assert resolve_data_dir(tmp_path / "explicit") == tmp_path / "explicit"
    monkeypatch.delenv("ECHO_DATA_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert default_data_dir() == tmp_path / ".echo_agent"

    (tmp_path / ".echo_agent").mkdir()
    write_json(tmp_path / ".echo_agent" / "install.json", {"data_dir": str(tmp_path / "from-meta")})
    assert resolve_data_dir() == tmp_path / "from-meta"
