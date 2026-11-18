"""Setup wizard: non-interactive paths, config merge, markers, abort safety."""

from __future__ import annotations

import builtins
import io
from pathlib import Path

import pytest
import tomllib

from python_core.core.config import EchoConfig
from python_core.lifecycle import is_setup_complete
from python_core.setup_wizard import (
    SetupOptions,
    probe_llm,
    run_setup,
    validate_discord,
    validate_telegram,
)


def _options(tmp_path: Path, **kwargs: object) -> SetupOptions:
    defaults: dict[str, object] = {
        "interactive": False,
        "stream": io.StringIO(),
        "data_dir": str(tmp_path / "data"),
        "llm_base_url": "http://example.test/v1",
        "llm_model": "fake-model",
    }
    defaults.update(kwargs)
    return SetupOptions(**defaults)  # type: ignore[arg-type]


def test_quick_setup_writes_loadable_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ECHO_DATA_DIR", str(tmp_path / "data"))
    stream = io.StringIO()
    rc = run_setup(EchoConfig(data_dir=tmp_path / "data"), _options(tmp_path, stream=stream))
    assert rc == 0

    config_path = tmp_path / "data" / "config.toml"
    assert config_path.is_file()
    assert config_path.stat().st_mode & 0o777 == 0o600
    raw = tomllib.loads(config_path.read_text())
    assert raw["llm"]["base_url"] == "http://example.test/v1"
    assert raw["llm"]["model"] == "fake-model"

    loaded = EchoConfig.load()
    assert loaded.llm_base_url == "http://example.test/v1"
    assert loaded.telegram_dm_policy == "pairing"
    assert is_setup_complete(tmp_path / "data")
    assert "Setup complete" in stream.getvalue()


def test_non_interactive_flags_and_validated_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ECHO_DATA_DIR", str(tmp_path / "data"))
    calls: list[str] = []

    def fake_telegram(token: str, *, timeout: float = 5.0) -> tuple[bool, str]:
        calls.append(token)
        return True, "fakebot"

    monkeypatch.setattr("python_core.setup_wizard.validate_telegram", fake_telegram)
    monkeypatch.setattr("python_core.setup_wizard.validate_discord",
                        lambda token, **kw: (False, "no network"))
    rc = run_setup(
        EchoConfig(data_dir=tmp_path / "data"),
        _options(
            tmp_path,
            telegram_token="123:fake",
            discord_token="discord-secret",
            port=9999,
            llm_api_key="sk-123",
        ),
    )
    assert rc == 0
    assert calls == ["123:fake"]
    raw = tomllib.loads((tmp_path / "data" / "config.toml").read_text())
    assert raw["telegram"]["token"] == "123:fake"
    assert raw["discord"]["token"] == "discord-secret"
    assert raw["llm"]["api_key"] == "sk-123"
    assert raw["manifest_port"] == 9999
    text = (tmp_path / "data" / "config.toml").read_text()
    assert "sk-123" in text  # stored in config only, never in metadata
    assert not (tmp_path / "data" / "install.json").exists()


def test_reset_drops_unknown_keys_and_keeps_marker(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "config.toml").write_text('stale_key = 1\n\n[llm]\nmodel = "old"\n')
    rc = run_setup(EchoConfig(data_dir=data), _options(tmp_path, reset=True))
    assert rc == 0
    raw = tomllib.loads((data / "config.toml").read_text())
    assert "stale_key" not in raw
    assert raw["llm"]["model"] == "fake-model"
    assert is_setup_complete(data)


def test_probe_functions_degrade_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "ok": True,
                "username": "fakebot",
                "result": {"username": "fakebot"},
                "data": [{"id": "m1"}],
            }

    monkeypatch.setattr("httpx.get", lambda *a, **kw: FakeResponse())
    monkeypatch.setattr("httpx.post", lambda *a, **kw: FakeResponse())
    assert probe_llm("http://x/v1") == ["m1"]
    assert validate_telegram("token") == (True, "fakebot")
    assert validate_discord("token") == (True, "fakebot")


def test_ctrl_c_aborts_without_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("python_core.setup_wizard.probe_llm", lambda url, **kw: [])
    monkeypatch.setattr(builtins, "input", lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt()))
    rc = run_setup(
        EchoConfig(data_dir=tmp_path / "data"),
        _options(tmp_path, interactive=True, quick=False, llm_base_url=None),
    )
    assert rc == 130
    assert not (tmp_path / "data" / "config.toml").exists()


def test_first_run_hook_non_tty_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    from python_core import main as cli_module

    config = EchoConfig(data_dir=tmp_path / "data")
    monkeypatch.setattr(EchoConfig, "load", classmethod(lambda cls, *a, **kw: config))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    async def fake_run(*a: object, **kw: object) -> int:
        return 0

    monkeypatch.setattr(cli_module, "_cmd_run", fake_run)

    def forbidden(*a: object, **kw: object) -> int:
        raise AssertionError("wizard must not run in non-TTY first-run")

    monkeypatch.setattr(cli_module, "run_setup", forbidden)
    assert cli_module.cli(["run", "hello"]) == 0
    assert "hint: run `echo-agent setup`" in capsys.readouterr().out
