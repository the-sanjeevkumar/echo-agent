"""Uninstaller: cleanup order, safety rails, dry-run, keep-data, idempotency."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from python_core.lifecycle import has_marker_block, write_json
from python_core.uninstall import Uninstaller, UnsafePathError


def _fixture(tmp_path: Path):
    home = tmp_path / "home"
    data = home / ".echo_agent"
    data.mkdir(parents=True)
    write_json(data / "install.json", {"method": "uv-tool", "version": "0.2.0"})
    (data / "echo-agent.pid").write_text("999999", encoding="utf-8")
    rc = home / ".bashrc"
    rc.write_text(
        "export FOO=1\n"
        "# >>> echo-agent >>>\n"
        'export PATH="$HOME/.local/bin:$PATH"\n'
        "# <<< echo-agent <<<\n"
        "export BAR=2\n",
        encoding="utf-8",
    )
    bin_dir = home / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "echo-agent").write_text("shim", encoding="utf-8")
    return home, data, rc, bin_dir


def _patch(monkeypatch: pytest.MonkeyPatch, calls: list[list[str]]) -> None:
    monkeypatch.setattr(
        "python_core.uninstall.shutil.which",
        lambda name: "/usr/bin/uv" if name == "uv" else None,
    )

    def fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("python_core.uninstall.subprocess.run", fake_run)
    monkeypatch.setattr("python_core.uninstall.time.sleep", lambda _: None)


def test_uninstall_order_and_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home, data, rc, bin_dir = _fixture(tmp_path)
    calls: list[list[str]] = []
    _patch(monkeypatch, calls)

    uninstaller = Uninstaller(data_dir=data, home=home, yes=True)
    rc_code = uninstaller.run()
    assert rc_code == 0

    assert calls[0][1:3] == ["tool", "dir"]
    assert calls[1][1:3] == ["tool", "uninstall"]
    assert calls[2][1:3] == ["cache", "clean"]
    assert not data.exists()
    assert not (bin_dir / "echo-agent").exists()
    text = rc.read_text()
    assert not has_marker_block(text)
    assert "export FOO=1" in text and "export BAR=2" in text


def test_keep_data_and_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    home, data, rc, _bin = _fixture(tmp_path)
    calls: list[list[str]] = []
    _patch(monkeypatch, calls)

    result = Uninstaller(data_dir=data, home=home, yes=True, keep_data=True)
    assert result.run() == 0
    assert data.exists()
    assert not has_marker_block(rc.read_text())

    home2, data2, rc2, _ = _fixture(tmp_path / "dry")
    calls.clear()
    result = Uninstaller(data_dir=data2, home=home2, yes=True, dry_run=True)
    assert result.run() == 0
    assert all(call[1:3] == ["tool", "dir"] for call in calls)  # read-only discovery only
    assert data2.exists()
    assert has_marker_block(rc2.read_text())
    out = capsys.readouterr().out
    assert "uv tool uninstall echo-agent" in out
    assert str(data2) in out


def test_safety_refusals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home, data, _rc, _bin = _fixture(tmp_path)
    _patch(monkeypatch, [])

    with pytest.raises(UnsafePathError, match=r"\$HOME"):
        Uninstaller(data_dir=home, home=home).validate()
    with pytest.raises(UnsafePathError, match=r"HOME"):
        Uninstaller(data_dir=data, home="/").validate()
    with pytest.raises(UnsafePathError, match=r"'/'"):
        Uninstaller(data_dir="/", home=home).validate()

    outside = tmp_path / "outside"
    outside.mkdir()
    link = home / "escape"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(UnsafePathError, match="outside"):
        Uninstaller(data_dir=link, home=home).validate()


def test_second_run_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    home, data, _rc, _bin = _fixture(tmp_path)
    calls: list[list[str]] = []
    _patch(monkeypatch, calls)

    assert Uninstaller(data_dir=data, home=home, yes=True).run() == 0
    capsys.readouterr()
    assert Uninstaller(data_dir=data, home=home, yes=True).run() == 0
    assert "nothing to do" in capsys.readouterr().out


def test_non_tty_requires_yes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home, data, _rc, _bin = _fixture(tmp_path)
    _patch(monkeypatch, [])
    monkeypatch.setattr("python_core.uninstall.sys.stdin.isatty", lambda: False)
    assert Uninstaller(data_dir=data, home=home, yes=False).run() == 2
    assert data.exists()
