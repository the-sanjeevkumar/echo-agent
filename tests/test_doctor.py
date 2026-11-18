"""Doctor: read-only checks, hints, and exit codes (no real network)."""

from __future__ import annotations

import io
from pathlib import Path

from python_core import __version__
from python_core.core.config import EchoConfig
from python_core.doctor import run_doctor


def _config(tmp_path: Path) -> EchoConfig:
    return EchoConfig(data_dir=tmp_path / "data", llm_base_url="http://llm.test/v1")


def test_doctor_reports_hard_failure_for_missing_entry_point(tmp_path: Path) -> None:
    config = _config(tmp_path)
    stream = io.StringIO()
    # shutil.which may find the dev venv binary; force the missing case.
    from python_core import doctor

    original = doctor.shutil.which
    doctor.shutil.which = lambda name: None  # type: ignore[assignment]
    try:
        rc = run_doctor(config, stream=stream, metadata={})
    finally:
        doctor.shutil.which = original  # type: ignore[assignment]
    assert rc == 1
    assert "✗ entry point" in stream.getvalue()
    assert "hard failure" in stream.getvalue()


def test_doctor_clean_run(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    config.ensure_dirs()
    stream = io.StringIO()
    monkeypatch.setattr("python_core.doctor.shutil.which", lambda name: "/usr/bin/echo-agent")
    rc = run_doctor(
        config,
        stream=stream,
        metadata={"method": "uv-tool", "version": __version__},
    )
    assert rc == 0
    assert "✓ entry point" in stream.getvalue()
    assert "✓ data dir" in stream.getvalue()
    assert "no hard failures" in stream.getvalue()


def test_doctor_probe_reports_models(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("python_core.setup_wizard.probe_llm", lambda url, **kw: ["m1", "m2"])
    monkeypatch.setattr("python_core.setup_wizard.validate_telegram", lambda t, **kw: (True, "bot"))
    monkeypatch.setattr("python_core.setup_wizard.validate_discord", lambda t, **kw: (False, "bad"))
    config = _config(tmp_path)
    config.telegram_token = "tg"
    stream = io.StringIO()
    rc = run_doctor(config, probe=True, stream=stream, metadata={"version": __version__})
    assert rc == 0
    assert "2 models" in stream.getvalue()
    assert "✓ telegram token: bot" in stream.getvalue()
