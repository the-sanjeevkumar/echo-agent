"""Builtin tools (main.py): expected filesystem failures must be *data*, not
crashes — the model gets a structured, self-correcting error it can recover
from in the next ReAct step."""

from pathlib import Path

from python_core.core.config import EchoConfig
from python_core.core.tool_registry import ToolRegistry
from python_core.main import _register_builtin_tools


def _registry() -> ToolRegistry:
    r = ToolRegistry()
    _register_builtin_tools(r, EchoConfig.load())
    return r


async def test_fs_read_missing_file_returns_nearby_hints(tmp_path: Path) -> None:
    """The classic hallucinated-path case: the model asks for
    plugins/self_healing.py when the real file is plugins/self_healing/plugin.py."""
    r = _registry()
    (tmp_path / "plugins").mkdir()
    (tmp_path / "plugins" / "self_healing").mkdir()
    (tmp_path / "plugins" / "self_healing" / "plugin.py").write_text("x = 1")

    env = await r.execute("fs_read", {"path": str(tmp_path / "plugins" / "self_healing.py")})
    # Envelope semantics: ok = "the call executed"; the tool's own failure
    # status lives inside result.
    assert env["ok"] is True
    assert env["result"]["ok"] is False
    assert "file not found" in env["result"]["error"]
    assert "self_healing/" in env["result"]["nearby"]  # hint at the real layout

    # The hint is actionable: reading the real path succeeds immediately.
    env2 = await r.execute(
        "fs_read", {"path": str(tmp_path / "plugins" / "self_healing" / "plugin.py")}
    )
    assert env2["ok"] is True
    assert env2["result"]["content"] == "x = 1"


async def test_fs_read_edge_cases(tmp_path: Path) -> None:
    r = _registry()
    (tmp_path / "sub").mkdir()

    env = await r.execute("fs_read", {"path": str(tmp_path / "sub")})
    assert env["result"]["ok"] is False
    assert "not a regular file" in env["result"]["error"]

    env = await r.execute("fs_read", {"path": str(tmp_path / "ghost" / "f.txt")})
    assert env["result"]["ok"] is False
    assert "file not found" in env["result"]["error"]


async def test_fs_list_missing_dir(tmp_path: Path) -> None:
    r = _registry()
    env = await r.execute("fs_list", {"dir": str(tmp_path / "nope")})
    assert env["ok"] is True
    assert env["result"]["entries"] == []
    assert "directory not found" in env["result"]["error"]
