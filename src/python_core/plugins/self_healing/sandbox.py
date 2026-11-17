"""Isolated execution environment for self-generated code (MODULE 2).

The golden rule of safe self-modification: **the live system never executes a
patch**. Every candidate change is materialized inside a throwaway sandbox
tree (a real copy of the project, minus caches/VCS), validated there with
pytest, and only then considered for atomic application.

Design:
    * ``create()`` snapshots a project into a temp directory (deterministic
      ignore list: .git, __pycache__, .venv, target/, node_modules, *.pyc).
    * ``run_pytest()`` executes the test suite with the *sandbox* root first
      on ``PYTHONPATH`` — imports resolve to the patched code, not the live one.
    * Everything runs in a subprocess with a hard timeout; a hung test suite
      is treated exactly like a failing one.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("echo.sandbox")

# Never copy these into a sandbox — huge, stateful, or irrelevant to tests.
DEFAULT_IGNORES = (
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".venv", "venv",
    "node_modules", "target", "dist", "build", ".mypy_cache", ".ruff_cache",
    "*.pyc", "*.sqlite3", "*.sqlite3-wal", "*.sqlite3-shm", ".echo_agent",
)


@dataclass(slots=True)
class SandboxResult:
    """Outcome of one sandboxed command, normalized for episodic logging."""

    ok: bool
    returncode: int
    stdout: str
    stderr: str
    elapsed_ms: int
    cmd: list[str] = field(default_factory=list)

    def tail(self, chars: int = 2000) -> str:
        """Compact failure evidence (keeps episodic memory lean)."""
        blob = (self.stdout + "\n--- stderr ---\n" + self.stderr).strip()
        return blob[-chars:] if len(blob) > chars else blob


class Sandbox:
    """Factory for isolated project copies + subprocess runner."""

    def __init__(self, base_dir: Path | None = None, *, ignores: tuple[str, ...] = DEFAULT_IGNORES) -> None:
        self.base_dir = base_dir  # None → tempfile default location
        self.ignores = ignores

    # ── sandbox construction ───────────────────────────────────────────────
    def create(
        self,
        source_tree: Path | None = None,
        *,
        files: dict[str, str] | None = None,
        prefix: str = "echo-sandbox-",
    ) -> Path:
        """Materialize a sandbox: copy *source_tree* (respecting ignores) and
        overlay *files* ({relative_path: content}). Returns the sandbox root.
        """
        root = Path(tempfile.mkdtemp(prefix=prefix, dir=self.base_dir))
        if source_tree is not None and source_tree.is_dir():
            shutil.copytree(
                source_tree,
                root,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(*self.ignores),
            )
        for rel_path, content in (files or {}).items():
            target = root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return root

    def write_file(self, root: Path, rel_path: str, content: str) -> Path:
        target = root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    # ── execution ──────────────────────────────────────────────────────────
    async def run(
        self,
        cmd: list[str],
        *,
        cwd: Path | None = None,
        timeout: float = 300.0,
        extra_env: dict[str, str] | None = None,
    ) -> SandboxResult:
        """Run a command in a subprocess. Never raises on failure — a result
        with ``ok=False`` is data the agent can learn from (episodic memory)."""
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        started = time.monotonic()
        try:
            proc = await asyncio_subprocess(cmd, cwd=str(cwd) if cwd else None, env=env, timeout=timeout)
            returncode, out, err = proc
        except subprocess.TimeoutExpired:
            return SandboxResult(
                ok=False, returncode=-1, stdout="", stderr=f"timeout after {timeout:.0f}s",
                elapsed_ms=int((time.monotonic() - started) * 1000), cmd=cmd,
            )
        except FileNotFoundError as exc:
            return SandboxResult(
                ok=False, returncode=-1, stdout="", stderr=f"executable not found: {exc}",
                elapsed_ms=int((time.monotonic() - started) * 1000), cmd=cmd,
            )
        return SandboxResult(
            ok=returncode == 0,
            returncode=returncode,
            stdout=out[-20000:],  # cap captured output
            stderr=err[-8000:],
            elapsed_ms=int((time.monotonic() - started) * 1000),
            cmd=cmd,
        )

    async def run_pytest(
        self,
        root: Path,
        *,
        test_path: str | None = None,
        timeout: float = 300.0,
    ) -> SandboxResult:
        """Run the sandbox's test suite with sandbox-first import resolution."""
        cmd = [sys.executable, "-m", "pytest", "-x", "-q", "--tb=short", "-p", "no:cacheprovider"]
        if test_path:
            cmd.append(test_path)
        # Sandbox root first: patched modules shadow the live installation.
        env = {"PYTHONPATH": str(root) + os.pathsep + env_pythonpath()}
        return await self.run(cmd, cwd=root, timeout=timeout, extra_env=env)

    # ── cleanup ────────────────────────────────────────────────────────────
    def cleanup(self, root: Path) -> None:
        shutil.rmtree(root, ignore_errors=True)


# ── helpers ──────────────────────────────────────────────────────────────────
def env_pythonpath() -> str:
    """Current PYTHONPATH (may be empty)."""
    return os.environ.get("PYTHONPATH", "")


async def asyncio_subprocess(
    cmd: list[str], *, cwd: str | None, env: dict[str, str], timeout: float
) -> tuple[int, str, str]:
    """Async subprocess with captured output and a hard timeout."""
    import asyncio

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise subprocess.TimeoutExpired(cmd, timeout) from None
    return (
        proc.returncode if proc.returncode is not None else -1,
        out_b.decode("utf-8", errors="replace"),
        err_b.decode("utf-8", errors="replace"),
    )


def project_copyable(source: Path) -> bool:
    """Guard: only real directories with some code get sandboxed."""
    return source.is_dir() and any(source.glob("**/*.py"))
