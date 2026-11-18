"""Total, safety-railed uninstaller for echo-agent.

Removes only known-owned artifacts: the daemon, the uv/pipx tool environment,
echo-created entry-point shims, the ``# >>> echo-agent >>>`` rc block, the data
root, and browser binaries explicitly recorded as ours. Shared caches are kept
unless an ownership marker exists. Standard library only.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from python_core.lifecycle import (
    browsers_owned_path,
    has_marker_block,
    read_install_metadata,
    read_json,
    resolve_data_dir,
    strip_marker_block,
)

SHIM_NAMES = ("echo-agent", "echo-uninstall")
RC_FILES = (".bashrc", ".zshrc", ".profile")


class UnsafePathError(RuntimeError):
    """Raised when a planned deletion target fails a safety rail."""


@dataclass
class Action:
    """One reportable uninstall step. ``run`` is None for informational lines."""

    label: str
    detail: str
    run: Callable[[], tuple[bool, str]] | None = None


class Uninstaller:
    def __init__(
        self,
        *,
        data_dir: str | os.PathLike[str] | None = None,
        yes: bool = False,
        keep_data: bool = False,
        dry_run: bool = False,
        verbose: bool = False,
        home: str | os.PathLike[str] | None = None,
    ) -> None:
        self.home = Path(home or os.environ.get("HOME") or Path.home()).expanduser()
        self.data_dir = Path(data_dir).expanduser() if data_dir else resolve_data_dir()
        self.yes = yes
        self.keep_data = keep_data
        self.dry_run = dry_run
        self.verbose = verbose
        self.metadata = read_install_metadata(self.data_dir)
        self._bin_dir = self._uv_bin_dir()
        shims_present = any((self._bin_dir / name).exists() for name in SHIM_NAMES)
        self._commands = self._program_commands() if (self.metadata or shims_present) else []
        self._browser_paths = self._owned_browser_paths()
        self._rc_files = [self.home / name for name in RC_FILES if (self.home / name).is_file()]
        self._tool_envs = self._tool_env_dirs()

    # ── discovery ───────────────────────────────────────────────────────────
    def _uv_bin_dir(self) -> Path:
        env = os.environ.get("UV_TOOL_BIN_DIR")
        return Path(env).expanduser() if env else self.home / ".local" / "bin"

    def _program_commands(self) -> list[list[str]]:
        """Tool-env removal command, then optional cache cleanup."""
        method = str(self.metadata.get("method") or "")
        uv = shutil.which("uv")
        pipx = shutil.which("pipx")
        if method == "pipx" and pipx:
            return [[pipx, "uninstall", "echo-agent"]]
        if uv:
            return [
                [uv, "tool", "uninstall", "echo-agent"],
                [uv, "cache", "clean", "echo-agent"],
            ]
        if pipx:
            return [[pipx, "uninstall", "echo-agent"]]
        return [[sys.executable, "-m", "pip", "uninstall", "-y", "echo-agent"]]

    def _owned_browser_paths(self) -> list[Path]:
        marker = read_json(browsers_owned_path(self.data_dir))
        if not marker:
            return []
        values = marker.get("paths")
        if not isinstance(values, list):
            return []
        return [Path(str(item)).expanduser() for item in values if str(item).strip()]

    def _tool_env_dirs(self) -> list[Path]:
        candidates = [self.home / ".local" / "share" / "uv" / "tools" / "echo-agent"]
        uv = shutil.which("uv")
        if uv:
            try:
                proc = subprocess.run(
                    [uv, "tool", "dir"], capture_output=True, text=True, timeout=10, check=False
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    candidates.append(Path(proc.stdout.strip().splitlines()[0]).expanduser() / "echo-agent")
            except (OSError, ValueError):
                pass
        return candidates

    # ── safety rails ────────────────────────────────────────────────────────
    def _under_home(self, path: Path) -> bool:
        try:
            return path.resolve().is_relative_to(self.home.resolve())
        except (OSError, RuntimeError, ValueError):
            return False

    def _uv_owned(self, path: Path) -> bool:
        """uv's documented tool dirs are allowed even outside $HOME."""
        resolved = path.resolve()
        uv_dirs = [self._bin_dir.resolve()]
        for env_dir in self._tool_envs:
            uv_dirs.append(env_dir.parent.resolve())
        try:
            return any(resolved.is_relative_to(parent) for parent in uv_dirs)
        except (OSError, RuntimeError, ValueError):
            return False

    def validate(self) -> None:
        home_text = str(self.home)
        if not home_text or home_text.strip() in {"", "/"}:
            raise UnsafePathError("$HOME is empty or '/' — refusing to delete anything")
        resolved_home = self.home.resolve()
        if resolved_home == Path(resolved_home.anchor):
            raise UnsafePathError("$HOME resolves to the filesystem root — refusing")
        resolved_data = self.data_dir.resolve()
        if resolved_data == resolved_home:
            raise UnsafePathError("data dir resolves to $HOME — refusing")
        if resolved_data == Path(resolved_data.anchor):
            raise UnsafePathError("data dir resolves to '/' — refusing")
        if not self._under_home(self.data_dir) and not self._uv_owned(self.data_dir):
            raise UnsafePathError(
                f"data dir {self.data_dir} is outside $HOME — refusing (symlink escape?)"
            )
        for path in self._browser_paths + self._tool_envs:
            if not (self._under_home(path) or self._uv_owned(path)):
                raise UnsafePathError(f"refusing to touch path outside $HOME: {path}")

    # ── planning ────────────────────────────────────────────────────────────
    def plan(self) -> list[Action]:
        actions: list[Action] = []

        pidfile = self.data_dir / "echo-agent.pid"
        if pidfile.is_file():
            actions.append(Action("daemon", f"stop daemon via {pidfile}", self._stop_daemon))
        else:
            actions.append(Action("daemon", f"no pidfile at {pidfile}"))

        for cmd in self._commands:
            text = " ".join(cmd)
            if cmd[1:3] == ["cache", "clean"]:
                actions.append(Action("dependencies", f"best-effort: {text}", lambda c=cmd: self._run(c)))
            else:
                actions.append(Action("program", text, lambda c=cmd: self._run(c)))

        if self._browser_paths:
            for path in self._browser_paths:
                actions.append(Action("browsers", str(path), lambda p=path: self._remove_path(p)))
        else:
            actions.append(Action("browsers", "shared browser caches: kept (no ownership marker)"))

        for rc in self._rc_files:
            if has_marker_block(rc.read_text(encoding="utf-8", errors="replace")):
                actions.append(Action("shell rc", f"strip echo-agent block from {rc}", lambda p=rc: self._strip_rc(p)))
        if not any(a.label == "shell rc" for a in actions):
            actions.append(Action("shell rc", "no echo-agent marker blocks in rc files"))

        if self.keep_data:
            actions.append(Action("data", f"kept by --keep-data: {self.data_dir}"))
        elif self.data_dir.exists():
            actions.append(Action("data", str(self.data_dir), lambda: self._remove_path(self.data_dir)))
        else:
            actions.append(Action("data", f"already absent: {self.data_dir}"))

        shims = [self._bin_dir / name for name in SHIM_NAMES if (self._bin_dir / name).exists()]
        for env in self._tool_envs:
            if env.exists():
                shims.append(env)
        if shims:
            for path in shims:
                actions.append(Action("leftovers", str(path), lambda p=path: self._remove_path(p)))
        else:
            actions.append(Action("leftovers", "no leftover shims or tool env"))
        return actions

    def print_plan(self) -> None:
        print("echo-agent uninstall plan:")
        for action in self.plan():
            mark = "run" if action.run else "skip"
            print(f"  [{mark:4s}] {action.label:12s} {action.detail}")

    # ── execution helpers ───────────────────────────────────────────────────
    def _stop_daemon(self) -> tuple[bool, str]:
        pidfile = self.data_dir / "echo-agent.pid"
        try:
            pid = int(pidfile.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return True, "no valid pidfile"
        try:
            os.kill(pid, 0)
        except OSError:
            self._unlink(pidfile)
            return True, f"pid {pid} not running"
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as exc:
            return False, str(exc)
        for _ in range(100):
            try:
                os.kill(pid, 0)
            except OSError:
                break
            time.sleep(0.1)
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        self._unlink(pidfile)
        return True, f"stopped pid {pid}"

    def _run(self, cmd: list[str]) -> tuple[bool, str]:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
        except OSError as exc:
            return False, str(exc)
        text = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = text[-1] if text else ""
        if self.verbose and detail:
            print(f"       {detail}")
        return proc.returncode == 0, detail

    def _remove_path(self, path: Path) -> tuple[bool, str]:
        if not (self._under_home(path) or self._uv_owned(path)):
            return False, f"refusing path outside $HOME: {path}"
        try:
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
        except OSError as exc:
            return False, str(exc)
        return True, f"removed {path}"

    def _strip_rc(self, path: Path) -> tuple[bool, str]:
        if not self._under_home(path):
            return False, f"refusing path outside $HOME: {path}"
        try:
            original = path.read_text(encoding="utf-8", errors="replace")
            cleaned = strip_marker_block(original)
            if cleaned != original:
                path.write_text(cleaned, encoding="utf-8")
        except OSError as exc:
            return False, str(exc)
        return True, f"cleaned {path}"

    @staticmethod
    def _unlink(path: Path) -> None:
        try:
            path.unlink()
        except OSError:
            pass

    # ── main flow ───────────────────────────────────────────────────────────
    def run(self) -> int:
        self.validate()
        actions = self.plan()
        if self.dry_run:
            self.print_plan()
            return 0
        if not self.yes:
            if not sys.stdin.isatty():
                print(
                    "uninstall needs confirmation; rerun with --yes or inspect with --dry-run",
                    file=sys.stderr,
                )
                return 2
            self.print_plan()
            try:
                answer = input("Remove everything listed above? [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = ""
            if answer not in {"y", "yes"}:
                print("aborted; nothing was removed")
                return 1

        did_something = False
        failed = False
        for action in actions:
            if action.run is None:
                print(f"  - {action.label}: {action.detail}")
                continue
            ok, detail = action.run()
            did_something = True
            mark = "✓" if ok else "✗"
            print(f"  {mark} {action.label}: {detail or action.detail}")
            if not ok:
                failed = True
        if not did_something:
            print("nothing to do — echo-agent is not installed")
            return 0
        if failed:
            print("uninstall finished with errors; check the ✗ lines above", file=sys.stderr)
            return 1
        print("echo-agent removed")
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="echo-uninstall", description="Remove echo-agent and all of its data"
    )
    parser.add_argument("--yes", "-y", action="store_true", help="no confirmation prompts")
    parser.add_argument("--keep-data", action="store_true", help="keep ~/.echo_agent")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; change nothing")
    parser.add_argument("--verbose", action="store_true", help="show command output")
    parser.add_argument("--data-dir", help="override the data root")
    args = parser.parse_args(argv)
    try:
        return Uninstaller(
            data_dir=args.data_dir,
            yes=args.yes,
            keep_data=args.keep_data,
            dry_run=args.dry_run,
            verbose=args.verbose,
        ).run()
    except UnsafePathError as exc:
        print(f"refusing to uninstall: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
