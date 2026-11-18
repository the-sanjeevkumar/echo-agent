#!/usr/bin/env bash
# End-to-end smoke test for install.sh + echo-uninstall in a disposable HOME.
# Skips gracefully when uv/git/network are unavailable.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

skip() { printf 'skip: %s\n' "$*"; exit 0; }

command -v uv >/dev/null 2>&1 || skip "uv not installed"
command -v git >/dev/null 2>&1 || skip "git not installed"
GIT_TERMINAL_PROMPT=0 git ls-remote https://github.com/the-sanjeevkumar/echo-agent.git HEAD >/dev/null 2>&1 || skip "GitHub unreachable"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/home"
export HOME="$TMP/home"
export XDG_DATA_HOME="$TMP/xdg-data"
export XDG_CACHE_HOME="$TMP/xdg-cache"
export UV_TOOL_BIN_DIR="$TMP/bin"
export ECHO_NO_SETUP=1

printf '== install ==\n'
if [ "${ECHO_INSTALL_LOCAL:-0}" = "1" ]; then
    # Verify the working tree end-to-end before changes are pushed to GitHub.
    uv tool install --force "$ROOT"
else
    bash install.sh
fi
[ -x "$UV_TOOL_BIN_DIR/echo-agent" ] || { printf 'FAIL: echo-agent shim missing\n' >&2; exit 1; }
"$UV_TOOL_BIN_DIR/echo-agent" version

printf '== uninstall dry-run ==\n'
"$UV_TOOL_BIN_DIR/echo-uninstall" --dry-run

printf '== uninstall ==\n'
"$UV_TOOL_BIN_DIR/echo-uninstall" --yes

[ ! -d "$HOME/.echo_agent" ] || { printf 'FAIL: data root survived uninstall\n' >&2; exit 1; }
if [ -f "$HOME/.bashrc" ] && grep -Fq '# >>> echo-agent >>>' "$HOME/.bashrc"; then
    printf 'FAIL: PATH marker survived uninstall\n' >&2
    exit 1
fi
printf 'PASS: install → version → uninstall is clean\n'
