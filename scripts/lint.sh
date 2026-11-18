#!/usr/bin/env bash
# Lint everything this project owns without touching the pre-existing repo-wide
# ruff baseline.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

printf '== shell syntax ==\n'
bash -n install.sh uninstall.sh scripts/*.sh

printf '== shellcheck ==\n'
if command -v shellcheck >/dev/null 2>&1; then
    shellcheck install.sh uninstall.sh scripts/*.sh
elif command -v uvx >/dev/null 2>&1; then
    uvx --from shellcheck-py shellcheck install.sh uninstall.sh scripts/*.sh
else
    printf 'skip (shellcheck not installed)\n'
fi

printf '== ruff (owned modules) ==\n'
uv run ruff check \
    src/python_core/lifecycle.py \
    src/python_core/uninstall.py \
    src/python_core/setup_wizard.py \
    src/python_core/doctor.py

printf '== pytest ==\n'
uv run pytest tests/ -q -p no:cacheprovider
