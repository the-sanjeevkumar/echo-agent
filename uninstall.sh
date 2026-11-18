#!/usr/bin/env bash
# echo-agent · standalone uninstaller (curl-able even if the install is broken)
#
#   curl -fsSL <uninstall.sh-url> | bash -s -- --yes
#   bash uninstall.sh --dry-run
#
# Delegates to the Python uninstaller when available; otherwise cleans up here.
set -euo pipefail

MARKER_START='# >>> echo-agent >>>'
MARKER_END='# <<< echo-agent <<<'

say() { printf '%s\n' "$*"; }
warn() { printf '! %s\n' "$*" >&2; }

usage() {
    cat <<'EOF'
echo-agent uninstaller

Usage: bash uninstall.sh [--yes] [--keep-data] [--dry-run] [--verbose]

  --yes, -y   no confirmation prompts
  --keep-data remove the program but keep ~/.echo_agent
  --dry-run   print the exact plan; change nothing
  --verbose   show command output
EOF
}

fallback_uninstall() {
    local yes=0 keep_data=0 dry_run=0 verbose=0 data_dir="${ECHO_DATA_DIR:-$HOME/.echo_agent}"
    local pid uvBin bin rc_file
    while [ $# -gt 0 ]; do
        case "$1" in
            --yes|-y) yes=1; shift ;;
            --keep-data) keep_data=1; shift ;;
            --dry-run) dry_run=1; shift ;;
            --verbose) verbose=1; shift ;;
            --data-dir) [ $# -ge 2 ] || { warn "--data-dir needs a value"; exit 2; }; data_dir="$2"; shift 2 ;;
            -h|--help) usage; exit 0 ;;
            *) warn "ignoring unknown argument: $1"; shift ;;
        esac
    done

    [ -n "${HOME:-}" ] || { warn "HOME is not set; refusing to delete anything"; exit 2; }
    [ "$HOME" != "/" ] || { warn "HOME is /; refusing to delete anything"; exit 2; }
    case "$data_dir" in
        "$HOME"|"/") warn "refusing to delete $data_dir"; exit 2 ;;
    esac

    uvBin="${UV_TOOL_BIN_DIR:-$HOME/.local/bin}"

    say "echo-agent fallback uninstall plan:"
    say "  stop daemon:  $data_dir/echo-agent.pid"
    say "  tool env:     uv tool uninstall echo-agent"
    say "  shims:        $uvBin/echo-agent, $uvBin/echo-uninstall"
    say "  rc blocks:    $HOME/.bashrc, $HOME/.zshrc, $HOME/.profile"
    if [ "$keep_data" -eq 1 ]; then
        say "  data dir:     kept (--keep-data)"
    else
        say "  data dir:     $data_dir"
    fi
    if [ "$dry_run" -eq 1 ]; then
        exit 0
    fi

    if [ "$yes" -ne 1 ]; then
        if [ ! -t 0 ]; then
            warn "confirmation required; rerun with --yes or --dry-run"
            exit 2
        fi
        printf 'Remove everything listed above? [y/N]: '
        read -r answer || answer=""
        case "$answer" in
            y|Y|yes|YES) ;;
            *) say "aborted; nothing was removed"; exit 1 ;;
        esac
    fi

    pid_file="$data_dir/echo-agent.pid"
    if [ -f "$pid_file" ]; then
        pid="$(cat "$pid_file" 2>/dev/null || printf '')"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
            say "✓ stopped daemon $pid"
        fi
        rm -f "$pid_file"
    fi

    if command -v uv >/dev/null 2>&1; then
        if uv tool uninstall echo-agent; then say "✓ removed uv tool environment"; else warn "uv tool uninstall reported a problem"; fi
        uv cache clean echo-agent >/dev/null 2>&1 || true
    else
        warn "uv not found; skip tool environment removal"
    fi

    for rc_file in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.profile"; do
        if [ -f "$rc_file" ] && grep -Fq "$MARKER_START" "$rc_file"; then
            awk -v start="$MARKER_START" -v end="$MARKER_END" '
                $0 == start { inside = 1; next }
                $0 == end   { inside = 0; next }
                !inside     { print }
            ' "$rc_file" > "$rc_file.tmp" && mv "$rc_file.tmp" "$rc_file"
            say "✓ cleaned $rc_file"
        fi
    done

    if [ "$keep_data" -ne 1 ] && [ -d "$data_dir" ]; then
        rm -rf "$data_dir"
        say "✓ removed $data_dir"
    fi

    for bin in "$uvBin/echo-agent" "$uvBin/echo-uninstall"; do
        if [ -L "$bin" ] || [ -f "$bin" ]; then
            rm -f "$bin"
            say "✓ removed $bin"
        fi
    done
    if [ "$verbose" -eq 1 ]; then
        say "done"
    fi
    say "echo-agent removed"
}

main() {
    local exe
    if command -v echo-uninstall >/dev/null 2>&1; then
        exe="$(command -v echo-uninstall)"
    else
        exe="${UV_TOOL_BIN_DIR:-$HOME/.local/bin}/echo-uninstall"
    fi
    if [ -x "$exe" ]; then
        "$exe" "$@"
        return $?
    fi
    warn "Python uninstaller unavailable; using bash fallback"
    fallback_uninstall "$@"
}

main "$@"
