#!/usr/bin/env bash
# echo-agent · one-line installer (Linux / macOS / WSL)
#
#   curl -fsSL https://raw.githubusercontent.com/the-sanjeevkumar/echo-agent/main/install.sh | bash
#
# Installs as an isolated uv global tool — no clone, no repo checkout needed.
set -euo pipefail

ECHO_REPO="https://github.com/the-sanjeevkumar/echo-agent.git"

echo_color() {
    local code="$1"; shift
    if [ -t 1 ] && [ -z "${NO_COLOR:-}" ] && [ "${TERM:-dumb}" != "dumb" ]; then
        printf '\033[%sm%s\033[0m\n' "$code" "$*"
    else
        printf '%s\n' "$*"
    fi
}
echo_info() { echo_color "36" "$*"; }
echo_ok() { echo_color "32" "✓ $*"; }
echo_warn() { echo_color "33" "! $*"; }
echo_err() { echo_color "31" "✗ $*" >&2; }
die() { echo_err "$*"; exit 1; }

usage() {
    cat <<'EOF'
echo-agent installer

Usage:
  curl -fsSL <install.sh-url> | bash
  bash install.sh [--extras all] [--version v0.2.0] [--no-setup]

Options:
  --extras LIST     uv extras to install (e.g. all, native,discord)
  --version REF     branch / tag / commit to install (default: main)
  --no-setup        do not launch the setup wizard after installing
  -h, --help        show this help

Environment:
  ECHO_EXTRAS       same as --extras
  ECHO_VERSION      same as --version
  ECHO_NO_SETUP=1   skip the post-install wizard
  ECHO_NO_PROMPT=1  fully non-interactive
EOF
}

detect_os() {
    local uname_s
    uname_s="$(uname -s)"
    case "$uname_s" in
        Linux*)
            if [ -n "${WSL_DISTRO_NAME:-}" ] || grep -qi microsoft /proc/version 2>/dev/null; then
                echo "wsl"
            else
                echo "linux"
            fi
            ;;
        Darwin*) echo "macos" ;;
        MINGW*|MSYS*|CYGWIN*)
            echo_err "native Windows is not supported."
            echo_err "Install WSL (wsl --install), then rerun this command inside Ubuntu."
            exit 1
            ;;
        *) echo "unknown" ;;
    esac
}

ensure_uv() {
    if command -v uv >/dev/null 2>&1; then
        echo_ok "uv found: $(command -v uv)"
        return
    fi
    echo_info "uv not found — installing it (no sudo needed)…"
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh || die "uv installation failed; see https://docs.astral.sh/uv/"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh || die "uv installation failed; see https://docs.astral.sh/uv/"
    else
        die "neither curl nor wget is available; install one and retry"
    fi
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 || die "uv installed but not on PATH; add $HOME/.local/bin to PATH"
    echo_ok "uv installed"
}

tool_bin_dir() {
    if [ -n "${UV_TOOL_BIN_DIR:-}" ]; then
        printf '%s\n' "$UV_TOOL_BIN_DIR"
    elif [ -n "${XDG_BIN_HOME:-}" ]; then
        printf '%s\n' "$XDG_BIN_HOME"
    else
        printf '%s\n' "$HOME/.local/bin"
    fi
}

extras_json() {
    local extras="$1" out="[" first=1 item
    if [ -n "$extras" ]; then
        local old_ifs="$IFS"
        IFS=','
        for item in $extras; do
            [ -n "$item" ] || continue
            if [ "$first" -eq 0 ]; then out="$out, "; fi
            out="$out\"$item\""
            first=0
        done
        IFS="$old_ifs"
    fi
    printf '%s]\n' "$out"
}

add_path_block() {
    local bin="$1" path_block changed=0 rc_file
    path_block="# >>> echo-agent >>>
export PATH=\"$bin:\$PATH\"
# <<< echo-agent <<<"
    case ":$PATH:" in
        *":$bin:"*)
            echo_ok "PATH already contains $bin"
            return
            ;;
    esac
    for rc_file in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.profile"; do
        if [ -f "$rc_file" ]; then
            if grep -Fq '# >>> echo-agent >>>' "$rc_file"; then
                continue
            fi
            printf '\n%s\n' "$path_block" >> "$rc_file"
            echo_ok "updated PATH in $rc_file"
            changed=1
        fi
    done
    if [ "$changed" -eq 0 ]; then
        printf '%s\n' "$path_block" > "$HOME/.profile"
        echo_ok "created $HOME/.profile with echo-agent on PATH"
    fi
    export PATH="$bin:$PATH"
}

write_metadata() {
    local data_dir="$1" extras="$2" ref="$3" now
    now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    mkdir -p "$data_dir"
    chmod 700 "$data_dir"
    cat > "$data_dir/install.json" <<EOF
{
  "method": "uv-tool",
  "extras": $(extras_json "$extras"),
  "version": "$ref",
  "installed_at": "$now",
  "repo": "$ECHO_REPO"
}
EOF
    chmod 600 "$data_dir/install.json"
    echo_ok "wrote $data_dir/install.json"
}

maybe_setup() {
    local bin="$1"
    if [ "${ECHO_NO_SETUP:-0}" = "1" ] || [ "${NO_SETUP:-0}" = "1" ] || [ "${ECHO_NO_PROMPT:-0}" = "1" ]; then
        return 0
    fi
    if [ -t 0 ]; then
        echo_info "launching setup wizard…"
        "$bin" setup || echo_warn "setup did not finish; run: echo-agent setup"
    elif [ -r /dev/tty ]; then
        echo_info "launching setup wizard on /dev/tty…"
        "$bin" setup < /dev/tty || echo_warn "setup did not finish; run: echo-agent setup"
    else
        echo_info "no TTY available; run 'echo-agent setup' when ready"
    fi
}

main() {
    local extras="${ECHO_EXTRAS:-}" ref="${ECHO_VERSION:-}" no_setup=0
    local os spec bin data_dir full_bin installed_version

    while [ $# -gt 0 ]; do
        case "$1" in
            --extras) [ $# -ge 2 ] || die "--extras needs a value"; extras="$2"; shift 2 ;;
            --extras=*) extras="${1#*=}"; shift ;;
            --version) [ $# -ge 2 ] || die "--version needs a value"; ref="$2"; shift 2 ;;
            --version=*) ref="${1#*=}"; shift ;;
            --no-setup) no_setup=1; shift ;;
            -h|--help) usage; exit 0 ;;
            *) echo_warn "ignoring unknown argument: $1"; shift ;;
        esac
    done
    NO_SETUP=$no_setup
    ref="${ref:-main}"

    os="$(detect_os)"
    echo_info "echo-agent installer · $os · ref=$ref"
    [ -n "${HOME:-}" ] || die "HOME is not set"
    command -v git >/dev/null 2>&1 || die "git is required (install git, then retry)"

    ensure_uv
    bin="$(tool_bin_dir)"
    mkdir -p "$bin"
    add_path_block "$bin"

    if [ -n "$extras" ]; then
        spec="echo-agent[$extras] @ git+${ECHO_REPO}@${ref}"
    else
        spec="echo-agent @ git+${ECHO_REPO}@${ref}"
    fi
    echo_info "installing $spec …"
    uv tool install --force "$spec" || die "installation failed; check your network and the ref '$ref'"

    data_dir="${ECHO_DATA_DIR:-$HOME/.echo_agent}"
    write_metadata "$data_dir" "$extras" "$ref"

    full_bin="$bin/echo-agent"
    [ -x "$full_bin" ] || full_bin="$(command -v echo-agent || printf 'echo-agent')"
    installed_version="$("$full_bin" version 2>/dev/null || printf 'echo-agent (version unknown)')"
    echo_ok "installed: $installed_version"
    echo_info "next: echo-agent setup  ·  echo-agent agent  ·  echo-agent doctor"
    maybe_setup "$full_bin"
}

main "$@"
