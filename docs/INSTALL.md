# Installation details

## One-line install

```bash
curl -fsSL https://raw.githubusercontent.com/the-sanjeevkumar/echo-agent/main/install.sh | bash
```

The script installs echo-agent as an isolated **uv global tool** from GitHub —
no repository clone, no `uv sync`, no manually managed venv. Re-running it is an
idempotent upgrade (`uv tool install --force`).

Supported: Linux, macOS, and WSL. Native Windows without WSL prints WSL setup
instructions and exits; the Discord/Telegram gateways connect outward, so no
inbound ports or public webhook URL are needed.

## What the installer does

1. Detects the OS and verifies `git`; installs `uv` if missing (no sudo).
2. Installs `echo-agent[EXTRAS] @ git+…@REF` with `uv tool install --force`.
3. Adds the uv tool bin directory to PATH using a guarded block in your rc file:
   ```bash
   # >>> echo-agent >>>
   export PATH="$HOME/.local/bin:$PATH"
   # <<< echo-agent <<<
   ```
   Only the block is removed on uninstall; other rc content stays intact.
4. Writes `~/.echo_agent/install.json` (method, extras, version, timestamp,
   repo) and creates the data root with mode `0700`.
5. If stdin is a TTY (or `/dev/tty` is readable), launches `echo-agent setup`.

## Environment flags

| Variable | Meaning |
|---|---|
| `ECHO_EXTRAS` | Extras to install, default empty (core: Telegram works). E.g. `all`, `native,discord`. |
| `ECHO_VERSION` | Branch, tag, or commit to install (default `main`). |
| `ECHO_NO_SETUP=1` | Skip the post-install wizard. |
| `ECHO_NO_PROMPT=1` | Fully non-interactive install. |
| `ECHO_DATA_DIR` | Override the data root (default `~/.echo_agent`). |
| `UV_TOOL_BIN_DIR` | Override where uv tool shims are installed. |
| `NO_COLOR=1` / `TERM=dumb` | Plain output, no ANSI colors. |

Script arguments work when you download and run the file directly:

```bash
bash install.sh --extras all --version v0.2.0 --no-setup
bash install.sh --help
```

### Pinning a version

```bash
ECHO_VERSION=v0.2.0 bash install.sh          # release tag
ECHO_VERSION=9f3c1a2 bash install.sh         # exact commit
```

### Upgrading / switching extras

```bash
ECHO_EXTRAS=all bash install.sh              # reinstall with every extra
uv tool install --force 'echo-agent[discord] @ git+https://github.com/the-sanjeevkumar/echo-agent.git@main'
```

Discord requires the `discord` extra; Telegram uses plain `httpx` and works with
core dependencies. Chromadb/tiktoken (the `native` extra) and Scrapling browser
binaries are optional.

## First-run setup

```bash
echo-agent setup                    # wizard (QuickStart or Advanced)
echo-agent setup --quick            # accept safe defaults
echo-agent setup --llm-base-url http://127.0.0.1:11434/v1 --llm-model llama3.1 --yes
echo-agent agent                    # start (alias of `start`)
```

`agent`/`start`/`run` trigger the wizard once if `<data_dir>/.setup-complete`
is missing and a TTY is available. Scripts get one hint line and boot with
defaults. Configuration is merged into `<data_dir>/config.toml` (mode `0600`);
unknown keys survive. Tokens are read with `getpass`, never printed.

## WSL notes

- Install inside the WSL distro, not from native Windows: `wsl --install -d Ubuntu`.
- Keep the data root on the Linux filesystem (`~/.echo_agent`), not a `/mnt/c/`
  Windows mount — file modes and `os.replace` behave correctly on ext4.
- PATH edits go into the distro's `.bashrc`/`.profile`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `echo-agent: command not found` after install | Open a new shell, or add `~/.local/bin` to PATH; run `echo-agent doctor`. |
| `uv tool install` fails | Check network/DNS and that the `ECHO_VERSION` ref exists; retry with `ECHO_VERSION=main`. |
| Token validation fails in setup | The token is still saved; verify with `echo-agent gateway test telegram\|discord`. |
| LLM endpoint unreachable | Start Ollama/LM Studio or set `[llm] base_url`; `echo-agent doctor --probe`. |
| Browser features missing | `uv run scrapling install` (one-time download into shared caches). |
| Want a clean slate | `echo-uninstall --dry-run`, then `echo-uninstall --yes`. |

## Uninstall

```bash
echo-uninstall --dry-run            # print every path and command
echo-uninstall --yes                # remove everything
echo-uninstall --yes --keep-data    # keep ~/.echo_agent
curl -fsSL <uninstall.sh-url> | bash -s -- --yes   # standalone fallback
```

The uninstaller stops the daemon, removes the uv tool environment and shims,
strips only its own rc marker block, and deletes the data root after one
confirmation. Browser binaries are removed only when `browsers-owned.json`
records them as ours; shared caches are otherwise kept. See the removal table
in the main README.
