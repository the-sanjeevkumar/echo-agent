# echo-agent

Local-first, self-improving, event-driven AI agent. A compact Python microkernel
orchestrates an LLM through a ReAct-style tool loop, with dual-process memory,
semantic caching, self-healing patches, skill ingestion, a universal inter-agent
manifest protocol, and production-ready **Telegram** and **Discord** gateways.

Everything is plugin-based behind a strict `Protocol`/event-bus architecture, so
features can be deleted, replaced, or hot-swapped without touching the kernel.

---

## Highlights

- **ReAct agent loop** with structured JSON outputs, bounded steps, and delta-first
  tool observations.
- **Dual-process memory**: working memory, episodic SQLite journal, semantic vector
  store (chromadb or NumPy fallback).
- **Token efficiency**: tiktoken-exact counting, pruning, folding, and a semantic
  cache scoped per chat session.
- **Self-healing**: sandboxed patch application, tests, atomic rollback, hot reload.
- **Skill ingestion**: turn any git repo into live tools at runtime.
- **Inter-agent protocol**: `/manifest.json` + JSON-RPC `/execute`; universal client
  imports remote agents as local tools.
- **Gateways**: Telegram (plain `httpx` long polling) and Discord (`discord.py`),
  with OpenClaw-style access control, pairing, mention gating, per-chat sessions,
  chunking, reply threading, and reconnect/backoff.
- **CLI**: compact/animated output, `pairing`, `channels`/`gateway`, daemon
  start/stop/restart/status.

---

## Quick start

```bash
git clone <your-fork> echo-agent
cd echo-agent
uv sync                 # core deps (Telegram works out of the box)
uv sync --extra discord # optional Discord support
uv run pytest tests/ -p no:cacheprovider   # 151 passed
```

Run a one-shot task:

```bash
uv run echo-agent run "list the files in this repo"
```

Start the daemon (manifest server + gateways + REPL):

```bash
uv run echo-agent start
# or headless
uv run echo-agent start --headless
```

Without an LLM server the kernel still boots; the provider simply reports
connection failures.

---

## Configuration

State lives under one data root (default `~/.echo_agent`), so the agent is
portable and trivially backed up. Configuration comes from an optional TOML file
(`~/.echo_agent/config.toml`, or `--config`), then environment variables **win**
over TOML.

```toml
[llm]
base_url = "http://127.0.0.1:11434/v1"   # ollama, LM Studio, vLLM, llama.cpp …
model = "llama3.1"
api_key = ""                              # local servers rarely need one
timeout = 120.0

[tokens]
max_context_tokens = 8192                 # hard prompt-window budget
cache_similarity_threshold = 0.93         # semantic cache cosine threshold

[memory]
idle_seconds = 20.0                       # idle trigger for consolidation
wm_capacity = 64                          # working-memory ring capacity

[web]
max_chars = 8000                          # page text bound fed to the LLM
timeout = 30.0                            # per-fetch timeout (seconds)
rate_limit_per_min = 30                   # per-domain fetch budget
auto_stealth = true                       # retry JS-only pages in headless browser
allowed_domains = []                      # empty ⇒ all public hosts
blocked_domains = []                      # suffix match; wins over allowed

[discord]
token = ""                                # or ECHO_DISCORD_TOKEN
dm_policy = "pairing"                     # pairing | allowlist | open | disabled
allow_from = []                           # DM user IDs / usernames
group_policy = "allowlist"                # allowlist | open | disabled
guilds = []                               # guild IDs allowed (empty ⇒ no guilds)
channels = []                             # optional channel IDs (empty ⇒ whole guild)
group_allow_from = []                     # optional guild sender filter
require_mention = true
allow_bots = false
command_prefix = "!"
slash_commands = true

[telegram]
token = ""                                # or ECHO_TELEGRAM_TOKEN
dm_policy = "pairing"
allow_from = []                           # numeric user IDs or @usernames
group_policy = "allowlist"
groups = []                               # chat IDs allowed (empty ⇒ no groups)
group_allow_from = []
require_mention = true
ignore_other_mentions = true
link_preview = false

[gateway]
reply_to_mode = "off"                     # off | first | all
typing = true
pairing_ttl = 3600                        # seconds
pairing_max_pending = 3                   # per channel

[session]
history_limit = 40                        # messages retained per chat
max_sessions = 128                        # LRU cap
persistence = true                        # store under <data_dir>/sessions.json
```

Top-level keys such as `manifest_port = 8792` must appear **before** any section.

### Environment variables

| Variable | Meaning |
|---|---|
| `ECHO_DATA_DIR` | Data root (default `~/.echo_agent`). |
| `ECHO_LLM_BASE_URL` / `ECHO_LLM_MODEL` / `ECHO_LLM_API_KEY` | LLM endpoint settings. |
| `ECHO_LLM_RETRIES` | Bounded retries on HTTP 429/5xx. |
| `ECHO_DISCORD_TOKEN` / `ECHO_TELEGRAM_TOKEN` | Enable the respective gateway. |
| `ECHO_MANIFEST_PORT` | Manifest server port (default `8790`). |
| `ECHO_MAX_CONTEXT_TOKENS` | Prompt budget for pruning. |
| `ECHO_BUS_SOCKET` / `ECHO_ENABLE_BUS=1` | Rust bus bridge. |
| `ECHO_ALLOW_SHELL=1` | Opt-in for `shell_exec`. |

Data layout:

```
~/.echo_agent/
├── config.toml          # optional
├── sessions.json        # per-DM/group/thread chat history (bounded)
├── echo-agent.pid       # background daemon pid
├── echo-agent.log       # daemon log
├── gateways/
│   ├── pairing.json     # pending/approved DM senders (mode 0600)
│   └── telegram-offset  # durable long-poll offset
├── memory/
│   ├── episodes.sqlite3 # episodic store
│   └── vectors/         # semantic tier
└── skills/<name>/       # ingested skill wrappers
```

---

## CLI reference

| Command | Effect |
|---|---|
| `echo-agent run "task"` | One-shot task through the ReAct loop. |
| `echo-agent start [--headless]` | Foreground daemon (REPL unless headless). |
| `echo-agent gateway start\|stop\|restart\|status` | Manage the background daemon. |
| `echo-agent status` / `stop` / `restart` | Top-level daemon shortcuts. |
| `echo-agent skills add <url\|path>` | Ingest a skill repository. |
| `echo-agent tools` | List builtin tools. |
| `echo-agent plugins` | Dry manifest of builtin plugins. |
| `echo-agent channels list` | Show channel status. |
| `echo-agent channels add --channel <telegram\|discord> [--token ...]` | Add/configure a channel. |
| `echo-agent channels remove --channel <telegram\|discord>` | Remove a channel token. |
| `echo-agent channels test --channel <telegram\|discord>` | Validate a bot token. |
| `echo-agent channels status [--probe]` | Status, optionally probing tokens. |
| `echo-agent pairing list [telegram\|discord] [--json]` | Pending/approved DM senders. |
| `echo-agent pairing approve <channel> <code>` | Approve a DM pairing code. |
| `echo-agent pairing revoke <channel> <sender_id>` | Revoke an approved sender. |
| `echo-agent bus` | Launch the Rust `echo-bus`. |
| `echo-agent version` | Print version. |

Global flags: `--config path/to/config.toml`, `--verbose/-v`.

---

## Gateways

`plugins/gateways/base.py` defines the shared gateway contract and behavior:
lifecycle, reply routing, access decisions, chunking, native reply tags, typing
indicators, and per-chat session keys. Platform adapters stay thin.

### Telegram (plain `httpx`, no `python-telegram-bot`)

- Long polling, `getMe` identity, native command menu, offset persistence,
  reconnect/backoff.
- DMs default to **pairing**; groups default to an **allowlist**.
- Commands: `/ask`, `/status`, `/reset`, `/whoami`, `/activation`, `/help`, `/start`.
- Group behavior: optional sender filtering, @mention gating (or reply to bot, or
  native command), per-topic sessions.

### Discord (`discord.py`, optional extra)

- Gateway client with message content intent, guild/channel/sender allowlists.
- Text commands (`!ask`, `!status`, `!reset`, `!whoami`, `!activation`, `!help`)
  and native slash commands (`/ask`, `/status`, `/reset`, `/whoami`,
  `/activation`, `/help`).
- Join introductions, chunking, reply threading, reconnect/backoff.

### Access control defaults (OpenClaw-style)

- **DMs**: `pairing` by default. Unknown senders receive an 8-character code
  (no ambiguous `0/O/1/I`) and their message is **not** processed.
- **Groups/guilds**: `allowlist` by default; empty allowlist means closed.
- Approve from the host while the daemon keeps running:
  ```bash
  uv run echo-agent pairing list telegram
  uv run echo-agent pairing approve telegram ABCD2345
  uv run echo-agent pairing revoke telegram 123456789
  ```

### Sessions

Each DM, group, forum topic, Discord channel/thread gets its own session key
(`telegram:dm:<id>`, `telegram:chat:<id>:topic:<id>`,
`discord:guild:<g>:channel:<c>`). Tasks in one session serialize through a
per-session lock; history is bounded by `session_history_limit`; sessions persist
to `sessions.json`. `/reset` clears only the current chat. Semantic-cache entries
are scoped per session.

### Platform prerequisites

- **Telegram**: disable *Group Privacy* in BotFather (or make the bot an admin)
  for ordinary group messages; commands still work with privacy enabled.
- **Discord**: enable the **Message Content** privileged intent and the
  `applications.commands` scope for slash commands.

---

## Internet access (`plugins/web/`)

Three tools (`web_fetch`, `web_extract`, `web_links`) with a backend ladder:
Scrapling `Fetcher` → `StealthyFetcher` (auto for JS-only pages) → plain `httpx`.
Safety: http/https only, SSRF guard, domain allow/block lists, per-domain rate
limit, output truncation. Web failures are returned as data, never raised.

Browser binaries are a one-time install:

```bash
uv run scrapling install
```

---

## Event bus topics

| Topic | Emitted by | Meaning |
|---|---|---|
| `task:incoming` | gateways | user text arrived (includes `session_id`) |
| `task:received` | orchestrator | task accepted into the loop |
| `task:completed` | orchestrator | final answer produced (includes `session_id`) |
| `session:reset` / `session:status` | gateways ↔ orchestrator | per-chat reset / message count |
| `tool:register` / `tool:unregister` | registry, skills, universal_client | tool schema lifecycle |
| `tool:executed` | tool registry | envelope with ok/error + timing |
| `memory:*` | memory plugin | tier writes and consolidation |
| `selfheal:*` | self-healing | patch lifecycle |
| `plugin:*` | plugin manager | lifecycle + per-plugin events |
| `skill:ingested` | skills | a repo became tools |

---

## Project layout

```
echo-agent/
├── Cargo.toml
├── CMakeLists.txt
├── pyproject.toml
├── uv.lock
├── src/
│   ├── rust_bus/main.rs
│   ├── cpp_memory/index.cpp
│   └── python_core/
│       ├── main.py               # CLI + kernel bootstrap
│       ├── orchestrator.py        # ReAct loop, pruning, cache, sessions
│       ├── core/                 # event_bus, plugin_manager, interfaces, config
│       └── plugins/
│           ├── memory/
│           ├── self_healing/
│           ├── skills/
│           ├── llm/
│           ├── web/
│           └── gateways/         # base, pairing, discord, telegram
└── tests/                        # 151-test full-kernel suite
```

---

## Optional native layers

- **C++ cognitive engine**: `cmake -B build -S . && cmake --build build`
  (falls back to NumPy store).
- **Accelerated deps**: `uv sync --extra native` (tiktoken, chromadb).
- **Web access**: Scrapling is a core dependency; only browser binaries need the
  one-time install.

Extras: `native`, `discord`, `telegram` (empty: Telegram uses plain `httpx`),
`all` — e.g. `uv sync --extra all`.

---

## Testing

```bash
uv run pytest tests/ -q -p no:cacheprovider
# 151 passed
```

The full-kernel fixture builds the *real* system (only the LLM is swapped for
`FakeProvider`), so plugin wiring, gateways, sessions, and access control are
tested end-to-end. The suite covers memory tiers, consolidation, diff
round-trips, hot reload, skill ingestion, manifest server, universal client,
gateway pairing/access/chunking, per-chat sessions + persistence, web
safety/fetch backends, orchestrator loop, terminal UI, config precedence, and the
Rust bridge protocol codec.