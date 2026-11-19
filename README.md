# echo-agent

Local-first, self-improving, event-driven AI agent. A compact Python microkernel
orchestrates an LLM through a ReAct-style tool loop, with durable **owner-profile
memory** (name survives `/reset` and daemon restarts), **curated USER.md/MEMORY.md**
stores injected into every prompt, a Hermes-style **learning loop** (memory +
skills + self-healing), **live activity streaming** to Telegram/Discord/terminal,
**server-control** tools with chat confirmations, and **supervisor mode** over
remote agents via the universal manifest protocol.

Everything is plugin-based behind a strict `Protocol`/event-bus architecture, so
features can be deleted, replaced, or hot-swapped without touching the kernel.

LLM: any OpenAI-compatible endpoint — the intended default is a strong NVIDIA
Nemotron model via `https://integrate.api.nvidia.com/v1` (`ECHO_LLM_API_KEY`).
No local LLM inference; local embeddings (ONNX all-MiniLM, ~100MB) are optional
and degrade to a deterministic local fallback.

Hardware ceiling: 16GB RAM / 477GB SSD. Single owner (one user across CLI,
Telegram and Discord).

---

## Highlights

- **ReAct agent loop** with structured JSON outputs, bounded steps, and delta-first
  tool observations.
- **Durable owner profile** (`<data_dir>/profiles/owner.json`, atomic writes):
  CLI maps to `owner`; Telegram/Discord user IDs listed in `owner_user_ids`
  resolve to the same shared profile; every other sender is isolated. Injected
  into every system prompt — survives `/reset` and restarts.
- **Curated memory**: `USER.md` (owner identity, ≤1375 chars) and `MEMORY.md`
  (agent notes: server facts, conventions, lessons, ≤2200 chars) snapshotted into
  every prompt with usage headers (`USER.md [42% — 580/1375]`) and `§` separators.
  Written only through the `memory` tool (add/patch/remove), unique-substring
  matching, exact-duplicate rejection, capacity errors that never silently drop,
  optional write-approval gate.
- **Hermes-style learning loop**: a post-turn background review extracts facts,
  preferences, lessons and skill candidates and writes them only through the
  memory/profile/skills tools. Recurring two-step tasks synthesize exactly one
  `skills/<name>/SKILL.md` (visible in the next prompt); consolidator dedups
  (cosine ≥0.90), applies salience decay and evicts stale patterns; recurring
  failures propose code patches through the existing self-healing sandbox with
  chat approval by default.
- **Live streaming**: `task:progress` bus events (thinking, tool + args gist,
  tool_result, file_edit diff gist, memory, skill, selfheal, supervisor) — one
  editable status message per task on Telegram/Discord (debounce ≥2s), final
  answer sent separately; CLI spinner shows tool names and edit diffs.
- **Server-ops pack**: `sys_metrics`, `service_status`, `service_restart`
  (systemd allowlist), `journal_tail`, `disk_usage`, `process_list`, `docker_ps`.
  Read-only by default; mutations need an allowlisted unit + `/confirm <id>`;
  cron health checks alert the owner chat on disk/RAM/unit breaches.
- **Supervisor mode**: registry (`agents.json`), `agents_list`, `agent_health`,
  `agent_delegate` over JSON-RPC `/execute`; retry-with-backoff, heartbeat cron,
  daily digest, failure alerts; outcomes feed MEMORY.md reliability lessons.
- **Self-healing**: sandboxed patch validation, full pytest suite, atomic
  rollback, hot reload.
- **Skill ingestion**: turn any git repo into live tools at runtime.
- **Inter-agent protocol**: `/manifest.json` + JSON-RPC `/execute`; universal
  client imports remote agents as local namespaced tools.
- **Gateways**: Telegram (plain `httpx` long polling) and Discord (`discord.py`),
  OpenClaw-style access control, pairing, mention gating, per-chat sessions,
  chunking, reply threading, reactions (👀 / ✅ / ❌), live `/model` switching.
- **CLI**: compact/animated output, `pairing`, `channels`/`gateway`, `profile`,
  `memory`, daemon start/stop/restart/status.
- **Filesystem sandbox**: `fs_allowed_roots` in config controls which directories
  the agent may read/write (default: project root, data dir, cwd, system temp;
  set `["/"]` for full access).

---

## Quick start

```bash
curl -fsSL https://raw.githubusercontent.com/the-sanjeevkumar/echo-agent/main/install.sh | bash
echo-agent setup        # wizard (auto-runs on first TTY install)
echo-agent agent        # start the agent (alias of `start`)
```

No clone and no `uv sync` — the installer puts echo-agent in an isolated uv tool
environment and adds it to your PATH. Re-running the one-liner upgrades in place
(`install.sh` now defaults to `--extras all`).

Run a one-shot task:

```bash
echo-agent run "list the files in this repo"
```

Start the daemon (manifest server + gateways + REPL):

```bash
echo-agent agent
# or headless
echo-agent agent --headless
```

Without an LLM server the kernel still boots; the provider simply reports
connection failures. Bare `echo-agent` runs the wizard on first use, then
prints help + daemon status.

---

## Configuration

State lives under one data root (default `~/.echo_agent`), so the agent is
portable and trivially backed up. Configuration comes from an optional TOML file
(`~/.echo_agent/config.toml`, or `--config`), then environment variables **win**
over TOML.

### Example config — NVIDIA Nemotron via OpenAI-compatible API

```toml
# ~/.echo_agent/config.toml  ·  single owner across CLI, Telegram, Discord

[llm]
base_url = "https://integrate.api.nvidia.com/v1"   # Nemotron OpenAI-compatible endpoint
model = "nvidia/nemotron-3-ultra-550b-a55b"          # set via ECHO_LLM_MODEL too
api_key = ""                                         # ECHO_LLM_API_KEY (never committed)
timeout = 120.0
retries = 3

# single owner identity shared across platforms
owner_user_ids = ["7974643102"]                      # Telegram/Discord user IDs treated as owner

[tokens]
max_context_tokens = 8192
cache_similarity_threshold = 0.93

[memory]
idle_seconds = 20.0
wm_capacity = 64

# curated memory write gate: off (auto-save, Hermes-style) | on
memory_write_approval = false

# embeddings: auto (ONNX MiniLM if model files present) | hash (offline fallback)
embeddings_backend = "auto"
embedding_model_dir = ""        # default: <data_dir>/models/all-MiniLM-L6-v2
enable_embeddings = true

# learning loop (PHASE 4)
learning_enabled = true
review_debounce_seconds = 30.0
review_max_chars = 2400
auto_apply_if_tests_pass = false   # code patches need /patch approve by default

# server ops (PHASE 6)
server_ops_enabled = true
server_ops_units = ["nginx.service", "postgres.service"]   # systemd allowlist
server_ops_preapproved_units = []                          # skip /confirm
server_ops_confirm_ttl = 120.0
server_ops_health_interval = "every 5m"
server_ops_disk_threshold = 90.0
server_ops_ram_threshold = 90.0
server_ops_alerts_reply_channel = "telegram:111222333"

# supervisor (PHASE 7)
supervisor_enabled = true
supervisor_heartbeat_interval = "every 5m"
supervisor_digest_schedule = "daily at 9am"
supervisor_retries = 2
supervisor_alerts_reply_channel = "telegram:111222333"

# gateways (MODULE 6)
[discord]
token = ""                                # or ECHO_DISCORD_TOKEN
dm_policy = "pairing"
allow_from = []
group_policy = "allowlist"
guilds = []
channels = []
group_allow_from = []
require_mention = true
allow_bots = false
command_prefix = "!"
slash_commands = true

[telegram]
token = ""                                # or ECHO_TELEGRAM_TOKEN
dm_policy = "pairing"
allow_from = []
group_policy = "allowlist"
groups = []
group_allow_from = []
require_mention = true
ignore_other_mentions = true
link_preview = false

[gateway]
reply_to_mode = "off"                     # off | first | all
progress = "compact"                      # off | compact | verbose — live status edits
status_debounce = 2.0                     # seconds between status-message edits
typing = true
pairing_ttl = 3600                        # seconds
pairing_max_pending = 3                   # per channel

[web]
max_chars = 8000
timeout = 30.0
rate_limit_per_min = 30
auto_stealth = true
allowed_domains = []
blocked_domains = []

[fs]
fs_allowed_roots = []                     # empty ⇒ safe defaults; set ["/"] for full access

[session]
history_limit = 40
max_sessions = 128
persistence = true
```

Top-level keys such as `manifest_port = 8792` must appear **before** any section.

### Environment variables

| Variable | Meaning |
|---|---|
| `ECHO_DATA_DIR` | Data root (default `~/.echo_agent`). |
| `ECHO_LLM_BASE_URL` / `ECHO_LLM_MODEL` / `ECHO_LLM_API_KEY` | Nemotron/OpenAI-compatible endpoint. |
| `ECHO_LLM_RETRIES` | Bounded retries on HTTP 429/5xx. |
| `ECHO_DISCORD_TOKEN` / `ECHO_TELEGRAM_TOKEN` | Enable the respective gateway. |
| `ECHO_MANIFEST_PORT` | Manifest server port (default `8790`). |
| `ECHO_MAX_CONTEXT_TOKENS` | Prompt budget for pruning. |
| `ECHO_BUS_SOCKET` / `ECHO_ENABLE_BUS=1` | Rust bus bridge. |
| `ECHO_ALLOW_SHELL=1` | Opt-in for `shell_exec`. |
| `ECHO_OWNER_USER_IDS` | Comma-separated owner IDs (Telegram/Discord). |
| `ECHO_MEMORY_WRITE_APPROVAL` | Gate curated-memory writes behind `/memory approve`. |
| `ECHO_ENABLE_EMBEDDINGS` / `ECHO_EMBEDDINGS_BACKEND` / `ECHO_EMBEDDING_MODEL_DIR` | Embedding backend selection. |
| `ECHO_LEARNING_ENABLED` / `ECHO_AUTO_APPLY_IF_TESTS_PASS` | Learning loop and patch auto-apply. |
| `ECHO_SERVER_OPS_ENABLED` / `ECHO_SUPERVISOR_ENABLED` | Server-ops / supervisor toggles. |

### Data layout

```
~/.echo_agent/
├── config.toml          # optional (mode 0600 — never committed)
├── sessions.json        # per-DM/group/thread chat history (bounded)
├── echo-agent.pid       # background daemon pid
├── echo-agent.log       # daemon log
├── install.json         # installer metadata
├── gateways/
│   ├── pairing.json     # pending/approved DM senders (mode 0600)
│   └── telegram-offset  # durable long-poll offset
├── profiles/
│   └── owner.json       # single owner profile: name, preferences, pinned facts
│   └── u:<id>.json      # one isolated profile per non-owner sender (no leaks)
├── memory/
│   ├── USER.md          # owner identity (≤1375 chars) — injected in every prompt
│   ├── MEMORY.md        # agent notes (≤2200 chars) — injected in every prompt
│   ├── pending.json     # write-approval gate queue
│   ├── episodes.sqlite3 # episodic store + FTS5 session archive
│   └── vectors/         # semantic tier (chromadb / numpy fallback)
├── models/
│   └── all-MiniLM-L6-v2/# optional: model.onnx + tokenizer.json (ONNX embeddings)
├── skills/<name>/       # ingested + synthesized skills (SKILL.md)
├── backups/             # self-healing patch backups
└── agents.json          # supervisor registry {name, url, token}
```

### Security — personal information stays local

Your API keys, bot tokens, and pairing codes live **only** in
`~/.echo_agent/config.toml` (mode `0600`) and
`~/.echo_agent/gateways/pairing.json`. Both are listed in `.gitignore` and are
**never** pushed to GitHub.

Memory files hold only what the user or the review loop has explicitly saved —
never secrets, never invisible/control unicode. Non-owner senders get isolated
profiles under `profiles/u:<id>.json`; nothing they teach reaches the owner.

If you clone this repo on another machine, run `echo-agent setup` to create your
own local `config.toml` — never copy it from GitHub.

---

## Owner profile & curated memory

### Profile (`echo-agent profile show | forget --confirm`)

The owner profile (`profiles/owner.json`) holds name, preferences and pinned
facts, written atomically and loaded at plugin start. It is injected into every
system prompt as a USER PROFILE block, so `/reset` and daemon restarts never
forget it. CLI → `user_id="owner"`; Telegram/Discord IDs in `owner_user_ids`
resolve to the same profile; any other sender is isolated.

Extraction runs once per user turn (structured LLM call, temperature 0) plus
zero-LLM regex fast paths (`myself X`, `this is X`, `X here`, bare name when we
just asked). Corrections are last-write-wins; ephemera like "my question is…" is
never saved.

### Curated memory (`echo-agent memory pending|approve|reject`)

`USER.md` (≤1375 chars) and `MEMORY.md` (≤2200 chars) are snapshotted into every
system prompt with usage headers and `§` separators:

```
USER.md [42% — 580/1375]
- The user's name is Sanjeev.
- User preference: reply_style = terse
§
MEMORY.md [67% — 1474/2200]
- Server disk is on /dev/sda1; deploy via the release checklist.
- Lesson: check file existence before fs_read.
§
```

The `memory` tool (`{target: "user"|"memory", operations: [{action: add|patch|remove, content, old_text?}]}`) is
the **only** write path — a fact is remembered only when the call reports `ok`.
Patch/remove use unique-substring matching; exact duplicates are rejected; a
capacity overflow returns an error instructing one consolidating batch instead of
silently dropping content. With `memory_write_approval = true` writes queue in
`pending.json` until `/memory approve <id>`.

---

## Learning loop (Hermes-style)

After every non-trivial `task:completed`, a background review (debounced, token-capped,
deferred while the bus is busy) makes one bounded structured LLM call and writes
only through the memory/profile/skills tools:

- **memory**: facts, preferences and lessons → `MEMORY.md`.
- **skills**: a recurring 2+ step task synthesizes exactly one
  `skills/<name>/SKILL.md` (when-to-use, steps, tool sequence, pitfalls), visible
  in the next prompt; later corrections patch it, never duplicate.
- **self-healing**: recurring `PATCH:` lessons propose a code patch through the
  existing sandbox (pytest + atomic rollback + hot reload). Code patches require
  chat approval via `/patch list | approve <id> | reject <id>`; set
  `auto_apply_if_tests_pass = true` to opt out.

All review writes go through tool paths — the FakeProvider in the test suite
asserts this by recording review-schema calls separately from action scripts.

---

## Live activity streaming

`task:progress` events carry `{session_id, reply_channel, platform, kind, detail}`
with kinds `thinking`, `tool` (name + args gist), `tool_result`, `file_edit`
(diff gist ≤300 chars), `memory`, `skill`, `selfheal`, `supervisor`. Gateways
keep **one editable status message per task** (Telegram `editMessageText`,
Discord message edit), debounced ≥2s; the final answer is sent separately and the
status ends as a compact summary line. `gateway_progress = off | compact | verbose`
(default compact).

Hermes notices from plugin events: "💾 Memory updated", "🛠 Skill created",
"🩹 Patch applied", "🛰 Delegated to <agent>". Streaming failures are logged and
never break the task loop. CLI: the spinner shows tool names, arg gists and edit
diffs.

---

## Server-ops pack (`plugins/server_ops/`)

Read-only local metrics (`sys_metrics`, `disk_usage`, `process_list`,
`docker_ps` with auto-detect, `service_status`, `journal_tail`) plus
`service_restart` for allowlisted systemd units. Every action is audited to
episodic memory. Mutations need `/confirm <id>` (short TTL) unless the unit is
pre-approved. Cron health checks via the scheduler alert the owner chat when
disk >90%, RAM >90%, or an allowlisted unit is down.

---

## Supervisor mode

Registry in `agents.json`; tools `agents_list`, `agent_health(name)`,
`agent_delegate(name, task)` over JSON-RPC `/execute`; remote tools imported
namespaced through the universal client. Delegation retries with backoff, then
alerts. Heartbeat cron + daily digest to the owner chat; outcomes feed
per-agent reliability lessons into `MEMORY.md`. Delegation progress streams
through the Phase-5 status messages ("🛰 supervisor" events).

---

## CLI reference

| Command | Effect |
|---|---|
| `echo-agent setup` / `onboard` | Interactive first-run wizard (`--quick`, `--reset`, non-interactive flags). |
| `echo-agent agent [--headless]` | Alias of `start`: foreground daemon (REPL unless headless). |
| `echo-agent start [--headless]` | Foreground daemon. |
| `echo-agent doctor [--probe]` | Read-only health check with fix hints (extras, embeddings, LLM, tokens, agents). |
| `echo-agent run "task"` | One-shot task through the ReAct loop. |
| `echo-agent profile show` | Show the owner profile block injected into prompts. |
| `echo-agent profile forget --confirm` | Wipe the owner profile (confirmation-gated). |
| `echo-agent memory pending\|approve <id>\|reject <id>` | Curated-memory write-approval gate. |
| `echo-agent patch [list\|approve <id>\|reject <id>]` | Code-patch approvals (owner-only). |
| `echo-agent confirm <id>` | Confirm a pending server-ops mutation (owner-only). |
| `echo-agent gateway start\|stop\|restart\|status` | Manage the background daemon. |
| `echo-agent status` / `stop` / `restart` | Top-level daemon shortcuts. |
| `echo-agent skills add <url\|path>` | Ingest a skill repository. |
| `echo-agent tools` | List builtin tools. |
| `echo-agent plugins` | Dry manifest of builtin plugins. |
| `echo-agent channels list` | Show channel status (`channels` is an alias of `gateway`). |
| `echo-agent channels add --channel <telegram\|discord> [--token ...]` | Add/configure a channel. |
| `echo-agent channels remove --channel <telegram\|discord>` | Remove a channel token. |
| `echo-agent channels test --channel <telegram\|discord>` | Validate a bot token. |
| `echo-agent channels status [--probe]` | Status, optionally probing tokens. |
| `echo-agent pairing list [telegram\|discord] [--json]` | Pending/approved DM senders. |
| `echo-agent pairing approve <channel> <code> [--notify]` | Approve a DM pairing code. |
| `echo-agent pairing dismiss <channel> <code>` | Drop a pending code without approving. |
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
- Commands: `/ask`, `/status`, `/reset`, `/new`, `/model`, `/whoami`, `/activation`, `/help`, `/start`.
  Info commands (`/model`, `/status`, `/whoami`, `/help`, `/reset`, `/new`) work
  in DMs without pairing.
- Group behavior: optional sender filtering, @mention gating (or reply to bot, or
  native command), per-topic sessions.

### Discord (`discord.py`, optional extra)

- Gateway client with message content intent, guild/channel/sender allowlists.
- Text commands (`!ask`, `!status`, `!reset`, `!new`, `!model`, `!whoami`, `!activation`, `!help`)
  and native slash commands (`/ask`, `/status`, `/reset`, `/new`, `/model`, `/whoami`,
  `/activation`, `/help`). Info commands work in DMs without pairing.
- Reactions: 👀 = received and working, ✅ = task succeeded, ❌ = task failed.
- Join introductions, chunking, reply threading, reconnect/backoff.

### Access control defaults (OpenClaw-style)

- **DMs**: `pairing` by default. Unknown senders receive an 8-character code
  (no ambiguous `0/O/1/I`) and their message is **not** processed.
- **Groups/guilds**: `allowlist` by default; empty allowlist means closed.
- Codes are 8 uppercase characters (no ambiguous `0/O/1/I`), expire after one
  hour, are single-use, and are capped at 3 pending requests per channel.
- Approval grants **DM access only** — group/guild access stays config-driven.

### Sessions

Each DM, group, forum topic, Discord channel/thread gets its own session key
(`telegram:dm:<id>`, `telegram:chat:<id>:topic:<id>`,
`discord:guild:<g>:channel:<c>`). Tasks in one session serialize through a
per-session lock; history is bounded by `session_history_limit`; sessions persist
to `sessions.json`. `/reset` clears only the current chat and **archives it
before clearing** (searchable via `session_search`); that session's semantic-cache
entries are evicted. Semantic-cache entries are scoped per session.

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
| `task:incoming` | gateways | user text arrived (includes `session_id`, `user_id`) |
| `task:received` | orchestrator | task accepted into the loop |
| `task:progress` | orchestrator | live streaming: thinking/tool/tool_result/file_edit/memory/skill/selfheal/supervisor |
| `task:completed` | orchestrator | final answer produced (includes `session_id`, `user_id`, `steps`) |
| `session:reset` / `session:status` | gateways ↔ orchestrator | per-chat reset / message count |
| `tool:register` / `tool:unregister` | registry, skills, universal_client | tool schema lifecycle |
| `tool:executed` | tool registry | envelope with ok/error + timing |
| `memory:*` | memory plugin | tier writes and consolidation |
| `selfheal:*` | self-healing | patch lifecycle |
| `plugin:*` | plugin manager | lifecycle + per-plugin events |
| `skill:synthesized` / `skill:patched` / `skill:ingested` | skills / learning | skill created or patched |
| `learning:patch_proposed` / `learning:patch_approved` | learning loop | sandbox patch awaiting/approved |
| `server:alert` | server_ops | disk/RAM/unit breach → owner chat |
| `supervisor:heartbeat` / `supervisor:digest` / `supervisor:alert` / `supervisor:delegated` | supervisor | agent supervision |

---

## Project layout

```
echo-agent/
├── Cargo.toml
├── CMakeLists.txt
├── install.sh / uninstall.sh     # one-line install + standalone cleanup
├── pyproject.toml
├── uv.lock
├── README.md / MANUAL_TEST.md
├── config.example.toml           # Nemotron API example
├── scripts/lint.sh               # shellcheck + ruff + pytest
├── src/
│   ├── rust_bus/main.rs
│   ├── cpp_memory/index.cpp
│   └── python_core/
│       ├── main.py               # CLI + kernel bootstrap
│       ├── setup_wizard.py        # first-run onboarding
│       ├── doctor.py              # read-only health checks
│       ├── uninstall.py           # safety-railed total uninstall
│       ├── lifecycle.py           # shared paths, metadata, config merge
│       ├── orchestrator.py        # ReAct loop, pruning, cache, sessions, streaming
│       ├── core/                 # event_bus, plugin_manager, interfaces, config
│       └── plugins/
│           ├── memory/
│           │   ├── plugin.py     # profile store, extraction, curated + recall tools
│           │   ├── profile.py    # ProfileStore (owner.json + isolated profiles)
│           │   ├── curated.py    # USER.md / MEMORY.md bounded stores
│           │   ├── learning.py   # post-turn review, skill synthesis, patch proposals
│           │   ├── embeddings.py # ONNX MiniLM + hash fallback
│           │   ├── storage.py    # WM / EM (SQLite+FTS5) / SM
│           │   └── consolidator.py # dedup, decay, eviction
│           ├── self_healing/
│           ├── skills/
│           ├── server_ops/       # metrics, services, health, confirmations
│           ├── supervisor/       # registry, health, delegation, heartbeat, digest
│           ├── llm/
│           ├── web/
│           └── gateways/         # base, pairing, discord, telegram
├── tests/                        # 254-test full-kernel suite
└── docs/
```

---

## Development

Clone the repo when you want to hack on the agent itself:

```bash
git clone https://github.com/the-sanjeevkumar/echo-agent.git
cd echo-agent
uv sync                     # core deps (Telegram works out of the box)
uv sync --extra discord     # optional Discord support
uv sync --extra embeddings  # ONNX runtime + tokenizers for real embeddings
uv run echo-agent run "list the files in this repo"
uv run pytest tests/ -q -p no:cacheprovider
bash scripts/lint.sh        # shellcheck (via uvx if needed) + ruff + pytest
```

---

## Optional native layers

- **C++ cognitive engine**: `cmake -B build -S . && cmake --build build`
  (falls back to NumPy store).
- **Accelerated deps**: `uv sync --extra native` (tiktoken, chromadb).
- **Embeddings**: `uv sync --extra embeddings` (onnxruntime + tokenizers for
  ONNX all-MiniLM; falls back to deterministic hash embeddings).
- **Web access**: Scrapling is a core dependency; only browser binaries need the
  one-time install.

Extras: `native`, `discord`, `telegram` (empty: Telegram uses plain `httpx`),
`embeddings`, `all` — e.g. `uv sync --extra all`.

---

## Testing

```bash
uv run pytest tests/ -q -p no:cacheprovider
```

The full-kernel fixture builds the *real* system (only the LLM is swapped for
`FakeProvider`), so plugin wiring, gateways, sessions, and access control are
tested end-to-end. The suite covers memory tiers, consolidation, profile and
curated stores, FTS5 recall, learning loop, skill synthesis, self-healing,
streaming, server-ops, supervisor, web safety/fetch backends, orchestrator loop,
terminal UI, config precedence, and the Rust bridge protocol codec.
