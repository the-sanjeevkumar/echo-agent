# echo-agent

A **local-first, self-improving, event-driven AI agent** — a polyglot system
(Python + Rust + C++) built as a **microkernel with dynamically-loaded plugins**.
Everything runs on your machine: memory, skills, self-modification, and any
OpenAI-compatible LLM endpoint (including `ollama serve`). Nothing is sent
anywhere except to the LLM you configure.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              echo-agent CLI                              │
│         run · start (daemon+REPL) · skills add · tools · bus             │
└───────────────┬──────────────────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────────────────┐
│  PYTHON MICROKERNEL  (src/python_core)                                   │
│  ┌────────────┐   ┌──────────────┐   ┌─────────────────────────────┐     │
│  │ EventBus   │◄──┤ PluginManager│──►│ ToolRegistry (live schemas) │     │
│  │ (pub/sub)  │   │ + hot reload │   └─────────────────────────────┘     │
│  └─────┬──────┘   └──────┬───────┘                                        │
│        │      ┌──────────┴───────────────────────────────────┐           │
│        │      │ Orchestrator: ReAct loop · tiktoken pruning  │           │
│        │      │ · semantic cache · structured (schema) calls │           │
│        │      └──────────────────────────────────────────────┘           │
└────────┼─────────────────────────────────────────────────────────────────┘
         │ plugins (strict Protocols, loaded at runtime)
┌────────▼─────────────────────────────────────────────────────────────────┐
│  memory      dual-process memory + idle consolidation                    │
│  self_healing  sandboxed self-modification with atomic rollback           │
│  skills      GitHub → live tools, no restart                             │
│  llm         OpenAI-compatible provider (local ollama, vLLM, …)          │
│  gateways    Discord / Telegram bridges                                  │
│  universal_client + manifest_server  HTTP/JSON-RPC agent protocol        │
└────────┬─────────────────────────────────────────────────────────────────┘
         │ optional native layers
┌────────▼─────────────────────────────┐ ┌────────────────────────────────┐
│  RUST echo-bus (tokio JSON-RPC bus)  │ │  C++ cognitive engine          │
│  unix socket / TCP, cross-process    │ │  (pybind11: vector index +     │
│  pub/sub for sub-agent swarms        │ │   cause-graph; numpy fallback) │
└──────────────────────────────────────┘ └────────────────────────────────┘
```

| Module | Spec | Where |
|---|---|---|
| 1 · Dual-process memory | working / episodic / semantic tiers, idle-time consolidation | `src/python_core/plugins/memory/` |
| 2 · Self-healing | sandboxed patch validation, pytest gate, atomic rollback, hot reload | `src/python_core/plugins/self_healing/` |
| 3 · Skill acquisition | GitHub/local repo → analyzed → wrapped → hot-registered tools | `src/python_core/plugins/skills/` |
| 4 · Token efficiency | tiktoken pruning, semantic cache, JSON-schema outputs, delta diffs | `src/python_core/orchestrator.py` |
| 5 · Universal agent protocol | `/manifest.json` + JSON-RPC `/execute`, manifest-first ingestion | `src/python_core/inter_agent/` |
| 6 · Gateways & CLI | Discord/Telegram bridges, `echo-agent` CLI | `src/python_core/plugins/gateways/`, `src/python_core/main.py` |

---

## Quick start

Requirements: **Python ≥ 3.10** (developed and tested on 3.14). One optional
tool: [uv](https://docs.astral.sh/uv/) for dependency management (install with
`curl -LsSf https://astral.sh/uv/install.sh | sh`). Optional native toolchains:
Rust for `echo-bus`, a C++ compiler + CMake for the native engine.

```bash
# 1 · install (uv creates .venv, resolves uv.lock, installs echo-agent editable;
#     dev tooling comes from the default `dev` dependency-group)
uv sync

# 1b · one-time headless-browser install (renders JS-heavy sites; optional but
#      recommended — plain HTTP/Scrapling fetching works without it)
uv run scrapling install
# if that fails at the system-deps step (common in sandboxes), the browser alone
# is usually enough:
# python -m patchright install chromium

# 2 · point it at a local model server (ollama is the default target)
ollama serve &            # any OpenAI-compatible server works
ollama pull llama3.1

# 3 · run one task through the full agent loop
uv run echo-agent run "summarize the files in ./notes and write notes/summary.md"

# 4 · or start the daemon: plugins + manifest server + interactive REPL
uv run echo-agent start
```

Classic-pip equivalent (works identically):
`python -m venv .venv && source .venv/bin/activate && pip install -e . && pip install --group dev`
(pip ≥ 25.1 for `--group`; on older pips install `pytest pytest-asyncio` manually).

Without an LLM server the kernel still boots and every subsystem is testable —
the provider simply reports connection failures (see
[Configuration](#configuration) for `ECHO_LLM_BASE_URL`).

## CLI reference

| Command | Effect |
|---|---|
| `echo-agent run "task"` | One-shot task through the ReAct loop. On a TTY: a small spinner, the answer, and a compact step line (`✓ fs_read · ✗ web_fetch`). Piped output (or `--json`) prints a JSON envelope for scripting. |
| `echo-agent start [--headless]` | Boots all plugins, the manifest server, optional gateways, then an interactive REPL (or runs until SIGINT). |
| `echo-agent skills add <url\|path>` | Ingest a skill repository immediately (git URL or local path). |
| `echo-agent tools` | Lists builtin tools with live descriptions. |
| `echo-agent plugins` | Dry manifest of the builtin plugin set (including enabled gateways). |
| `echo-agent pairing list [telegram\|discord] [--json]` | Shows pending DM pairing codes and approved senders. |
| `echo-agent pairing approve <channel> <code>` | Approves a DM pairing code from a running or stopped gateway. |
| `echo-agent pairing revoke <channel> <sender_id>` | Revokes a previously approved sender. |
| `echo-agent gateway list` | Shows gateway token/policy status. |
| `echo-agent gateway configure <telegram\|discord> [--token ...]` | Interactively or non-interactively configure a gateway and write `config.toml`. |
| `echo-agent gateway test <telegram\|discord>` | Validate a configured bot token against the platform API. |
| `echo-agent gateway remove <telegram\|discord>` | Remove a gateway token from config. |
| `echo-agent channels list|status [--probe]` | OpenClaw-style channel management (alias for `gateway`). |
| `echo-agent channels add --channel <telegram\|discord> [--token ...]` | Add or update a channel. |
| `echo-agent channels remove --channel <telegram\|discord>` | Remove a channel token. |
| `echo-agent channels test --channel <telegram\|discord>` | Validate a channel token. |
| `echo-agent gateway start|stop|restart|status` | Manage the background daemon. |
| `echo-agent status` / `stop` / `restart` | Top-level daemon shortcuts. |
| `echo-agent bus` | Launches the Rust `echo-bus` (built binary, or `cargo run` fallback). |
| `echo-agent version` | Prints the version. |

Global flags: `--config path/to/config.toml`, `--verbose/-v`. Invoke as
`uv run echo-agent …` (uv manages the venv), or the bare `echo-agent …` inside
an activated venv.

The REPL is deliberately quiet: routine INFO logs (including `httpx` request
lines) are suppressed, and a braille spinner shows `thinking` / `tool: <name>`
while work is in flight. Use `-v` for timestamped debug output. Animations turn
themselves off automatically when stdout is not a TTY or `NO_COLOR` is set.

## Configuration

State lives under a single data root — default `~/.echo_agent` — so the agent is
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

| Setting / variable | Meaning |
|---|---|
| `ECHO_DATA_DIR` | Data root (default `~/.echo_agent`). |
| `ECHO_LLM_BASE_URL` / `ECHO_LLM_MODEL` / `ECHO_LLM_API_KEY` | LLM endpoint settings. |
| `ECHO_LLM_RETRIES` | Bounded retries on HTTP 429/5xx (default 3; honors `Retry-After`, capped at 5 s). |
| `web_allowed_domains` / `web_blocked_domains` | Domain allowlist (empty ⇒ all public hosts) / blocklist (suffix match, wins). |
| `web_max_chars` / `web_timeout` | Page content bound (default 8000) and per-fetch timeout (default 30 s). |
| `web_rate_limit_per_min` | Per-domain fetch budget to stop hammering loops (default 30/min). |
| `web_auto_stealth` | Auto-retry JS-only pages with the headless browser (default `true`). |
| `fs_allowed_roots` | Roots the filesystem tools may touch (empty ⇒ project root + data dir + cwd + temp). |
| `ECHO_MAX_CONTEXT_TOKENS` | Prompt budget for pruning. |
| `ECHO_MANIFEST_PORT` | Manifest server port (default `8790`). |
| `ECHO_BUS_SOCKET` | echo-bus unix socket path (default `<data_dir>/echo-bus.sock`). |
| `ECHO_ENABLE_BUS=1` | Start the daemon with the Rust-bus bridge enabled. |
| `ECHO_BUS_BIN` | Path to an `echo-bus` binary (default `target/release/echo-bus`). |
| `ECHO_DISCORD_TOKEN` / `ECHO_TELEGRAM_TOKEN` | Enables the respective gateway plugin when set. |
| `discord_*` / `telegram_*` | Per-platform DM/group policy, allowlists, guild/chat IDs, mention gating (see the TOML above). |
| `gateway_reply_to_mode` | Native reply threading: `off`, `first` (first chunk), or `all`. Agents can also emit `[[reply_to_current]]` / `[[reply_to:<id>]]`. |
| `session_history_limit` / `max_sessions` / `session_persistence` | Per-chat context size, LRU session cap, and disk persistence. |
| `ECHO_AGENT_TOKEN` | If set, `/manifest.json` and `/execute` require `Authorization: Bearer <token>`. |
| `ECHO_ALLOW_SHELL=1` | Opt-in switch for the `shell_exec` tool (off by default). |

Data layout under `ECHO_DATA_DIR`:

```
~/.echo_agent/
├── config.toml          # optional, see above
├── sessions.json        # per-DM/group/thread chat history (bounded)
├── gateways/
│   ├── pairing.json     # pending/approved DM senders (mode 0600)
│   └── telegram-offset  # durable long-poll offset
├── memory/
│   ├── episodes.sqlite3 # episodic store (append-only events)
│   └── vectors/         # semantic tier (chromadb, or numpy fallback)
└── skills/<name>/       # ingested skill wrappers (live-imported)
```

## Project layout

```
echo-agent/
├── Cargo.toml                    # rust workspace: the echo-bus binary
├── CMakeLists.txt                # optional C++ cognitive engine
├── pyproject.toml                # python packaging + pytest config
├── uv.lock                       # uv lockfile (exact dependency resolution)
├── src/
│   ├── rust_bus/main.rs          # tokio actor-model JSON-RPC bus (unix socket / TCP)
│   ├── cpp_memory/index.cpp      # pybind11: VectorIndex + CauseGraph
│   └── python_core/              # the microkernel (single top-level package)
│       ├── main.py               # CLI + kernel bootstrap (build_kernel)
│       ├── orchestrator.py       # ReAct loop, pruning, semantic cache
│       ├── core/                 # event_bus · plugin_manager · interfaces
│       │                         # tool_registry · config · utils
│       ├── plugins/
│       │   ├── memory/           # storage (sqlite) + consolidator + plugin
│       │   ├── self_healing/     # sandbox + patch_manager + plugin
│       │   ├── skills/           # ingester (repo → live tools)
│       │   ├── llm/              # openai_compat provider
│       │   └── gateways/        # base · pairing · discord · telegram
│       └── inter_agent/          # manifest_server · universal_client · rust_bridge
└── tests/                        # full-kernel + gateway/session test suite
```

The microkernel knows only the **Protocols** in `core/interfaces.py`
(`Plugin`, `MemoryPlugin`, `ToolPlugin`, `GatewayPlugin`, `LLMProvider`,
`SelfHealingPlugin`, `SkillPlugin`). Every feature above is a plugin that could
be deleted, replaced, or hot-swapped without touching the kernel.

---

## Module guide

### 1 · Dual-process memory (`plugins/memory/`)

Modeled loosely on Kahneman's System-1/System-2:

* **Working memory** — bounded ring buffer (`wm_capacity`), per-role salience;
  feeds the orchestrator prompt builder every turn.
* **Episodic memory** — append-only SQLite journal of *causal events*
  (task → action → outcome), queryable by recency/salience.
* **Semantic memory** — vector store over consolidated facts. Uses **chromadb**
  when installed, otherwise a pure-NumPy local store — embeddings come from a
  built-in hashed bag-of-words (`hash_embedding`), so retrieval works with zero
  network dependencies. Swap in a stronger embedding by implementing the
  `embedding` protocol in `core/utils.py`.
* **Consolidation** — after `consolidation_idle_seconds` of bus idleness the
  consolidator summarizes recent episodes and promotes stable facts into the
  semantic tier (`memory:consolidated`).

### 2 · Self-healing (`plugins/self_healing/`)

The agent can modify **its own source**:

1. A proposed patch arrives as a **unified diff** (`fs_apply_diff` or the
   self-healing tool call). Whole-file rewrites are discouraged — diffs are
   reviewable, cheap, and reversible.
2. `patch_manager.py` parses/validates the diff (offset-tolerant application,
   strict rejection on mismatch — malformed hunks never touch disk) and writes
   **atomically** (temp file + `os.replace`).
3. `sandbox.py` runs the test suite in a subprocess against the patched tree.
   Failure ⇒ automatic rollback (`selfheal:rolledback`), success ⇒ keep patch,
   and a **hot reload** re-imports the changed module into the running kernel
   (`selfheal:hotreload`).
4. Bus trail: `selfheal:applied` / `selfheal:rejected` / `selfheal:rolledback`.

`plugin_manager.hot_reload` purges stale bytecode, invalidates import caches
and deletes the old registration before re-import — the reload path is exactly
what makes live self-modification safe.

### 3 · Skill acquisition (`plugins/skills/`)

`echo-agent skills add <url-or-path>` runs the three-phase pipeline:

* **Phase 1 · deep analysis** — shallow-clone the repo, map the tree, read the
  README/entry files; with an LLM registered, one structured call proposes
  candidate tools.
* **Phase 2 · manifest priority** — a repo-root `agent_manifest.json` or
  `skill.md` **overrides** LLM guesses (see [Skill manifests](#skill-manifests)).
* **Phase 3 · dynamic registration** — standardized wrapper modules are
  generated under `skills/<name>/`, imported, and hot-registered through
  `tool:register` bus events. The new tools are callable in the *current*
  process, no restart, and immediately visible in the external manifest.

Executable tool kinds: `entry` (`{module, function}` imported from the cloned
repo) and `command` (subprocess template with `{placeholder}` substitution).
Knowledge-only skills are stored to semantic memory so they still shape future
recall.

### 4 · Token efficiency (`orchestrator.py`)

* **ReAct loop** (`max_steps=5`): the model answers with a JSON **action schema**
  (`think` / `tool` / `final`); tool results are appended and the loop repeats,
  bounded.
* **Context pruning** — exact `tiktoken` counting when available (else a
  char-based estimator); older tool results are summarized to hold the prompt
  under `max_context_tokens`.
* **Semantic cache** — before an LLM call, the normalized prompt is embedded
  and compared (cosine ≥ `cache_similarity_threshold`) against recent
  successful answers; hits skip the LLM entirely and are reported as
  `"cached": true`.
* **Structured outputs** — provider calls accept a JSON schema
  (`provider.complete(messages, schema=…)`), enforced/extracted by
  `core.utils.extract_json`, so parsing never guesses.
* **Delta-first editing** — `fs_apply_diff` sends diffs, not rewrites
  (see module 2).

### 5 · Universal agent protocol (`inter_agent/`)

Two complementary halves:

* **Server** — `manifest_server.py` (FastAPI + uvicorn) exposes:
  * `GET /manifest.json` — identity, capabilities, **live tool schemas**
    (a skill ingested seconds ago is instantly visible), auth hints, features.
  * `POST /execute` — JSON-RPC 2.0; `method` = tool name, `params` = tool args.
    Unknown tool ⇒ JSON-RPC error `-32601`; other failures ⇒ `-32000`.
  * `GET /health` — liveness.

  ```bash
  curl -s localhost:8790/manifest.json | python -m json.tool | head
  curl -s -X POST localhost:8790/execute \
       -H 'content-type: application/json' \
       -d '{"jsonrpc":"2.0","id":1,"method":"fs_list","params":{"dir":"."}}'
  ```

* **Client** — `universal_client.py` is a plugin that fetches *another* agent's
  manifest, validates it (`echo-agent-manifest/1`), and registers each of its
  remote tools locally — so remote agents become plain tools in the registry
  and the orchestrator can call them like any local capability. No shared SDK,
  no codegen: manifest + JSON-RPC is the whole contract.

### 6 · Gateways & CLI

`plugins/gateways/base.py` defines the shared gateway contract and behavior:
lifecycle (`start(ctx)` / `stop()`), reply routing, access decisions, chunking,
native reply tags, typing indicators, and per-chat session keys. Platform
adapters stay thin:

* **Telegram** (`telegram.py`) uses plain `httpx` long polling — no
  `python-telegram-bot` dependency. It learns the bot identity with `getMe`,
  registers a native command menu, persists its update offset, and reconnects
  with exponential backoff.
* **Discord** (`discord.py`) uses the official gateway client with message
  content intent, guild/channel/sender allowlists, text commands (`!ask`), and
  native slash commands (`/ask`, `/status`, `/reset`, `/whoami`,
  `/activation`, `/help`).

**Access control defaults mirror OpenClaw:** DMs require explicit **pairing**,
while groups/guilds are closed until their IDs are allowlisted. Unknown DM
senders receive an 8-character code (no ambiguous `0/O/1/I`) and their message
is *not* processed. Approve from the host while the daemon keeps running:

```bash
echo-agent pairing list telegram
echo-agent pairing approve telegram ABCD2345
echo-agent pairing revoke telegram 123456789
```

Group behavior: an allowed chat can optionally filter senders with
`group_allow_from`, and by default requires an @mention (or a reply to the bot,
or a native command). `/activation always` disables the mention requirement for
that chat until restart; `/activation mention` restores it. Unauthorized group
traffic is ignored rather than answered, so the bot does not spam strangers.

> **Platform prerequisites.** Telegram bots must have *Group Privacy* disabled
> in BotFather (or be a group admin) to see ordinary group messages; commands
> still work with privacy enabled. Discord requires the **Message Content**
> privileged intent in the Developer Portal and the `applications.commands`
> scope for slash commands.

**Sessions** are owned by the orchestrator, not the gateways: each DM, group,
forum topic, Discord channel/thread gets its own key (`telegram:dm:<id>`,
`telegram:chat:<id>:topic:<id>`, `discord:guild:<g>:channel:<c>`). Tasks in one
session serialize through a per-session lock, history is bounded by
`session_history_limit`, and sessions persist to `sessions.json` so restarts do
not lose context. `/reset` clears only the current chat. Semantic-cache entries
are scoped per session, so one chat cannot receive another chat's answer.

Outbound replies are chunked at platform limits (Telegram 4 000, Discord 1 900),
link previews are suppressed where configured, and reply threading is controlled
by `gateway_reply_to_mode` plus explicit agent tags (`[[reply_to_current]]`,
`[[reply_to:<message-id>]]`). New transports (Slack, IRC, Matrix…) only need to
translate inbound messages and implement `send_reply`; the shared controls come
for free.

## Internet access (`plugins/web/`)

Three tools are registered by the `web` plugin and available to the
orchestrator, gateways, and manifest clients:

| Tool | Purpose |
|---|---|
| `web_fetch` | Fetch a page → clean text/markdown, status, title (bounded). |
| `web_extract` | Fetch + CSS selector, e.g. `a::attr(href)`, `h1::text`. |
| `web_links` | Deduplicated, capped link list for navigation. |

**Backend ladder** (`fetcher.py`), auto-detected at import time:

1. **Scrapling `Fetcher`** — fast HTTP with browser TLS impersonation.
2. **Scrapling `StealthyFetcher`** — headless Chromium for Cloudflare/JS-heavy
   pages. Used automatically when a normal fetch returns a JavaScript shell
   (`web_auto_stealth`, default on), or explicitly with `stealth=true`.
3. **Plain `httpx`** — always-available fallback when Scrapling is absent.

Browser binaries are a one-time install:

```bash
uv run scrapling install          # browsers + system deps
python -m patchright install chromium   # browser-only fallback (sandboxes)
```

**Safety** is enforced before any bytes leave the process:

* `http`/`https` only; credentials-in-URL rejected.
* Every resolved IP must be globally routable — loopback, private, link-local,
  and reserved ranges are blocked (SSRF, cloud metadata, DNS rebinding).
* Optional allowlist / blocklist (suffix match; blocklist wins).
* Per-domain sliding-window rate limit (default 30 fetches/min).
* Output is truncated before it reaches the LLM (default 8 000 chars).
* Web failures are returned as data (`{"ok": false, "error": …}`), never raised
  into the agent loop.

```bash
uv run echo-agent run "fetch https://example.com and summarise it"
```

## The Rust bus (`echo-bus`)

`src/rust_bus/main.rs` is a tokio **actor-model broker**: one owning actor
holds all subscriptions; connections are newline-delimited JSON-RPC frames over
a unix socket (TCP fallback). Multi-line output of `echo-agent bus`:

```console
$ echo-agent bus
launching: target/release/echo-bus --socket /home/you/.echo_agent/echo-bus.sock
INFO echo_bus: echo-bus broker actor online
INFO echo_bus: echo-bus listening (unix) socket=/home/you/.echo_agent/echo-bus.sock
```

Protocol (one JSON object per line):

```jsonc
{"jsonrpc":"2.0","id":1,"method":"subscribe","params":{"topic":"agent:*"}}
{"jsonrpc":"2.0","id":2,"method":"publish","params":{"topic":"agent:hello","payload":{"msg":"hi"}}}
// responses: {"id":1,"jsonrpc":"2.0","result":{"subscribed":true}}
//            {"id":2,"jsonrpc":"2.0","result":{"delivered":1}}
// push:      {"jsonrpc":"2.0","method":"event","params":{"topic":"agent:hello","payload":{…}}}
```

Topics are glob-matched (`agent:*`). The Python side connects through
`inter_agent/rust_bridge.py`: Rust events are re-published on the Python
EventBus under `bus:<topic>`, local topics (`task:*`, `tool:executed`) are
mirrored out, with auto-reconnect. Set `ECHO_ENABLE_BUS=1` on `echo-agent start`
to enable it. Purpose: sub-agent swarms running as *separate processes* share
one high-throughput bus while the in-process Python bus remains the default.

Build it:

```bash
cargo build --release          # → target/release/echo-bus
```

> **No C toolchain installed?** `cargo` needs a C linker for the glibc target.
> A self-contained one can be borrowed from the `ziglang` PyPI package:
> `uv pip install --target /tmp/zigenv ziglang`, then a two-line wrapper
> (`exec env PYTHONPATH=/tmp/zigenv python3 -m ziglang cc "$@"`) passed as
> `RUSTFLAGS="-C linker=/path/to/wrapper" cargo build --release`.
> A benign `ignoring deprecated linker optimization setting "1"` warning is
> expected from that route.

## Event bus topics

All plugins communicate through these (publish/subscribe, awaited when
determinism matters):

| Topic | Emitted by | Meaning |
|---|---|---|
| `task:incoming` | gateways | user text arrived from a transport (includes `session_id`) |
| `task:received` | orchestrator | task accepted into the loop |
| `task:completed` | orchestrator | final answer produced (includes `session_id`) |
| `session:reset` / `session:status` | gateways ↔ orchestrator | per-chat reset request / message-count reply |
| `tool:register` / `tool:unregister` | registry, skills, universal_client | tool schema lifecycle (registry subscribes) |
| `tool:executed` | tool registry | envelope with ok/error + timing |
| `memory:working` / `memory:episodic` / `memory:semantic` | memory plugin | tier writes |
| `memory:consolidated` | consolidator | idle-time promotion happened |
| `selfheal:applied` / `selfheal:rejected` / `selfheal:rolledback` / `selfheal:hotreload` | self-healing | patch lifecycle |
| `plugin:started` / `plugin:stopped` / `plugin:reloaded` / `plugin:<name>` | plugin manager | lifecycle + per-plugin events |
| `skill:ingested` | skills | a repo became tools |
| `bus:<topic>` | rust bridge | events arriving from the Rust bus |

## Tool envelope semantics

Every tool call goes through the registry and returns a uniform envelope:

```json
{"ok": true,  "result": {…}, "tool": "fs_read", "elapsed_ms": 3}
{"ok": false, "error": "…",  "tool": "fs_read", "elapsed_ms": 1}
```

`envelope.ok` means *"the call executed"*; the tool's own success lives inside
`result` (e.g. `shell_exec` reports `{"ok": false, "returncode": 1}` inside a
successful envelope). The orchestrator and tests read accordingly.

---

## Skill manifests

Manifests beat LLM guessing. Drop either file at a repo root and ingestion is
deterministic.

**`agent_manifest.json`** — pure machine format:

```json
{
  "name": "text-tools",
  "summary": "Small local text utilities",
  "version": "1.0.0",
  "tools": [
    {
      "name": "count_words",
      "description": "Count words in a text file",
      "kind": "entry",
      "module": "tools.words",
      "function": "count_words",
      "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    },
    {
      "name": "strip_html",
      "description": "Strip HTML tags from a file",
      "kind": "command",
      "command": ["python", "tools/strip.py", "{path}"],
      "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
    }
  ]
}
```

**`skill.md`** — human-friendly: YAML-ish frontmatter, optional prose (stored
as semantic-memory knowledge), optional fenced `json` tools block:

~~~~markdown
---
name: text-tools
description: Small local text utilities
version: 1.0.0
---
Counts words, strips HTML. Free prose here becomes recallable knowledge.

```json
{"tools": [{"name": "count_words", "description": "Count words in a file",
            "kind": "entry", "module": "tools.words", "function": "count_words",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}]}
```
~~~~

Kinds: `entry` imports `{module}.{function}` from the ingested repo and calls
it as `fn(args)`; `command` runs a subprocess template with `{placeholder}`
substitution from the tool args; omitted kind + prose-only ⇒ `knowledge`.

## Writing a plugin

Implement the `Plugin` protocol from `core/interfaces.py` — that is the entire
kernel contract:

```python
"""my_plugin.py — drop into any directory listed in config.plugin_dirs."""
from python_core.core.event_bus import EventBus

class MyPlugin:
    name = "my_plugin"
    version = "0.1.0"
    capabilities = ["example"]

    async def start(self, ctx) -> None:          # ctx: KernelContext
        ctx.bus.subscribe("tool:executed", self._on_tool)
        ctx.services["my_plugin"] = self          # other plugins can find you

    async def stop(self) -> None:
        pass

PLUGIN = MyPlugin()
```

Discovery contract for `PluginManager.discover(dir)` (every `*.py` directly in
the directory, `_`-prefixed files skipped): expose **one of**

* `get_plugins() → sequence` (multiple plugins),
* `create_plugin() → instance`,
* a module-level `PLUGIN` instance.

Anything failing to start is logged and skipped — one broken plugin can never
kill the kernel. While the daemon runs, drop a fixed file into the plugin dir
and `plugin_manager.hot_reload(module_name)` re-imports it live. Structured
plugins extend the specific Protocols (`MemoryPlugin`, `ToolPlugin`,
`GatewayPlugin`, `LLMProvider`, `SelfHealingPlugin`, `SkillPlugin`) so the
kernel can type-check their capabilities.

## Optional native layers

* **C++ cognitive engine** — `src/cpp_memory/index.cpp` binds a
  brute-force-cosine `VectorIndex` and a `networkx`-backed-exportable
  `CauseGraph` via pybind11:

  ```bash
  cmake -B build -S . && cmake --build build   # needs a C++17 compiler + pybind11
  ```

  The Python memory plugin uses it when importable and silently falls back to
  the NumPy store otherwise — both pass the same test suite.
* **Accelerated deps** — `uv sync --extra native` adds tiktoken (exact token
  counts) and chromadb (persistent vector DB); everything degrades gracefully
  without them. Extras: `native`, `discord`, `telegram` (empty: Telegram uses
  plain `httpx`), `all` — e.g. `uv sync --extra all`.
* **Web access** — Scrapling is a core dependency (no extra needed); only the
  headless browser binaries require the one-time `uv run scrapling install`.

## Testing

```bash
uv run pytest tests/ -q
# 151 passed — kernel, memory tiers, consolidation (incl. lazy LLM resolution),
# diff round-trips, hot reload, skill ingestion (entry+command+knowledge),
# manifest server, universal client, gateway pairing/access/chunking, per-chat
# sessions + persistence, web safety/fetch backends, orchestrator loop with a
# scripted FakeProvider, terminal UI, config precedence, and the Rust bridge
# protocol codec.
```

The full-kernel fixture builds the *real* system (only the LLM is swapped for
`FakeProvider`), so plugin wiring is tested end-to-end, not mocked away.

## Design notes

* **Local-first**: state is files + SQLite under one directory; no cloud
  services; the only outbound calls are your LLM endpoint, skill-repo clones,
  and remote agents you explicitly registered.
* **Microkernel**: `main.py:build_kernel` is ~40 lines — bus, registry,
  context, plugin registration, start. Everything else is a plugin.
* **Interface segregation**: `core/interfaces.py` protocols are narrow; the
  orchestrator depends on `LLMProvider` + `ToolRegistry`, never on plugins.
* **Event-driven**: any subscriber can extend behavior — e.g. a future
  "cost accountant" plugin would just subscribe to `tool:executed` and
  `task:completed` — without a single line changed in the kernel.

MIT licensed. Contributions welcome — the fastest path is: implement a
Protocol, publish a manifest, let the tests gate the patch.
