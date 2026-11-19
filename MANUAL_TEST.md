# echo-agent — manual verification

Exact steps proving each acceptance criterion. Run against a fresh data dir
(`ECHO_DATA_DIR=/tmp/echo-manual`) so nothing touches your real `~/.echo_agent`.

Prerequisites:
- `uv run echo-agent --version` works (installed or `uv sync` from the repo).
- LLM endpoint reachable: `ECHO_LLM_BASE_URL` + `ECHO_LLM_API_KEY` (e.g. NVIDIA
  Nemotron at `https://integrate.api.nvidia.com/v1`) or any OpenAI-compatible
  server. `echo-agent doctor --probe` should show the model list.
- Optional: Telegram bot token + Discord bot token for the platform checks.

## M1 — name survives /reset and a daemon restart

```bash
export ECHO_DATA_DIR=/tmp/echo-manual

# 1. teach the name via CLI
uv run echo-agent run "my name is Sanjeev"
# expect: the answer addresses you as Sanjeev

# 2. check the profile block is injected in prompts
uv run echo-agent profile show
# expect: USER PROFILE block containing "The user's name is Sanjeev."

# 3. /reset via CLI (start the daemon, then send /reset in REPL, or via gateway)
uv run echo-agent agent --headless &
# in another terminal, against the same ECHO_DATA_DIR:
uv run echo-agent run "/reset"
uv run echo-agent run "hello"
# expect: the agent still calls you Sanjeev (USER PROFILE block present)

# 4. restart the daemon
kill %1
uv run echo-agent agent --headless &
uv run echo-agent run "what is my name?"
# expect: "Sanjeev" — the name survived restart via profiles/owner.json

# 5. isolation: a different sender is isolated
uv run echo-agent run "my name is Stranger"   # user_id default = owner → overwrites!
# (for true isolation, drive via Telegram/Discord with a non-owner user id)
```

## M2 — corrections are last-write-wins

```bash
uv run echo-agent run "call me Alex"
uv run echo-agent run "what is my name?"
# expect: Alex (Sanjeev was corrected)
uv run echo-agent profile show
# expect: name = Alex only
```

## M3 — bare name after we ask

In the REPL (or a Telegram/Discord DM):
```
you: hello
agent: What should I call you?
you: Sanjeev
agent: Thanks, Sanjeev.
you: what is my name?
# expect: Sanjeev — bare-name rescue after we asked
```

## M4 — edits appear live in Telegram/Discord (streaming)

Start the daemon with a configured bot, then DM it:

Telegram:
```
you: /ask read pyproject.toml
bot: 👀 (eyes reaction on your message)
     🔄 working…         ← status message created
     ⚙ tool: fs_read(...)      ← edited in place as tools run
     ✓ fs_read: ok
     ✅ done · 1 step(s)      ← status finalized
     <final answer as a separate message>
```

Discord: same flow with one editable bot message per task, then the answer.

You can watch edits accumulate while the task runs; with `gateway_progress =
verbose` even `thinking` steps appear.

## M5 — `/forget` wipes the profile (confirmation-gated)

```bash
uv run echo-agent run "/forget"
# expect: asks to send /forget confirm within N minutes
uv run echo-agent run "/forget confirm"
# expect: "Your profile has been wiped"
uv run echo-agent run "what is my name?"
# expect: the agent no longer knows your name ("I don't know your name yet")
```

## M6 — server alert works (health check)

Configure `server_ops_units` (e.g. a real unit like `sshd.service`) and
`server_ops_alerts_reply_channel = "telegram:<your-chat-id>"`. Then:

Option A — simulate a breach by lowering the threshold:
```toml
server_ops_disk_threshold = 1.0   # anything ≥1% triggers
```
Restart the daemon, wait for the health cron (`every 5m`), or run:
```bash
uv run echo-agent run "run the server health check"
```
Expect a Telegram message: `🚨 WARNING: Disk usage N% >= 1.0%`.

Option B — stop an allowlisted unit (only in a safe sandbox):
```bash
systemctl stop nginx.service   # if allowlisted
# expect: 🚨 CRITICAL: systemd unit nginx.service is inactive
```

## M7 — delegation works (supervisor)

Register a remote agent speaking the manifest protocol (or run a second local
echo-agent on another port with its manifest server), then:

```bash
uv run echo-agent run "register agent worker at http://worker:8790"
# (uses agent_register via the LLM tool loop, or call the tool directly)
uv run echo-agent run "delegate to worker: summarize the system status"
# expect: 🛰 Delegated to worker → 🛰 worker completed the delegation → result
```

Failure path: point the agent URL at a dead port; after `supervisor_retries`
(2) attempts with backoff, expect:
```
🚨 CRITICAL: Agent worker delegation failed after 3 attempts: transport error: ...
```

## M8 — fresh install is complete by default

```bash
bash install.sh               # defaults to --extras all
echo-agent doctor
# expect: extras include native(tiktoken), native(chromadb), discord,
#         embeddings(onnxruntime+tokenizers), scrapling (when installed)
#         embeddings → hash unless model.onnx + tokenizer.json are present
uv run scrapling install      # optional: browser binaries for JS-only pages
echo-agent setup --quick      # configure LLM endpoint; done
```

## Acceptance checklist

- [ ] name survives `/reset`
- [ ] name survives daemon restart (profiles/owner.json)
- [ ] "call me Alex" corrects "Sanjeev" (last-write-wins)
- [ ] bare name right after "What should I call you?" is captured
- [ ] Telegram/Discord show one live-editing status message per task
- [ ] server alert fires on a simulated threshold breach
- [ ] delegation to a registered agent succeeds; failure alerts after retries
- [ ] fresh `install.sh` (default extras) installs everything
