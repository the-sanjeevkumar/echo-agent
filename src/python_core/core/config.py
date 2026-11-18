"""Local-first configuration.

All state lives under a single data root (default ``~/.echo_agent``) so the
agent is fully self-contained, portable, and trivially backed up. Configuration
can be provided via an optional TOML file (``~/.echo_agent/config.toml``) or
environment variables; environment always wins.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _default_data_dir() -> Path:
    env = os.environ.get("ECHO_DATA_DIR")
    return Path(env).expanduser() if env else Path.home() / ".echo_agent"


@dataclass(slots=True)
class EchoConfig:
    """Runtime configuration for the entire agent (single source of truth)."""

    # ── local-first storage ────────────────────────────────────────────────
    data_dir: Path = field(default_factory=_default_data_dir)

    # ── LLM backend (any OpenAI-compatible endpoint, incl. local servers) ──
    llm_base_url: str = "http://127.0.0.1:11434/v1"   # e.g. ollama serve
    llm_model: str = "llama3.1"
    llm_api_key: str = ""                              # optional for local servers
    llm_timeout: float = 120.0
    llm_retries: int = 3                               # bounded retries on 429/5xx
    available_models: list[str] = field(default_factory=list)  # discovered models from /v1/models

    # ── user profile (memory) ──────────────────────────────────────────────
    user_name: str = ""                                      # learned from conversation
    user_preferences: dict[str, str] = field(default_factory=dict)  # key → value preferences

    # ── token efficiency (MODULE 4) ────────────────────────────────────────
    max_context_tokens: int = 8192     # hard budget for the prompt window
    prune_target_ratio: float = 0.6    # prune when used tokens exceed this ratio
    enable_semantic_cache: bool = True
    cache_similarity_threshold: float = 0.93   # cosine similarity for cache hits

    # ── memory plugin (MODULE 1) ───────────────────────────────────────────
    consolidation_idle_seconds: float = 20.0    # idle trigger for consolidation
    wm_capacity: int = 64                        # working-memory ring capacity

    # ── IPC / native layers ────────────────────────────────────────────────
    bus_socket: str = ""                          # echo-bus unix socket path
    manifest_host: str = "127.0.0.1"
    manifest_port: int = 8790                     # GET /manifest.json server

    # ── gateways (MODULE 6) ────────────────────────────────────────────────
    discord_token: str = ""
    telegram_token: str = ""

    # Access control mirrors OpenClaw: DMs default to explicit pairing, groups
    # default to an allowlist (empty allowlist ⇒ groups are closed).
    discord_dm_policy: str = "pairing"          # pairing | allowlist | open | disabled
    telegram_dm_policy: str = "pairing"         # pairing | allowlist | open | disabled
    discord_allow_from: list[str] = field(default_factory=list)   # user IDs / usernames
    telegram_allow_from: list[str] = field(default_factory=list)  # numeric user IDs / @usernames
    discord_group_policy: str = "allowlist"     # allowlist | open | disabled
    telegram_group_policy: str = "allowlist"    # allowlist | open | disabled
    discord_group_allow_from: list[str] = field(default_factory=list)
    telegram_group_allow_from: list[str] = field(default_factory=list)
    discord_guilds: list[str] = field(default_factory=list)       # guild IDs allowed
    discord_channels: list[str] = field(default_factory=list)     # channel IDs (empty ⇒ all in guild)
    telegram_groups: list[str] = field(default_factory=list)      # chat IDs allowed
    discord_require_mention: bool = True
    telegram_require_mention: bool = True
    telegram_ignore_other_mentions: bool = True
    discord_allow_bots: bool = False
    discord_group_dm_enabled: bool = False
    discord_command_prefix: str = "!"
    discord_slash_commands: bool = True
    discord_suppress_embeds: bool = True
    discord_join_intro: bool = True
    discord_ignore_other_mentions: bool = True
    telegram_link_preview: bool = False
    gateway_reply_to_mode: str = "off"          # off | first | all
    gateway_typing: bool = True
    gateway_pairing_ttl: float = 3600.0         # pairing code lifetime (seconds)
    gateway_pairing_max_pending: int = 3        # per channel account

    # ── chat sessions (per DM / group / thread) ────────────────────────────
    session_history_limit: int = 40             # messages retained per session
    max_sessions: int = 128                     # LRU cap for live sessions
    session_persistence: bool = True            # persist chats to data_dir/sessions.json

    # ── self-healing (MODULE 2) ────────────────────────────────────────────
    sandbox_test_timeout: float = 300.0           # pytest timeout for patches

    # ── skills (MODULE 3) ──────────────────────────────────────────────────
    skills_dir: str = ""                          # defaults to <data_dir>/skills

    # ── web plugin (internet access) ───────────────────────────────────────
    web_allowed_domains: list[str] = field(default_factory=list)  # empty = all public hosts
    web_blocked_domains: list[str] = field(default_factory=list)  # suffix match, wins over allowed
    web_max_chars: int = 8000                     # page content bound fed to the LLM
    web_timeout: float = 30.0                     # per-fetch timeout
    web_rate_limit_per_min: int = 30              # per-domain fetch budget (anti-hammer)
    web_auto_stealth: bool = True                 # auto-retry JS-only pages in a headless browser

    # ── filesystem tool sandbox ────────────────────────────────────────────
    fs_allowed_roots: list[str] = field(default_factory=list)     # empty = safe defaults

    # ── self-modification target ───────────────────────────────────────────
    project_root: str = ""                        # agent's own repo; auto-detected if empty

    # ── plugin discovery ───────────────────────────────────────────────────
    plugin_dirs: list[str] = field(default_factory=list)

    # ── owner profile (MODULE 1 / PHASE 1) ─────────────────────────────────
    owner_user_ids: list[str] = field(default_factory=list)  # platform ids treated as the single owner

    # ── curated memory (MODULE 2 / PHASE 2) ────────────────────────────────
    memory_write_approval: bool = False  # gate `memory` tool writes behind /memory approve

    # ── embeddings (PHASE 3) ───────────────────────────────────────────────
    embeddings_backend: str = "auto"   # auto = ONNX MiniLM if installed, else hash
    embedding_model_dir: str = ""      # default: <data_dir>/models/all-MiniLM-L6-v2
    enable_embeddings: bool = True     # off = hash-only deterministic fallback

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir).expanduser()
        # Keep policy values in a known set; an invalid value must fail safe
        # (pairing for DMs, allowlist for groups) instead of silently opening up.
        dm_choices = {"pairing", "allowlist", "open", "disabled"}
        group_choices = {"allowlist", "open", "disabled"}
        for field_name, default, choices in (
            ("discord_dm_policy", "pairing", dm_choices),
            ("telegram_dm_policy", "pairing", dm_choices),
            ("discord_group_policy", "allowlist", group_choices),
            ("telegram_group_policy", "allowlist", group_choices),
        ):
            value = str(getattr(self, field_name) or default).strip().lower()
            setattr(self, field_name, value if value in choices else default)
        reply_mode = str(self.gateway_reply_to_mode or "off").strip().lower()
        self.gateway_reply_to_mode = reply_mode if reply_mode in {"off", "first", "all"} else "off"
        for list_field in (
            "discord_allow_from",
            "telegram_allow_from",
            "discord_group_allow_from",
            "telegram_group_allow_from",
            "discord_guilds",
            "discord_channels",
            "telegram_groups",
            "owner_user_ids",
        ):
            value = getattr(self, list_field)
            if isinstance(value, str):
                setattr(
                    self,
                    list_field,
                    [item.strip() for item in value.split(",") if item.strip()],
                )
        self.gateway_pairing_ttl = max(60.0, float(self.gateway_pairing_ttl))
        self.gateway_pairing_max_pending = max(1, int(self.gateway_pairing_max_pending))
        self.session_history_limit = max(2, int(self.session_history_limit))
        self.max_sessions = max(2, int(self.max_sessions))
        self.data_dir = Path(self.data_dir).expanduser()
        # Keep policy values in a known set; an invalid value must fail safe
        # (pairing for DMs, allowlist for groups) instead of silently opening up.
        dm_choices = {"pairing", "allowlist", "open", "disabled"}
        group_choices = {"allowlist", "open", "disabled"}
        for field_name, default, choices in (
            ("discord_dm_policy", "pairing", dm_choices),
            ("telegram_dm_policy", "pairing", dm_choices),
            ("discord_group_policy", "allowlist", group_choices),
            ("telegram_group_policy", "allowlist", group_choices),
        ):
            value = str(getattr(self, field_name) or default).strip().lower()
            setattr(self, field_name, value if value in choices else default)
        reply_mode = str(self.gateway_reply_to_mode or "off").strip().lower()
        self.gateway_reply_to_mode = reply_mode if reply_mode in {"off", "first", "all"} else "off"
        for list_field in (
            "discord_allow_from",
            "telegram_allow_from",
            "discord_group_allow_from",
            "telegram_group_allow_from",
            "discord_guilds",
            "discord_channels",
            "telegram_groups",
        ):
            value = getattr(self, list_field)
            if isinstance(value, str):
                setattr(
                    self,
                    list_field,
                    [item.strip() for item in value.split(",") if item.strip()],
                )
        self.gateway_pairing_ttl = max(60.0, float(self.gateway_pairing_ttl))
        self.gateway_pairing_max_pending = max(1, int(self.gateway_pairing_max_pending))
        self.session_history_limit = max(2, int(self.session_history_limit))
        self.max_sessions = max(2, int(self.max_sessions))

    # ── derived paths ──────────────────────────────────────────────────────
    @property
    def memory_dir(self) -> Path:
        return self.data_dir / "memory"

    @property
    def sqlite_path(self) -> Path:
        return self.memory_dir / "episodes.sqlite3"

    @property
    def vector_dir(self) -> Path:
        return self.memory_dir / "vectors"

    @property
    def resolved_skills_dir(self) -> Path:
        return Path(self.skills_dir) if self.skills_dir else self.data_dir / "skills"

    @property
    def resolved_project_root(self) -> Path:
        """The agent's own repository root (self-modification target).

        Auto-detected by walking up from this file until a directory holding
        pyproject.toml/Cargo.toml is found — the local-first repo layout.
        """
        if self.project_root:
            return Path(self.project_root).expanduser()
        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "pyproject.toml").is_file() or (parent / "Cargo.toml").is_file():
                return parent
        return here.parent

    @property
    def resolved_bus_socket(self) -> str:
        return self.bus_socket or str(self.data_dir / "echo-bus.sock")

    def ensure_dirs(self) -> None:
        """Create the local-first directory skeleton. Idempotent."""
        for p in (self.data_dir, self.memory_dir, self.vector_dir, self.resolved_skills_dir):
            p.mkdir(parents=True, exist_ok=True)

    # ── loading ────────────────────────────────────────────────────────────
    @classmethod
    def load(cls, config_file: str | os.PathLike[str] | None = None) -> "EchoConfig":
        """Build config from TOML file + env overrides (env wins).

        Recognized TOML sections mirror the dataclass fields, e.g.:
            [llm] base_url, model, api_key, timeout
            [memory] idle_seconds, wm_capacity
            [tokens] max_context_tokens, cache_similarity_threshold
        """
        values: dict[str, Any] = {}

        # 1. TOML file (config_file arg, else $ECHO_DATA_DIR/config.toml).
        path = Path(config_file or (Path(os.environ.get("ECHO_DATA_DIR", _default_data_dir())) / "config.toml"))
        if path.is_file():
            with open(path, "rb") as fh:
                raw = tomllib.load(fh)
            for section, fields in raw.items():
                if isinstance(fields, dict):
                    if section == "user":
                        if "name" in fields:
                            values["user_name"] = fields["name"]
                        for k, v in fields.items():
                            if k.startswith("pref__"):
                                pref_key = k[len("pref__"):]
                                prefs = values.setdefault("user_preferences", {})
                                if isinstance(prefs, dict):
                                    prefs[pref_key] = v
                    elif section in ("llm", "web", "discord", "telegram", "gateway", "session"):
                        # [llm] base_url → llm_base_url; [web] max_chars →
                        # web_max_chars; [discord] dm_policy → discord_dm_policy
                        # (fully-qualified keys are still accepted).
                        prefix = f"{section}_"
                        values.update({
                            k if k.startswith(prefix) else f"{prefix}{k}": v
                            for k, v in fields.items()
                        })
                    else:
                        values.update(fields)
                else:
                    values[section] = fields

        # 2. Environment overrides (flat, uppercase).
        env_map = {
            "ECHO_LLM_BASE_URL": "llm_base_url",
            "ECHO_LLM_MODEL": "llm_model",
            "ECHO_LLM_API_KEY": "llm_api_key",
            "ECHO_LLM_RETRIES": "llm_retries",
            "ECHO_DATA_DIR": "data_dir",
            "ECHO_BUS_SOCKET": "bus_socket",
            "ECHO_DISCORD_TOKEN": "discord_token",
            "ECHO_TELEGRAM_TOKEN": "telegram_token",
            "ECHO_MAX_CONTEXT_TOKENS": "max_context_tokens",
            "ECHO_MANIFEST_PORT": "manifest_port",
            "ECHO_USER_NAME": "user_name",
            "ECHO_OWNER_USER_IDS": "owner_user_ids",
            "ECHO_EMBEDDINGS_BACKEND": "embeddings_backend",
            "ECHO_EMBEDDING_MODEL_DIR": "embedding_model_dir",
            "ECHO_ENABLE_EMBEDDINGS": "enable_embeddings",
        }
        for env_key, field_name in env_map.items():
            if env_key in os.environ:
                raw_val = os.environ[env_key]
                anno = cls.__dataclass_fields__[field_name].type
                values[field_name] = (
                    int(raw_val) if "int" in str(anno)
                    else float(raw_val) if "float" in str(anno)
                    else raw_val
                )

        # 3. Env booleans (env_map leaves strings; cast them explicitly).
        for env_key, field_name in (
            ("ECHO_MEMORY_WRITE_APPROVAL", "memory_write_approval"),
            ("ECHO_ENABLE_EMBEDDINGS", "enable_embeddings"),
        ):
            if env_key in os.environ:
                values[field_name] = os.environ[env_key].strip().lower() in {"1", "true", "yes", "on"}

        # Alias TOML section names → dataclass field names.
        alias = {
            "timeout": "llm_timeout",
            "idle_seconds": "consolidation_idle_seconds",
            "session_max_sessions": "max_sessions",
            "llm_available_models": "available_models",
        }
        values = {alias.get(k, k): v for k, v in values.items()}

        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in values.items() if k in known})

    # ── config.toml persistence helpers ────────────────────────────────────
    def _config_path(self) -> Path:
        return Path(os.environ.get("ECHO_DATA_DIR", _default_data_dir())) / "config.toml"

    def _read_toml_text(self) -> str:
        path = self._config_path()
        if path.is_file():
            return path.read_text(encoding="utf-8")
        return ""

    def _write_toml_text(self, text: str) -> None:
        path = self._config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _upsert_toml_key(self, section: str, key: str, value: str) -> None:
        """Insert or replace `key = value` under `[section]` in config.toml."""
        text = self._read_toml_text()
        line = f'{key} = {value!r}'
        # Find the section header
        section_header = f"[{section}]"
        if section_header in text:
            # Replace within that section
            lines = text.split("\n")
            out = []
            in_section = False
            replaced = False
            for ln in lines:
                stripped = ln.strip()
                if stripped == section_header:
                    in_section = True
                    out.append(ln)
                    continue
                if in_section and stripped.startswith("["):
                    in_section = False
                if in_section and stripped.startswith(f"{key} ="):
                    out.append(line)
                    replaced = True
                    continue
                out.append(ln)
            if in_section and not replaced:
                # Section found but key not present; insert after section header
                for i, ln in enumerate(out):
                    if ln.strip() == section_header:
                        out.insert(i + 1, line)
                        break
            text = "\n".join(out)
        else:
            # Section doesn't exist; append it
            if text and not text.endswith("\n"):
                text += "\n"
            text += f"\n{section_header}\n{line}\n"
        self._write_toml_text(text)

    def save_user_profile(self) -> None:
        """Persist user_name and user_preferences to config.toml."""
        if self.user_name:
            self._upsert_toml_key("user", "name", self.user_name)
        for k, v in self.user_preferences.items():
            self._upsert_toml_key("user", f"pref__{k}", v)

    def save_llm_model(self, model_name: str) -> None:
        """Persist the chosen LLM model to config.toml."""
        self._upsert_toml_key("llm", "model", model_name)
        self.llm_model = model_name

    def save_available_models(self, models: list[str]) -> None:
        """Persist discovered model list to config.toml."""
        self._upsert_toml_key("llm", "available_models", models)
        self.available_models = models
