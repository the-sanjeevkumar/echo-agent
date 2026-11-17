"""Config loading: section-style TOML keys map to prefixed dataclass fields."""

from __future__ import annotations

from python_core.core.config import EchoConfig


def test_web_section_maps_to_prefixed_fields(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[web]
max_chars = 1234
timeout = 12.5
auto_stealth = false
rate_limit_per_min = 7
allowed_domains = ["example.com"]
blocked_domains = ["evil.example"]
""",
        encoding="utf-8",
    )
    cfg = EchoConfig.load(path)

    assert cfg.web_max_chars == 1234
    assert cfg.web_timeout == 12.5
    assert cfg.web_auto_stealth is False
    assert cfg.web_rate_limit_per_min == 7
    assert cfg.web_allowed_domains == ["example.com"]
    assert cfg.web_blocked_domains == ["evil.example"]


def test_llm_section_timeout_alias_still_works(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[llm]\nbase_url = \"http://localhost:1234/v1\"\ntimeout = 9.5\n", encoding="utf-8")
    cfg = EchoConfig.load(path)

    assert cfg.llm_base_url == "http://localhost:1234/v1"
    assert cfg.llm_timeout == 9.5


def test_gateway_and_session_sections(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[discord]
dm_policy = "allowlist"
allow_from = ["123"]
group_policy = "open"
guilds = ["456"]
channels = ["789"]
require_mention = false
command_prefix = "$"

[telegram]
dm_policy = "open"
group_policy = "allowlist"
groups = ["-100123"]
require_mention = false
ignore_other_mentions = false
link_preview = true

[gateway]
reply_to_mode = "first"
typing = false
pairing_ttl = 120
pairing_max_pending = 5

[session]
history_limit = 12
max_sessions = 7
persistence = false
""",
        encoding="utf-8",
    )
    cfg = EchoConfig.load(path)

    assert cfg.discord_dm_policy == "allowlist"
    assert cfg.discord_allow_from == ["123"]
    assert cfg.discord_group_policy == "open"
    assert cfg.discord_guilds == ["456"]
    assert cfg.discord_channels == ["789"]
    assert cfg.discord_require_mention is False
    assert cfg.discord_command_prefix == "$"
    assert cfg.telegram_dm_policy == "open"
    assert cfg.telegram_groups == ["-100123"]
    assert cfg.telegram_link_preview is True
    assert cfg.telegram_ignore_other_mentions is False
    assert cfg.gateway_reply_to_mode == "first"
    assert cfg.gateway_typing is False
    assert cfg.gateway_pairing_ttl == 120
    assert cfg.gateway_pairing_max_pending == 5
    assert cfg.session_history_limit == 12
    assert cfg.max_sessions == 7
    assert cfg.session_persistence is False


def test_invalid_policies_fail_safe(tmp_path) -> None:
    cfg = EchoConfig(
        data_dir=tmp_path,
        discord_dm_policy="public",
        telegram_group_policy="everyone",
        gateway_reply_to_mode="sometimes",
        discord_allow_from="123, @alice",
    )
    assert cfg.discord_dm_policy == "pairing"
    assert cfg.telegram_group_policy == "allowlist"
    assert cfg.gateway_reply_to_mode == "off"
    assert cfg.discord_allow_from == ["123", "@alice"]
