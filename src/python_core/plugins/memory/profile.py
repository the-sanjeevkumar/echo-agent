"""Durable owner profile store (PHASE 1).

Single owner across CLI, Telegram and Discord. Profiles live under
``<data_dir>/profiles/``:

* ``owner.json`` — the single owner profile (name, preferences, pinned facts,
  timestamps). Any sender whose platform user id is listed in
  ``config.owner_user_ids`` (CLI always) resolves to this profile.
* ``u:<sanitized_id>.json`` — one isolated profile per *other* sender, so a
  stranger's name/preferences never leak into the owner's profile and vice
  versa.

All writes are atomic (write tmp file, ``os.replace``). All text is stripped of
invisible/control unicode before it is persisted.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("echo.memory.profile")

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_NAME_STOPWORDS = {
    "hello", "hi", "hey", "yes", "yeah", "no", "ok", "okay", "please", "thanks",
    "thank you", "sure", "help", "stop", "start", "reset", "forget", "whoami",
    "status", "model", "echo", "ask", "task",
}


def _sanitize_text(text: str, *, max_len: int = 2000) -> str:
    """Remove control/invisible unicode and bound length."""
    text = _CONTROL_CHARS.sub("", text or "")
    text = "".join(ch for ch in text if ch.isprintable() or ch in " \t")
    return " ".join(text.split())[:max_len]


def _sanitize_key(user_id: str) -> str:
    """Filesystem-safe profile key for an isolated sender."""
    cleaned = "".join(ch if ch.isalnum() or ch in ":-_." else "_" for ch in str(user_id).strip())
    return ("u:" + cleaned)[:160] or "u:unknown"


def _count_tokens(text: str) -> int:
    """Cheap deterministic token estimate (same policy as the orchestrator's
    char/4 fallback, tiktoken-exact when available)."""
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:  # noqa: BLE001
        return max(1, len(text) // 4)


@dataclass(slots=True)
class Profile:
    """One user's durable profile."""

    user_id: str
    name: str = ""
    preferences: dict[str, str] = field(default_factory=dict)
    pinned_facts: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.updated_at = time.time()


class ProfileStore:
    """Durable, isolated per-user profiles with one shared owner profile."""

    def __init__(
        self,
        data_dir: Path,
        owner_user_ids: list[str] | None = None,
        confirm_ttl_seconds: float = 300.0,
    ) -> None:
        self.dir = Path(data_dir) / "profiles"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.owner_ids = {str(u).strip() for u in (owner_user_ids or []) if str(u).strip()}
        self.confirm_ttl = confirm_ttl_seconds
        self._cache: dict[str, Profile] = {}
        self._lock = threading.Lock()
        # pending profile deletions awaiting confirmation: key -> expiry ts
        self.pending_confirmations: dict[str, float] = {}
        # Load the owner profile eagerly (plugin start loads profiles).
        self.get("owner")

    # ── key resolution (access control) ──────────────────────────────────
    def resolve_key(self, user_id: str | None, source: str | None = None) -> str:
        """Map a platform sender + source to a profile key.

        CLI and any id in ``owner_user_ids`` ⇒ the shared ``owner`` profile.
        Everything else ⇒ an isolated profile that never touches the owner's.
        """
        if source == "cli" or (user_id and str(user_id) in self.owner_ids):
            return "owner"
        if not user_id:
            return "u:anonymous"
        return _sanitize_key(str(user_id))

    def is_owner(self, key: str) -> bool:
        return key == "owner"

    # ── load / save ──────────────────────────────────────────────────────
    def _path(self, key: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in ":-_." else "_" for ch in key)
        return self.dir / f"{safe}.json"

    def get(self, key: str) -> Profile:
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        path = self._path(key)
        profile = Profile(user_id=key)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = None
        if isinstance(raw, dict):
            profile.user_id = str(raw.get("user_id", key) or key)
            profile.name = _sanitize_text(str(raw.get("name", "") or ""), max_len=200)
            prefs = raw.get("preferences") or {}
            if isinstance(prefs, dict):
                profile.preferences = {
                    str(k)[:120]: _sanitize_text(str(v), max_len=500)
                    for k, v in prefs.items()
                }
            facts = raw.get("pinned_facts") or []
            if isinstance(facts, list):
                profile.pinned_facts = [
                    _sanitize_text(str(f), max_len=1000) for f in facts if str(f).strip()
                ]
            try:
                profile.created_at = float(raw.get("created_at") or time.time())
            except (TypeError, ValueError):
                profile.created_at = time.time()
            try:
                profile.updated_at = float(raw.get("updated_at") or time.time())
            except (TypeError, ValueError):
                profile.updated_at = time.time()
        with self._lock:
            self._cache[key] = profile
        return profile

    def save(self, profile: Profile) -> None:
        """Atomic write: tmp file + os.replace, never a torn profile file."""
        profile.touch()
        path = self._path(profile.user_id)
        payload = asdict(profile)
        tmp_name: str | None = None
        try:
            fd, tmp_name = tempfile.mkstemp(prefix=f".{profile.user_id}-", dir=str(self.dir), suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
                fh.write("\n")
            os.replace(tmp_name, path)
            tmp_name = None
        except OSError:
            logger.exception("failed to persist profile %s", profile.user_id)
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
        with self._lock:
            self._cache[profile.user_id] = profile

    # ── updates ──────────────────────────────────────────────────────────
    def update(
        self,
        key: str,
        *,
        name: str | None = None,
        preferences: dict[str, str] | None = None,
        facts: list[str] | None = None,
    ) -> Profile:
        """Last-write-wins merge. Name and preferences overwrite; pinned facts
        are appended (exact duplicates skipped) and never silently dropped."""
        profile = self.get(key)
        changed = False
        if name:
            name = _sanitize_text(name, max_len=200)
            if name.lower() not in _NAME_STOPWORDS and profile.name != name:
                profile.name = name
                changed = True
        if preferences:
            for k, v in preferences.items():
                k = _sanitize_text(k, max_len=120).replace(" ", "_").lower()
                v = _sanitize_text(v, max_len=500)
                if not k or not v:
                    continue
                # skip ephemera keys
                if k in {"question", "query", "task", "request", "issue", "problem", "error", "answer"}:
                    continue
                if profile.preferences.get(k) != v:
                    profile.preferences[k] = v
                    changed = True
        if facts:
            for f in facts:
                f = _sanitize_text(f, max_len=1000)
                if f and f not in profile.pinned_facts:
                    profile.pinned_facts.append(f)
                    changed = True
                    if len(profile.pinned_facts) > 200:  # bound growth
                        profile.pinned_facts = profile.pinned_facts[-200:]
        if changed:
            profile.touch()
            self.save(profile)
        return profile

    def clear(self, key: str, confirmed: bool = False) -> bool:
        """Wipe a profile. Requires confirmation (set via :meth:`request_clear`)."""
        if not confirmed:
            return False
        profile = Profile(user_id=key)
        self.save(profile)
        with self._lock:
            self.pending_confirmations.pop(key, None)
            self._cache[key] = profile
        logger.info("profile cleared: %s", key)
        return True

    def request_clear(self, key: str) -> float:
        """Mark a profile for deletion pending confirmation. Returns expiry ts."""
        expiry = time.time() + self.confirm_ttl
        with self._lock:
            self.pending_confirmations[key] = expiry
        return expiry

    def clear_confirmed(self, key: str) -> bool:
        """Complete a previously requested clear if it has not expired."""
        with self._lock:
            expiry = self.pending_confirmations.get(key)
        if expiry is None or time.time() > expiry:
            return False
        return self.clear(key, confirmed=True)

    # ── prompt block ─────────────────────────────────────────────────────
    def block(self, key: str, *, budget_tokens: int = 500) -> str:
        """USER PROFILE block injected into every system prompt.

        Always returned for the resolved key (empty profile ⇒ empty string),
        bounded to ``budget_tokens`` so restarts never grow the prompt.
        """
        profile = self.get(key)
        lines: list[str] = []
        if profile.name:
            lines.append(f"The user's name is {profile.name}.")
        for k, v in list(profile.preferences.items())[:40]:
            lines.append(f"User preference: {k} = {v}")
        for f in list(profile.pinned_facts)[:60]:
            lines.append(f"Pinned fact: {f}")

        if not lines:
            return ""
        while True:
            block = (
                "USER PROFILE (learned from conversation):\n"
                + "\n".join(f"- {line}" for line in lines)
                + "\n"
            )
            if _count_tokens(block) <= budget_tokens:
                return block
            if len(lines) <= 1:
                # Single line still over budget: truncate it hard rather than
                # returning an over-budget block.
                line = lines[0]
                char_budget = max(20, budget_tokens * 3 - 200)
                lines[0] = (line[:char_budget].rstrip() + "…") if line else ""
                if not lines[0]:
                    lines = []
                break
            # Drop a pinned fact first, then a preference (oldest first).
            pinned = [l for l in lines if l.startswith("Pinned fact:")]
            prefs = [l for l in lines if l.startswith("User preference:")]
            target = pinned[-1] if pinned else (prefs[-1] if prefs else None)
            if target is None:
                break
            lines.remove(target)
        if not lines:
            return ""
        return (
            "USER PROFILE (learned from conversation):\n"
            + "\n".join(f"- {line}" for line in lines)
            + "\n"
        )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            cached = len(self._cache)
            pending = len(self.pending_confirmations)
        files = len(list(self.dir.glob("*.json"))) if self.dir.is_dir() else 0
        return {"profiles_cached": cached, "profile_files": files, "pending_confirms": pending}


def create_store(data_dir: Path, owner_user_ids: list[str] | None = None) -> ProfileStore:
    return ProfileStore(data_dir, owner_user_ids=owner_user_ids)
