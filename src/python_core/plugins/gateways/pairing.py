"""Explicit DM pairing (OpenClaw-style access approval).

A chat bot should not obey the first stranger who finds its username. The
default DM policy is therefore ``pairing``: unknown senders receive a short,
single-use code and their message is *not* processed until an operator approves
the request from the machine that runs echo-agent::

    echo-agent pairing list telegram
    echo-agent pairing approve telegram ABCD2345

State is a small JSON file under ``<data_dir>/gateways/pairing.json`` (mode
0600). It is re-read on every decision so an approval performed by the CLI is
visible to an already-running gateway without a restart.
"""

from __future__ import annotations

import json
import os
import secrets
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

# Ambiguous characters (0/O, 1/I) are intentionally absent: codes get read out
# loud and copied by hand.
_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_CODE_LEN = 8


def new_pairing_code() -> str:
    return "".join(secrets.choice(_CODE_ALPHABET) for _ in range(_CODE_LEN))


def _empty_state() -> dict[str, Any]:
    return {"version": 1, "channels": {}}


class PairingStore:
    """File-backed pending/approved sender registry (one store, all channels)."""

    def __init__(
        self,
        path: str | Path,
        *,
        ttl: float = 3600.0,
        max_pending: int = 3,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path)
        self.ttl = float(ttl)
        self.max_pending = max(1, int(max_pending))
        self._clock = clock
        self._lock = threading.RLock()

    # ── file plumbing ──────────────────────────────────────────────────────
    def _read(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return _empty_state()
        if not isinstance(raw, dict):
            return _empty_state()
        raw.setdefault("version", 1)
        channels = raw.get("channels")
        if not isinstance(channels, dict):
            raw["channels"] = {}
        for entry in raw["channels"].values():
            if not isinstance(entry, dict):
                continue
            entry.setdefault("pending", {})
            entry.setdefault("approved", {})
        return raw

    def _write(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".pairing-", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2, ensure_ascii=False)
                fh.write("\n")
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, self.path)
        finally:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass

    def _channel(self, state: dict[str, Any], channel: str) -> dict[str, Any]:
        channels = state.setdefault("channels", {})
        entry = channels.get(channel)
        if not isinstance(entry, dict):
            entry = {"pending": {}, "approved": {}}
            channels[channel] = entry
        entry.setdefault("pending", {})
        entry.setdefault("approved", {})
        return entry

    def _purge(self, entry: dict[str, Any]) -> bool:
        now = self._clock()
        pending = entry.get("pending", {})
        expired = [
            sender for sender, item in pending.items()
            if not isinstance(item, dict) or now - float(item.get("created_at", 0)) >= self.ttl
        ]
        for sender in expired:
            pending.pop(sender, None)
        return bool(expired)

    @staticmethod
    def _norm(sender_id: str | int) -> str:
        return str(sender_id).strip()

    # ── queries ────────────────────────────────────────────────────────────
    def is_approved(self, channel: str, sender_id: str | int) -> bool:
        with self._lock:
            state = self._read()
            entry = state.get("channels", {}).get(channel, {})
            return self._norm(sender_id) in entry.get("approved", {})

    def request(
        self,
        channel: str,
        sender_id: str | int,
        *,
        username: str = "",
        chat_id: str = "",
    ) -> dict[str, Any]:
        """Create or inspect a pairing request.

        Returns a dict with ``status`` ∈ ``created | pending | approved | full``.
        Only ``created`` should trigger a chat message, so a stranger cannot
        spam the bot (or the operator) by messaging repeatedly.
        """
        sender = self._norm(sender_id)
        now = self._clock()
        with self._lock:
            state = self._read()
            entry = self._channel(state, channel)
            self._purge(entry)

            if sender in entry["approved"]:
                return {"status": "approved", "sender_id": sender}

            pending = entry["pending"]
            existing = pending.get(sender)
            if isinstance(existing, dict):
                return {
                    "status": "pending",
                    "sender_id": sender,
                    "code": str(existing.get("code", "")),
                    "expires_in": max(0, int(self.ttl - (now - float(existing.get("created_at", now))))),
                }

            if len(pending) >= self.max_pending:
                return {"status": "full", "sender_id": sender}

            code = new_pairing_code()
            pending[sender] = {
                "code": code,
                "created_at": now,
                "username": username,
                "chat_id": str(chat_id),
            }
            self._write(state)
            return {"status": "created", "sender_id": sender, "code": code, "expires_in": int(self.ttl)}

    def approve(self, channel: str, code: str) -> dict[str, Any] | None:
        wanted = (code or "").strip().upper()
        if not wanted:
            return None
        now = self._clock()
        with self._lock:
            state = self._read()
            entry = self._channel(state, channel)
            changed = self._purge(entry)
            for sender, item in list(entry["pending"].items()):
                if str(item.get("code", "")).upper() != wanted:
                    continue
                entry["pending"].pop(sender, None)
                entry["approved"][sender] = {
                    "approved_at": now,
                    "username": item.get("username", ""),
                    "chat_id": item.get("chat_id", ""),
                }
                self._write(state)
                return {
                    "channel": channel,
                    "sender_id": sender,
                    "username": item.get("username", ""),
                    "chat_id": item.get("chat_id", ""),
                }
            if changed:
                self._write(state)
            return None

    def revoke(self, channel: str, sender_id: str | int) -> bool:
        sender = self._norm(sender_id)
        with self._lock:
            state = self._read()
            entry = state.get("channels", {}).get(channel)
            if not isinstance(entry, dict):
                return False
            removed = entry.get("approved", {}).pop(sender, None) is not None
            removed = entry.get("pending", {}).pop(sender, None) is not None or removed
            if removed:
                self._write(state)
            return removed

    def list_pending(self, channel: str | None = None) -> list[dict[str, Any]]:
        now = self._clock()
        with self._lock:
            state = self._read()
            out: list[dict[str, Any]] = []
            for name, entry in state.get("channels", {}).items():
                if channel is not None and name != channel:
                    continue
                if not isinstance(entry, dict):
                    continue
                for sender, item in entry.get("pending", {}).items():
                    if not isinstance(item, dict):
                        continue
                    age = max(0.0, now - float(item.get("created_at", now)))
                    if age >= self.ttl:
                        continue
                    out.append({
                        "channel": name,
                        "sender_id": sender,
                        "code": item.get("code", ""),
                        "username": item.get("username", ""),
                        "chat_id": item.get("chat_id", ""),
                        "age_seconds": int(age),
                        "expires_in": max(0, int(self.ttl - age)),
                    })
            return out

    def list_approved(self, channel: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            state = self._read()
            out: list[dict[str, Any]] = []
            for name, entry in state.get("channels", {}).items():
                if channel is not None and name != channel:
                    continue
                if not isinstance(entry, dict):
                    continue
                for sender, item in entry.get("approved", {}).items():
                    out.append({
                        "channel": name,
                        "sender_id": sender,
                        "username": (item or {}).get("username", "") if isinstance(item, dict) else "",
                        "chat_id": (item or {}).get("chat_id", "") if isinstance(item, dict) else "",
                    })
            return out
