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
    return {"version": 1, "channels": {}, "notifications": []}


def _new_request_id() -> str:
    """Stable, opaque id for a pairing request (shown by ``pairing list --json``)."""
    return secrets.token_hex(8)


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
        if not isinstance(raw.get("notifications"), list):
            raw["notifications"] = []
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
        spam the bot (or the operator) by messaging repeatedly. A repeat request
        refreshes ``last_seen_at`` but keeps the same single-use code.
        """
        sender = self._norm(sender_id)
        now = self._clock()
        with self._lock:
            state = self._read()
            entry = self._channel(state, channel)
            purged = self._purge(entry)

            if sender in entry["approved"]:
                if purged:
                    self._write(state)
                return {"status": "approved", "sender_id": sender}

            pending = entry["pending"]
            existing = pending.get(sender)
            if isinstance(existing, dict):
                created_at = float(existing.get("created_at", now))
                existing["last_seen_at"] = now
                if username:
                    existing["username"] = username
                if chat_id:
                    existing["chat_id"] = str(chat_id)
                self._write(state)
                return {
                    "status": "pending",
                    "sender_id": sender,
                    "id": str(existing.get("id") or ""),
                    "code": str(existing.get("code", "")),
                    "created_at": created_at,
                    "last_seen_at": now,
                    "expires_in": max(0, int(self.ttl - (now - created_at))),
                }

            if len(pending) >= self.max_pending:
                if purged:
                    self._write(state)
                return {"status": "full", "sender_id": sender}

            code = new_pairing_code()
            request_id = _new_request_id()
            pending[sender] = {
                "id": request_id,
                "code": code,
                "created_at": now,
                "last_seen_at": now,
                "username": username,
                "chat_id": str(chat_id),
            }
            self._write(state)
            return {
                "status": "created",
                "sender_id": sender,
                "id": request_id,
                "code": code,
                "created_at": now,
                "last_seen_at": now,
                "expires_in": int(self.ttl),
            }

    def approve(
        self, channel: str, code: str, *, notify: bool = False
    ) -> dict[str, Any] | None:
        """Approve a pending code once. With *notify*, queue a confirmation
        that the running gateway will deliver to the requester's DM.
        """
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
                chat_id = str(item.get("chat_id") or "")
                deliver = bool(notify and chat_id)
                entry["approved"][sender] = {
                    "approved_at": now,
                    "username": item.get("username", ""),
                    "chat_id": chat_id,
                    "request_id": str(item.get("id") or ""),
                    "notified": deliver,
                }
                if deliver:
                    state.setdefault("notifications", []).append({
                        "channel": channel,
                        "sender_id": sender,
                        "chat_id": chat_id,
                        "created_at": now,
                    })
                self._write(state)
                return {
                    "channel": channel,
                    "sender_id": sender,
                    "username": item.get("username", ""),
                    "chat_id": chat_id,
                    "notified": deliver,
                }
            if changed:
                self._write(state)
            return None

    def dismiss(self, channel: str, code: str) -> dict[str, Any] | None:
        """Drop a pending request without approving it (not a permanent block:
        the sender may request again). Codes remain single-use either way.
        """
        wanted = (code or "").strip().upper()
        if not wanted:
            return None
        with self._lock:
            state = self._read()
            entry = state.get("channels", {}).get(channel)
            if not isinstance(entry, dict):
                return None
            changed = self._purge(entry)
            for sender, item in list(entry.get("pending", {}).items()):
                if str(item.get("code", "")).upper() != wanted:
                    continue
                entry["pending"].pop(sender, None)
                self._write(state)
                return {
                    "channel": channel,
                    "sender_id": sender,
                    "code": wanted,
                    "username": item.get("username", ""),
                    "chat_id": item.get("chat_id", ""),
                }
            if changed:
                self._write(state)
            return None

    def take_notifications(self, channel: str | None = None) -> list[dict[str, Any]]:
        """Drain queued approval confirmations for the running gateway."""
        with self._lock:
            state = self._read()
            items = state.get("notifications") or []
            if not isinstance(items, list) or not items:
                return []
            matched = [
                item for item in items
                if isinstance(item, dict)
                and (channel is None or item.get("channel") == channel)
            ]
            if not matched:
                return []
            state["notifications"] = [
                item for item in items
                if not (
                    isinstance(item, dict)
                    and (channel is None or item.get("channel") == channel)
                )
            ]
            self._write(state)
            return matched

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
                    created_at = float(item.get("created_at", now))
                    out.append({
                        "channel": name,
                        "sender_id": sender,
                        "id": item.get("id", ""),
                        "code": item.get("code", ""),
                        "username": item.get("username", ""),
                        "chat_id": item.get("chat_id", ""),
                        "created_at": created_at,
                        "last_seen_at": float(item.get("last_seen_at", created_at)),
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
                        "approved_at": float((item or {}).get("approved_at", 0)) if isinstance(item, dict) else 0.0,
                        "request_id": (item or {}).get("request_id", "") if isinstance(item, dict) else "",
                    })
            return out
