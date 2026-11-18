"""GitHub → live tool: autonomous skill ingestion (MODULE 3).

Pipeline (phases per spec):

  Phase 1 · DEEP ANALYSIS   clone (--depth 1), map the directory tree, read
                            README + core entry files; if an LLM is registered,
                            one structured call proposes candidate tools.
  Phase 2 · MANIFEST PRIORITY
                            repo-root ``agent_manifest.json`` or ``skill.md``
                            OVERRIDES LLM guesses — machine-readable manifests
                            give instant, reliable tool schemas.
  Phase 3 · DYNAMIC REGISTRATION
                            generate a standardized Python wrapper module
                            (JSON-Schema-conformant handlers), persist it under
                            ``skills/<name>/``, and hot-register the tools via
                            ``tool:register`` bus events — no restart.

Executable tool kinds:
    entry   {module, function}      imported from the cloned repo, called as fn(args)
    command [..] template           subprocess with {placeholder} substitution

Knowledge-only skills (no executable surface) are stored to Semantic Memory so
they still shape future behavior via recall.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from python_core.core.config import EchoConfig
from python_core.core.event_bus import EventBus
from python_core.core.interfaces import KernelContext, ToolSpec

logger = logging.getLogger("echo.skills")

MANIFEST_FILES = ("agent_manifest.json", "skill.md")
ENTRY_CANDIDATES = ("README.md", "readme.md", "README", "pyproject.toml", "setup.py", "package.json", "main.py", "index.py", "__init__.py")

PATTERNS_LLM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "tools": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "parameters": {"type": "object"},
                    "kind": {"type": "string", "enum": ["entry", "command", "knowledge"]},
                    "module": {"type": ["string", "null"]},
                    "function": {"type": ["string", "null"]},
                    "command": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "description", "parameters", "kind"],
            },
        },
    },
    "required": ["summary", "tools"],
}


@dataclass(slots=True)
class SkillDef:
    """Normalized representation of an acquired skill."""

    name: str
    source: str
    summary: str = ""
    version: str = "0.1.0"
    tools: list[dict[str, Any]] = field(default_factory=list)   # normalized tool dicts
    knowledge: list[str] = field(default_factory=list)
    manifest_origin: str = "none"   # agent_manifest.json | skill.md | llm | heuristic


# ── manifest parsing (Phase 2) ───────────────────────────────────────────────
def parse_agent_manifest(raw: str) -> SkillDef | None:
    """agent_manifest.json → normalized SkillDef (tools verified minimally)."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    tools = []
    for t in data.get("tools", []) if isinstance(data.get("tools"), list) else []:
        if not isinstance(t, dict) or not t.get("name"):
            continue
        kind = "entry" if (t.get("entry") or (t.get("module") and t.get("function"))) else ("command" if t.get("command") else "knowledge")
        tools.append({
            "name": str(t["name"]),
            "description": str(t.get("description", "")),
            "parameters": t.get("parameters") or {"type": "object", "properties": {}},
            "kind": kind,
            "module": (t.get("entry") or {}).get("module") if isinstance(t.get("entry"), dict) else t.get("module"),
            "function": (t.get("entry") or {}).get("function") if isinstance(t.get("entry"), dict) else t.get("function"),
            "command": t.get("command"),
        })
    return SkillDef(
        name=str(data.get("name") or "unnamed-skill"),
        source=str(data.get("source", "")),
        summary=str(data.get("description", "")),
        version=str(data.get("version", "0.1.0")),
        tools=tools,
        manifest_origin="agent_manifest.json",
    )


_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)


def parse_skill_md(raw: str) -> SkillDef | None:
    """skill.md → SkillDef.

    Supported shape (documented in README):
        ---                       ← optional YAML-ish frontmatter
        name: my-skill
        description: one line
        version: 0.1.0
        ---
        # prose ...               ← optional ```json {"tools":[...]} ``` block
    """
    m = _FRONTMATTER_RE.match(raw)
    meta: dict[str, str] = {}
    body = raw
    if m:
        body = m.group(2)
        for line in m.group(1).splitlines():
            if ":" in line and not line.strip().startswith("-"):
                key, _, value = line.partition(":")
                meta[key.strip().lower()] = value.strip()

    # Optional fenced JSON tools block in the body (pragmatic, reliable).
    tools: list[dict[str, Any]] = []
    knowledge: list[str] = []
    fence = re.search(r"```json\s*\n(.*?)\n```", body, re.DOTALL)
    if fence:
        try:
            data = json.loads(fence.group(1))
            for t in data.get("tools", []) if isinstance(data, dict) else []:
                if isinstance(t, dict) and t.get("name"):
                    kind = "entry" if (t.get("module") and t.get("function")) else ("command" if t.get("command") else "knowledge")
                    tools.append({
                        "name": str(t["name"]),
                        "description": str(t.get("description", "")),
                        "parameters": t.get("parameters") or {"type": "object", "properties": {}},
                        "kind": kind,
                        "module": t.get("module"),
                        "function": t.get("function"),
                        "command": t.get("command"),
                    })
            body_wo = body[: fence.start()] + body[fence.end():]
        except json.JSONDecodeError:
            body_wo = body
    else:
        body_wo = body

    # Prose body → knowledge lines (headings + first sentences).
    for line in body_wo.splitlines():
        line = line.strip()
        if line.startswith("#") or (line and not line.startswith(("```", "|"))):
            knowledge.append(line.lstrip("# ").strip())
    knowledge = [k for k in knowledge if k][:12]

    if not meta and not tools:
        return None
    return SkillDef(
        name=meta.get("name") or (tools[0]["name"] if tools else "unnamed-skill"),
        source=meta.get("source", ""),
        summary=meta.get("description") or (knowledge[0] if knowledge else ""),
        version=meta.get("version", "0.1.0"),
        tools=tools,
        knowledge=knowledge,
        manifest_origin="skill.md",
    )


# ── Phase 1 helpers ──────────────────────────────────────────────────────────
async def clone_repo(url: str, dest_parent: Path, timeout: float = 180.0) -> Path:
    """Shallow clone into dest_parent/<hash>. Raises RuntimeError on failure."""
    dest = dest_parent / f"clone-{uuid.uuid4().hex[:10]}"
    cmd = ["git", "clone", "--depth", "1", "--single-branch", url, str(dest)]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            _, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"git clone timed out after {timeout:.0f}s") from None
    except FileNotFoundError as exc:
        raise RuntimeError("git is not installed (required for skill ingestion)") from exc
    if proc.returncode != 0:
        err = (err_b or b"").decode(errors="replace")[:500]
        raise RuntimeError(f"git clone failed: {err}")
    if not dest.is_dir():
        raise RuntimeError("clone produced no directory")
    return dest


def map_tree(root: Path, *, max_entries: int = 200, max_depth: int = 4) -> str:
    """Compact directory map (token-efficient Phase 1 input for the LLM)."""
    lines: list[str] = []
    count = 0
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part.startswith((".", "__pycache__")) for part in rel.parts):
            continue
        if len(rel.parts) > max_depth:
            continue
        if path.is_dir():
            lines.append(f"{rel}/")
        else:
            lines.append(f"  {rel} ({path.stat().st_size}b)")
        count += 1
        if count >= max_entries:
            lines.append("…[truncated]")
            break
    return "\n".join(lines)


def read_entry_files(root: Path, *, char_budget: int = 6000) -> str:
    """README + core entry file excerpts (bounded — delta-first policy)."""
    chunks: list[str] = []
    used = 0
    for name in ENTRY_CANDIDATES:
        path = root / name
        if not path.is_file() or used >= char_budget:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")[:2400]
        chunks.append(f"### {name}\n{text}")
        used += len(text)
    return "\n\n".join(chunks)


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip()).strip("_").lower()
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"skill_{cleaned}"
    return cleaned[:48]


# ── generated wrapper module (Phase 3) ───────────────────────────────────────
WRAPPER_TEMPLATE = '''"""Auto-generated by echo-agent SkillIngester — DO NOT EDIT.

Skill: {skill_name}
Source: {source}
Every handler conforms to the echo-agent tool contract:
    handler(args: dict) -> dict | awaitable[dict]   (JSON-serializable result)
"""

import asyncio
import importlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent / "repo"
MANIFEST = {manifest!r}

def _entry(module_name, fn_name, required):
    def handler(args):
        sys.path.insert(0, str(REPO))
        missing = [k for k in required if k not in (args or {{}})]
        if missing:
            return {{"ok": False, "error": f"missing required args: {{missing}}"}}
        fn = getattr(importlib.import_module(module_name), fn_name)
        out = fn(args or {{}})
        if not isinstance(out, dict):
            out = {{"result": out}}
        return out
    return handler

def _command(argv, required):
    async def handler(args):
        args = args or {{}}
        missing = [k for k in required if k not in args]
        if missing:
            return {{"ok": False, "error": f"missing required args: {{missing}}"}}
        cmd = [part.format(**args) for part in argv]
        try:
            # cwd=REPO: command templates use repo-relative paths (how the
            # repo's own README would invoke them).
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=120.0)
        except Exception as exc:
            return {{"ok": False, "error": f"{{type(exc).__name__}}: {{exc}}"}}
        return {{
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": out_b.decode("utf-8", errors="replace")[:8000],
            "stderr": err_b.decode("utf-8", errors="replace")[:2000],
        }}
    return handler

def _null_handler(args):
    return {{"ok": False, "error": "knowledge-only tool; no executable surface"}}

def build_handlers():
    handlers = {{}}
    for tool in MANIFEST.get("tools", []):
        kind = tool.get("kind")
        if kind == "entry" and tool.get("module") and tool.get("function"):
            handlers[tool["name"]] = _entry(tool["module"], tool["function"], tool.get("required", []))
        elif kind == "command" and tool.get("command"):
            handlers[tool["name"]] = _command(tool["command"], tool.get("required", []))
        else:
            handlers[tool["name"]] = _null_handler
    return handlers

HANDLERS = build_handlers()
'''


def _required_from_schema(parameters: dict[str, Any]) -> list[str]:
    if not isinstance(parameters, dict):
        return []
    required = parameters.get("required")
    if isinstance(required, list):
        return [str(r) for r in required]
    return []


# ── the plugin ───────────────────────────────────────────────────────────────
class SkillIngester:
    """SkillPlugin implementation: repo URL / local path → live tools."""

    name = "skills"
    version = "0.1.0"
    capabilities = ["skill-acquisition", "dynamic-registration"]

    def __init__(self, config: EchoConfig | None = None) -> None:
        self.config = config or EchoConfig.load()
        self._ctx: KernelContext | None = None
        self._bus: EventBus | None = None
        self._loaded_wrappers: dict[str, Any] = {}
        # PHASE 4: synthesized procedure skills (name → one-line description).
        self._synthesized: dict[str, str] = {}

    # ── lifecycle ──────────────────────────────────────────────────────────
    async def start(self, ctx: KernelContext) -> None:
        self._ctx = ctx
        self._bus = ctx.bus
        ctx.config.ensure_dirs()
        ctx.services["skills"] = self
        await ctx.bus.emit("tool:register", {
            "spec": ToolSpec(
                name="skill_ingest",
                description=(
                    "Acquire a new skill from a GitHub URL, git URL, or local repo path. "
                    "Analyzes the repo, prioritizes skill.md/agent_manifest.json, generates "
                    "tool wrappers, and hot-registers them without restart."
                ),
                parameters={
                    "type": "object",
                    "properties": {"source": {"type": "string", "description": "repo URL or local path"}},
                    "required": ["source"],
                },
                source="plugin:skills",
                tags=["skills"],
            ),
            "handler": self._tool_ingest,
        })
        # Persistence: re-register previously acquired skills (restart-safe).
        loaded = await self.load_installed()
        self._load_synthesized_catalog()
        ctx.logger.info("skills plugin online (%d installed skill(s))", len(loaded))

    def stop(self) -> None:
        self._ctx = None

    # ── main ingestion pipeline ────────────────────────────────────────────
    async def ingest(self, source: str) -> dict[str, Any]:
        """Full Phase 1→3 pipeline for one repository. Returns a report."""
        source = (source or "").strip()
        if not source:
            return {"ok": False, "error": "empty source"}
        started = time.time()

        # Phase 1 · acquire the code.
        src_path = Path(source).expanduser()
        tmp_clone: Path | None = None
        if src_path.is_dir():
            repo = src_path
        else:
            tmp_clone = await clone_repo(source, Path(tempfile.gettempdir()))
            repo = tmp_clone

        try:
            tree = map_tree(repo)
            readme_material = read_entry_files(repo)

            # Phase 2 · manifest priority (manifests override LLM guesses).
            skill = None
            for fname in MANIFEST_FILES:
                fpath = repo / fname
                if fpath.is_file():
                    raw = fpath.read_text(encoding="utf-8", errors="replace")[:20000]
                    if fname == "agent_manifest.json":
                        skill = parse_agent_manifest(raw)
                    else:
                        skill = parse_skill_md(raw)
                    if skill is not None:
                        break

            # Phase 1 continued · LLM deep analysis when no manifest.
            if skill is None:
                skill = await self._analyze_with_llm(repo, tree, readme_material) or self._analyze_heuristic(repo, tree, readme_material)

            skill.name = _sanitize_name(skill.name)
            skill.source = source

            # Phase 3 · install + generate wrapper + hot-register.
            skill_dir = self.config.resolved_skills_dir / skill.name
            if skill_dir.exists():
                shutil.rmtree(skill_dir)  # re-ingest replaces the previous version
            skill_dir.mkdir(parents=True, exist_ok=True)
            if tmp_clone is not None:
                shutil.move(str(tmp_clone), skill_dir / "repo")
                tmp_clone = None  # ownership transferred
            else:  # local source: lightweight copy (repo content, ignores applied)
                shutil.copytree(
                    repo, skill_dir / "repo",
                    ignore=shutil.ignore_patterns(".git", "__pycache__", ".venv", "node_modules"),
                    dirs_exist_ok=True,
                )

            manifest_payload = {
                "name": skill.name,
                "version": skill.version,
                "description": skill.summary,
                "tools": [
                    {**t, "required": _required_from_schema(t.get("parameters", {}))}
                    for t in skill.tools
                ],
            }
            (skill_dir / "wrapper.py").write_text(
                WRAPPER_TEMPLATE.format(skill_name=skill.name, source=source, manifest=manifest_payload),
                encoding="utf-8",
            )
            (skill_dir / "skill.md").write_text(
                f"---\nname: {skill.name}\ndescription: {skill.summary}\nversion: {skill.version}\n---\n\n{skill.summary}\n",
                encoding="utf-8",
            )
            (skill_dir / "metadata.json").write_text(
                json.dumps({
                    "name": skill.name, "source": source, "installed_at": time.time(),
                    "origin": skill.manifest_origin, "tools": [t["name"] for t in skill.tools],
                }, indent=2),
                encoding="utf-8",
            )

            registered = await self._load_wrapper(skill_dir)
            self._store_knowledge(skill)

            report = {
                "ok": True,
                "skill": skill.name,
                "manifest_origin": skill.manifest_origin,
                "tools_registered": registered,
                "knowledge_items": len(skill.knowledge),
                "elapsed_ms": int((time.time() - started) * 1000),
                "dir": str(skill_dir),
            }
            if self._bus is not None:
                self._bus.publish("skill:ingested", report, source="skills")
            return report
        finally:
            if tmp_clone is not None:  # failed after clone, before install
                shutil.rmtree(tmp_clone, ignore_errors=True)

    # ── analysis backends ──────────────────────────────────────────────────
    async def _analyze_with_llm(self, repo: Path, tree: str, material: str) -> SkillDef | None:
        if self._ctx is None:
            return None
        llm = self._ctx.services.get("llm")
        if llm is None:
            return None
        try:
            out = await llm.complete(
                [{"role": "user", "content": (
                    "Analyze this repository for an agent. Propose executable tools it offers.\n"
                    "FILE TREE:\n" + tree[:3000] + "\n\nKEY FILES:\n" + material[:6000]
                    + "\n\nOnly propose tools with kind=entry (module+function), kind=command "
                      "(a shell template with {placeholders}), or kind=knowledge. Be precise."
                )}],
                schema=PATTERNS_LLM_SCHEMA,
                temperature=0.1,
                max_tokens=700,
            )
            content = out.get("content") if isinstance(out, dict) else None
            if not isinstance(content, dict):
                return None
            tools = []
            for t in content.get("tools", [])[:12]:
                if not t.get("name"):
                    continue
                tools.append({
                    "name": str(t["name"]),
                    "description": str(t.get("description", "")),
                    "parameters": t.get("parameters") or {"type": "object", "properties": {}},
                    "kind": t.get("kind", "knowledge"),
                    "module": t.get("module"),
                    "function": t.get("function"),
                    "command": t.get("command"),
                })
            return SkillDef(
                name=repo.name or "llm-skill",
                source="",
                summary=str(content.get("summary", ""))[:300],
                tools=tools,
                knowledge=[tools_summary(t) for t in tools if t["kind"] == "knowledge"],
                manifest_origin="llm",
            )
        except Exception:  # noqa: BLE001
            logger.exception("LLM skill analysis failed; falling back to heuristics")
            return None

    def _analyze_heuristic(self, repo: Path, tree: str, material: str) -> SkillDef:
        """Zero-LLM fallback: package the repo's documented behavior as knowledge."""
        first_lines = [l for l in material.splitlines() if l.strip()][:8]
        summary = " ".join(first_lines)[:280] if first_lines else repo.name
        return SkillDef(
            name=repo.name or "local-skill",
            source="",
            summary=summary,
            knowledge=[l[:160] for l in first_lines[:8]],
            manifest_origin="heuristic",
        )

    # ── dynamic registration (Phase 3) ─────────────────────────────────────
    async def _load_wrapper(self, skill_dir: Path) -> list[str]:
        """Import the generated wrapper and hot-register its tools."""
        wrapper_path = skill_dir / "wrapper.py"
        module_name = f"echo_skill_{skill_dir.name}"
        spec = importlib.util.spec_from_file_location(module_name, wrapper_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load wrapper: {wrapper_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        self._loaded_wrappers[skill_dir.name] = module

        registered: list[str] = []
        for tool in module.MANIFEST.get("tools", []):
            tool_name = f"{skill_dir.name}_{_sanitize_name(tool['name'])}"
            spec_obj = ToolSpec(
                name=tool_name,
                description=f"[skill:{skill_dir.name}] {tool.get('description', '')}"[:500],
                parameters=tool.get("parameters") or {"type": "object", "properties": {}},
                source=f"skill:{skill_dir.name}",
                version=str(module.MANIFEST.get("version", "0.1.0")),
                tags=["skill"],
            )
            handler = module.HANDLERS.get(tool["name"])
            if self._bus is not None:
                # Awaited emit: a tool listed in the returned report is
                # guaranteed callable the moment ingest() returns.
                await self._bus.emit("tool:register", {"spec": spec_obj, "handler": handler}, source="skills")
            registered.append(tool_name)
        return registered

    # ── PHASE 4: synthesized procedure skills ─────────────────────────────
    @staticmethod
    def _render_skill_md(
        name: str, when_to_use: str, steps: list[str], tool_sequence: list[str], pitfalls: list[str]
    ) -> str:
        def section(title: str, items: list[str]) -> str:
            body = "\n".join(f"- {item}" for item in items) if items else "- (none recorded)"
            return f"## {title}\n{body}\n"

        return (
            f"---\nname: {name}\ndescription: {when_to_use[:300]}\n"
            f"origin: synthesized\n---\n\n"
            f"# {name}\n\n"
            f"## When to use\n{when_to_use}\n\n"
            + section("Steps", steps)
            + "\n"
            + section("Tool sequence", tool_sequence)
            + "\n"
            + section("Pitfalls", pitfalls)
        )

    async def synthesize_skill(
        self,
        name: str,
        *,
        when_to_use: str,
        steps: list[str] | None = None,
        tool_sequence: list[str] | None = None,
        pitfalls: list[str] | None = None,
        patch: bool = False,
    ) -> dict[str, Any]:
        """Create or PATCH one procedural skill (SKILL.md) — never duplicate.

        Uses the same skills directory as the existing loader so synthesized
        procedures are restart-safe and visible in the next prompt.
        """
        safe = _sanitize_name(name)
        skill_dir = self.config.resolved_skills_dir / safe
        exists = skill_dir.is_dir() and (skill_dir / "SKILL.md").is_file()
        if exists and not patch:
            return {"ok": True, "created": False, "name": safe, "reason": "already exists"}

        skill_dir.mkdir(parents=True, exist_ok=True)
        content = self._render_skill_md(
            safe, when_to_use, steps or [], tool_sequence or [], pitfalls or []
        )
        tmp = skill_dir / "SKILL.md.tmp"
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, skill_dir / "SKILL.md")
        meta_path = skill_dir / "metadata.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        except (OSError, ValueError):
            meta = {}
        meta.update({
            "name": safe,
            "origin": "synthesized",
            "description": when_to_use[:300],
            "updated_at": time.time(),
        })
        meta.setdefault("created_at", meta["updated_at"])
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        self._synthesized[safe] = when_to_use[:300]

        if self._bus is not None:
            self._bus.publish(
                "skill:synthesized" if not exists else "skill:patched",
                {"name": safe, "created": not exists}, source="skills"
            )
        return {
            "ok": True,
            "created": not exists,
            "patched": exists,
            "name": safe,
            "path": str(skill_dir / "SKILL.md"),
        }

    def _load_synthesized_catalog(self) -> None:
        skills_root = self.config.resolved_skills_dir
        if not skills_root.is_dir():
            return
        for meta_path in sorted(skills_root.glob("*/metadata.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if meta.get("origin") == "synthesized":
                name = str(meta.get("name") or meta_path.parent.name)
                self._synthesized[name] = str(meta.get("description", ""))[:300]

    def catalog_block(self, *, limit: int = 20) -> str:
        """Frozen SKILLS block injected into the next system prompt."""
        if not self._synthesized:
            return ""
        items = sorted(self._synthesized.items())[:limit]
        return (
            "SKILLS (learned procedures available to you):\n"
            + "\n".join(f"- {name}: {description}" for name, description in items)
            + "\n"
        )

    async def load_installed(self) -> list[str]:
        """Restart-safe: load every installed skill's wrapper from disk."""
        loaded: list[str] = []
        skills_root = self.config.resolved_skills_dir
        if not skills_root.is_dir():
            return loaded
        for metadata_path in sorted(skills_root.glob("*/metadata.json")):
            try:
                data = json.loads(metadata_path.read_text(encoding="utf-8"))
                if data.get("origin") == "synthesized":
                    continue  # knowledge-only skills have no wrapper to hoist
                loaded.extend(await self._load_wrapper(metadata_path.parent))
                logger.info("skill reloaded from disk: %s", data.get("name"))
            except Exception:  # noqa: BLE001
                logger.exception("failed to load installed skill at %s", metadata_path.parent)
        return loaded

    def _store_knowledge(self, skill: SkillDef) -> None:
        """Knowledge-only surfaces still matter: store to semantic memory."""
        memory = self._ctx.service("memory") if self._ctx else None
        if memory is None or not hasattr(memory, "write_working"):
            return
        try:
            sm = getattr(memory, "sm", None)
            if sm is not None:
                text = f"skill '{skill.name}': {skill.summary}"
                sm.upsert(f"skill:{skill.name}", text, metadata={"kind": "skill", "source": skill.source}, salience=0.9)
                for i, k in enumerate(skill.knowledge[:6]):
                    sm.upsert(f"skill:{skill.name}:k{i}", f"skill '{skill.name}' note: {k}", metadata={"kind": "skill-note"}, salience=0.6)
        except Exception:  # noqa: BLE001
            logger.exception("failed to store skill knowledge")

    # ── tool handler ───────────────────────────────────────────────────────
    def _tool_ingest(self, args: dict[str, Any]) -> Any:
        return self.ingest(str(args.get("source", "")))


def tools_summary(tool: dict[str, Any]) -> str:
    return f"tool {tool.get('name', '?')}: {tool.get('description', '')}"[:160]


def create_plugin() -> SkillIngester:
    """Discovery-contract entry point."""
    return SkillIngester()
