"""Skill acquisition: manifest parsing, repo ingestion, dynamic registration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from python_core.plugins.skills.ingester import (
    SkillIngester,
    map_tree,
    parse_agent_manifest,
    parse_skill_md,
)


# ── Phase 2: manifest parsers ────────────────────────────────────────────────
def test_parse_agent_manifest() -> None:
    raw = json.dumps({
        "name": "adder",
        "version": "1.2",
        "description": "adds numbers",
        "tools": [
            {
                "name": "add",
                "description": "add two numbers",
                "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]},
                "entry": {"module": "adder", "function": "add"},
            },
            {"name": "shell_tool", "description": "runs a command", "command": ["echo", "{msg}"]},
            {"name": "knowledge", "description": "docs only"},
        ],
    })
    skill = parse_agent_manifest(raw)
    assert skill is not None
    assert skill.name == "adder"
    assert [t["kind"] for t in skill.tools] == ["entry", "command", "knowledge"]
    assert skill.tools[0]["module"] == "adder" and skill.tools[0]["function"] == "add"


def test_parse_agent_manifest_invalid_json() -> None:
    assert parse_agent_manifest("{not json") is None


def test_parse_skill_md_with_tools_block() -> None:
    raw = """---
name: greeter
description: says hello
version: 0.2
---

# Greeter skill

## Usage
Always greet politely.

```json
{"tools": [{"name": "greet", "description": "greet someone",
  "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
  "command": ["echo", "hello {name}"]}]}
```
"""
    skill = parse_skill_md(raw)
    assert skill is not None
    assert skill.name == "greeter"
    assert skill.version == "0.2"
    assert len(skill.tools) == 1 and skill.tools[0]["kind"] == "command"
    assert any("greet politely" in k for k in skill.knowledge)


def test_parse_skill_md_frontmatter_only() -> None:
    skill = parse_skill_md("---\nname: docs-skill\ndescription: pure knowledge\n---\nSome notes.\n")
    assert skill is not None
    assert skill.tools == []
    assert skill.knowledge


# ── Phase 1 helper: tree mapping ─────────────────────────────────────────────
def test_map_tree_compact(tmp_path: Path) -> None:
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "deep").mkdir()
    (tmp_path / "a" / "b" / "f.txt").write_text("x")
    (tmp_path / ".hidden").mkdir()
    tree = map_tree(tmp_path)
    assert "a/" in tree and "f.txt" in tree
    assert ".hidden" not in tree  # hidden dirs skipped


# ── Phases 1→3: full ingestion of a local repo ───────────────────────────────
@pytest.fixture
def skill_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "adder-repo"
    repo.mkdir()
    (repo / "adder.py").write_text(
        "def add(args):\n"
        "    return {'ok': True, 'sum': args['a'] + args['b']}\n"
        "\n"
        "def shout(args):\n"
        "    return {'ok': True, 'text': args['msg'].upper()}\n"
    )
    (repo / "agent_manifest.json").write_text(json.dumps({
        "name": "adder-skill",
        "description": "adds and shouts",
        "tools": [
            {"name": "add", "description": "add two numbers",
             "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]},
             "entry": {"module": "adder", "function": "add"}},
            {"name": "shout", "description": "uppercase a message",
             "parameters": {"type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"]},
             "entry": {"module": "adder", "function": "shout"}},
        ],
    }))
    return repo


async def test_ingest_registers_tools_hot(kernel, skill_repo: Path) -> None:
    ingester = kernel.ctx.service("skills")
    report = await ingester.ingest(str(skill_repo))

    assert report["ok"] is True
    assert report["manifest_origin"] == "agent_manifest.json"
    assert report["tools_registered"] == ["adder_skill_add", "adder_skill_shout"]

    # Phase 3 proof: hot-registered tools work IMMEDIATELY, no restart
    r = await kernel.ctx.tools.execute("adder_skill_add", {"a": 2, "b": 40})
    assert r["ok"] and r["result"]["sum"] == 42

    r2 = await kernel.ctx.tools.execute("adder_skill_shout", {"msg": "echo"})
    assert r2["ok"] and r2["result"]["text"] == "ECHO"

    # tool schemas visible to the manifest protocol (MODULE 5 integration)
    names = [s["name"] for s in kernel.ctx.tools.schemas()]
    assert "adder_skill_add" in names


async def test_ingest_persists_across_restart(kernel, skill_repo: Path) -> None:
    ingester = kernel.ctx.service("skills")
    await ingester.ingest(str(skill_repo))
    # simulate restart: a fresh ingester loads wrappers from disk
    fresh = SkillIngester(kernel.config)
    await fresh.start(kernel.ctx)
    r = await kernel.ctx.tools.execute("adder_skill_add", {"a": 1, "b": 1})
    assert r["ok"] and r["result"]["sum"] == 2


async def test_ingest_skill_md_command_tool(kernel, tmp_path: Path) -> None:
    repo = tmp_path / "cmd-repo"
    repo.mkdir()
    (repo / "skill.md").write_text(
        "---\nname: echoskill\ndescription: echo via shell\n---\n"
        "```json\n"
        + json.dumps({"tools": [{"name": "yell", "description": "yell it",
                                 "parameters": {"type": "object", "properties": {"w": {"type": "string"}}, "required": ["w"]},
                                 "command": ["echo", "{w}!!!"]}]})
        + "\n```\n"
    )
    ingester = kernel.ctx.service("skills")
    report = await ingester.ingest(str(repo))
    assert report["ok"] and report["manifest_origin"] == "skill.md"
    r = await kernel.ctx.tools.execute("echoskill_yell", {"w": "hi"})
    assert r["ok"] and "hi!!!" in r["result"]["stdout"]


async def test_ingest_heuristic_knowledge_only(kernel, tmp_path: Path) -> None:
    repo = tmp_path / "docs-repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Magic Library\n\nTurns water into wine.\n")
    kernel.ctx.services.pop("llm", None)  # force the zero-LLM heuristic path
    ingester = kernel.ctx.service("skills")
    report = await ingester.ingest(str(repo))
    assert report["ok"]
    assert report["manifest_origin"] == "heuristic"
    assert report["tools_registered"] == []  # no executable surface proposed
    # knowledge landed in semantic memory
    memory = kernel.ctx.service("memory")
    assert memory.sm.count() >= 1


async def test_skill_ingest_tool_handler(kernel, skill_repo: Path) -> None:
    """The skill_ingest tool itself is dynamically registered and callable."""
    result = await kernel.ctx.tools.execute("skill_ingest", {"source": str(skill_repo)})
    assert result["ok"] and result["result"]["ok"] is True
