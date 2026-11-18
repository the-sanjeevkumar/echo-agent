"""Local, dependency-free system metrics for the server-ops pack (PHASE 6).

Reads /proc, os.statvfs and bounded subprocess calls (systemctl / journalctl /
docker). No local LLM, no pip dependency, no shell=True anywhere.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


def _read_proc_stat_cpu() -> tuple[int, int]:
    try:
        line = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
        fields = [int(x) for x in line.split()[1:8]]
        idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
        return sum(fields), idle
    except (OSError, ValueError, IndexError):
        return 0, 0


def cpu_percent(interval: float = 0.05) -> float:
    total1, idle1 = _read_proc_stat_cpu()
    time.sleep(max(0.0, interval))
    total2, idle2 = _read_proc_stat_cpu()
    total_delta, idle_delta = total2 - total1, idle2 - idle1
    if total_delta <= 0:
        return 0.0
    return round(100.0 * (1.0 - idle_delta / total_delta), 1)


def ram_metrics() -> dict[str, float]:
    total = available = 0
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                total = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                available = int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    total_mb, available_mb = total / 1024, available / 1024
    percent = round(100.0 * (1.0 - available / total), 1) if total else 0.0
    return {"total_mb": round(total_mb, 1), "available_mb": round(available_mb, 1), "percent": percent}


def disk_usage(path: str = "/") -> dict[str, Any]:
    target = Path(path).expanduser()
    usage = shutil.disk_usage(str(target))
    return {
        "path": str(target),
        "total_gb": round(usage.total / 1024 ** 3, 2),
        "used_gb": round(usage.used / 1024 ** 3, 2),
        "free_gb": round(usage.free / 1024 ** 3, 2),
        "percent": round(100.0 * usage.used / usage.total, 1) if usage.total else 0.0,
    }


def uptime_seconds() -> float:
    try:
        return float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0])
    except (OSError, ValueError, IndexError):
        return 0.0


def sys_metrics() -> dict[str, Any]:
    ram = ram_metrics()
    disk = disk_usage("/")
    try:
        load = [round(v, 2) for v in os.getloadavg()]
    except (OSError, AttributeError):
        load = []
    return {
        "cpu_percent": cpu_percent(),
        "ram": ram,
        "disk": disk,
        "load_avg": load,
        "uptime_seconds": round(uptime_seconds(), 1),
    }


def unit_status(unit: str) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["systemctl", "is-active", unit], capture_output=True, text=True, timeout=15, check=False
        )
        active = proc.stdout.strip()
        return {"unit": unit, "active": active, "ok": active == "active"}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"unit": unit, "active": "unknown", "ok": False, "error": str(exc)[:200]}


def unit_restart(unit: str) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["systemctl", "restart", unit], capture_output=True, text=True, timeout=60, check=False
        )
        return {
            "unit": unit, "ok": proc.returncode == 0, "returncode": proc.returncode,
            "stderr": proc.stderr.strip()[-500:],
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"unit": unit, "ok": False, "error": str(exc)[:200]}


def journal_tail(unit: str, lines: int = 50) -> dict[str, Any]:
    count = max(1, min(int(lines), 500))
    try:
        proc = subprocess.run(
            ["journalctl", "-u", unit, "-n", str(count), "--no-pager"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return {"unit": unit, "lines": count, "text": proc.stdout[-8000:], "ok": proc.returncode == 0}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"unit": unit, "ok": False, "error": str(exc)[:200]}


def process_list(limit: int = 20) -> dict[str, Any]:
    processes: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")[:160]
            stat = (entry / "stat").read_text(encoding="utf-8").split()
            rss_pages = int(stat[23]) if len(stat) > 23 else 0
            processes.append({
                "pid": int(entry.name), "rss_mb": round(rss_pages * os.sysconf("SC_PAGE_SIZE") / 1024 ** 2, 1),
                "command": cmd or stat[1] if len(stat) > 1 else cmd,
            })
        except (OSError, ValueError, IndexError):
            continue
    processes.sort(key=lambda p: p["rss_mb"], reverse=True)
    return {"processes": processes[: max(1, min(int(limit), 100))], "total": len(processes)}


def docker_ps() -> dict[str, Any]:
    if not shutil.which("docker"):
        return {"ok": False, "error": "docker CLI not available"}
    try:
        proc = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Status}}\t{{.Ports}}"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return {"ok": proc.returncode == 0, "containers": proc.stdout.strip().splitlines()[:100]}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "error": str(exc)[:200]}
