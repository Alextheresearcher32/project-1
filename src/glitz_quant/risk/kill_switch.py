"""
Global kill switch. The orchestrator polls the file at KILL_SWITCH_PATH
every tick. If it exists, halt accepting new signals and cancel everything.
"""

from __future__ import annotations

from pathlib import Path

from glitz_quant.settings import get_settings


def is_killed() -> bool:
    s = get_settings()
    return Path(s.kill_switch_path).exists()


def arm() -> None:
    s = get_settings()
    p = Path(s.kill_switch_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()


def disarm() -> None:
    s = get_settings()
    p = Path(s.kill_switch_path)
    if p.exists():
        p.unlink()
