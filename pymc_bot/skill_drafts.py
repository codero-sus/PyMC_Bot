"""Operator-reviewed, data-only skill drafts. Generated Python is never executed by the bot."""
from __future__ import annotations

import hashlib
import json
import keyword
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

from pymc_bot import skills
from pymc_bot.experience import SENSITIVE

# Deliberately excludes player attacks, commands, arbitrary code, transfers and recursion.
ALLOWED = frozenset({"equip_weapon", "equip_tool", "chop_tree", "mine_ore", "gather", "harvest",
                     "plant", "till", "craft", "smelt", "place_torch", "light_up", "build_shelter"})
NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
VALUE = re.compile(r"^[a-zA-Z0-9_:./ -]{1,64}$")
_LOCK = threading.RLock()
MAX_DRAFTS = 16

# Skills can move, equip tools, place stations or replant in addition to their primary permission.
REQUIRED_FLAGS = {
    "equip_weapon": ("allow_inventory",), "equip_tool": ("allow_inventory",),
    "chop_tree": ("allow_mining", "allow_movement", "allow_inventory"),
    "mine_ore": ("allow_mining", "allow_movement", "allow_inventory"),
    "gather": ("allow_mining", "allow_movement", "allow_inventory"),
    "harvest": ("allow_mining", "allow_movement", "allow_inventory", "allow_building"),
    "plant": ("allow_building", "allow_movement", "allow_inventory"),
    "till": ("allow_building", "allow_movement", "allow_inventory"),
    "craft": ("allow_inventory", "allow_movement", "allow_building"),
    "smelt": ("allow_inventory", "allow_movement", "allow_building"),
    "place_torch": ("allow_inventory", "allow_movement", "allow_building"),
    "light_up": ("allow_inventory", "allow_movement", "allow_building"),
    "build_shelter": ("allow_inventory", "allow_movement", "allow_building"),
}


def validate(name: Any, steps: Any) -> dict[str, Any]:
    if not isinstance(name, str) or not NAME.fullmatch(name) or keyword.iskeyword(name):
        raise ValueError("name must be 1–32 lowercase letters, numbers or underscores, starting with a letter")
    if not isinstance(steps, list) or not 1 <= len(steps) <= 6:
        raise ValueError("a draft needs 1–6 steps")
    result = []
    for step in steps:
        if not isinstance(step, dict) or set(step) - {"action", "params"}:
            raise ValueError("each step must contain only action and params")
        action, params = step.get("action"), step.get("params", {})
        if not isinstance(action, str) or action not in ALLOWED or not isinstance(params, dict):
            raise ValueError("unsupported draft step")
        skill = skills.get(action)
        if not skill or skill.manual_only or set(params) - set(skill.params):
            raise ValueError(f"invalid parameters for {action}")
        for key, value in params.items():
            if isinstance(value, str):
                if not VALUE.fullmatch(value) or value.startswith("/") or SENSITIVE.search(value):
                    raise ValueError("invalid or sensitive parameter text")
                if ":" in value and not value.lower().startswith("minecraft:"):
                    raise ValueError("mod namespaces cannot be safely preserved by this skill executor")
            elif isinstance(value, bool):
                if skill.params[key].kind != "bool":
                    raise ValueError("invalid boolean parameter")
            elif not isinstance(value, (int, float)):
                raise ValueError("parameters must be simple text or numbers")
        cleaned = skills.clean_params(skill, params)
        if set(params) - set(cleaned):
            raise ValueError("invalid parameter values")
        result.append({"action": action, "params": cleaned})
    return {"name": name, "steps": result, "status": "draft"}


class DraftStore:
    def __init__(self, directory: Path, identity: str) -> None:
        key = hashlib.sha256(identity.lower().encode()).hexdigest()[:24]
        self.path = directory / "skill-drafts" / f"{key}.json"

    def _read(self) -> dict[str, Any]:
        try:
            if self.path.stat().st_size > 65536:
                raise ValueError("draft store is too large")
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        if not isinstance(raw, dict) or len(raw) > MAX_DRAFTS:
            raise ValueError("invalid draft store")
        # Revalidate disk data too; editing a file must not bypass step restrictions.
        return {key: validate(key, val.get("steps")) for key, val in raw.items() if isinstance(val, dict)}

    def list(self) -> list[dict[str, Any]]:
        with _LOCK:
            return list(self._read().values())

    def get(self, name: str) -> dict[str, Any]:
        with _LOCK:
            result = self._read().get(name)
            if result is None:
                raise KeyError(name)
            return result

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".draft-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                json.dump(data, out, ensure_ascii=False, separators=(",", ":"))
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def save(self, name: Any, steps: Any) -> dict[str, Any]:
        draft = validate(name, steps)
        with _LOCK:
            data = self._read()
            if draft["name"] not in data and len(data) >= MAX_DRAFTS:
                raise ValueError("draft limit reached; delete a draft first")
            data[draft["name"]] = draft
            self._write(data)
        return draft

    def delete(self, name: str) -> bool:
        with _LOCK:
            data = self._read()
            if name not in data:
                return False
            del data[name]
            if not data:
                self.path.unlink(missing_ok=True)
            else:
                self._write(data)
            return True


def python_code(draft: dict[str, Any]) -> str:
    """Generate reviewable Python; this text is never evaluated by PyMC Bot."""
    draft = validate(draft.get("name"), draft.get("steps"))
    return ("# Draft only. Review before running; PyMC Bot never executes this source.\n"
            "import math\n"
            "from pymc_bot.agent import Decision\n"
            "from pymc_bot.skill_drafts import REQUIRED_FLAGS\n\n"
            f"def {draft['name']}(agent):\n"
            f"    steps = {draft['steps']!r}\n"
            "    identity = (*agent.bot.memory_server, agent.bot.username.lower())\n"
            "    with agent.bot.action_scope():\n"
            "        return _run_steps(agent, steps, identity)\n\n"
            "def _run_steps(agent, steps, identity):\n"
            "    for step in steps:\n"
            "        state = agent.bot.snapshot()\n"
            "        health = state.get('health')\n"
            "        if (not agent.bot.connected or agent.bot.cancelled or\n"
            "            identity != (*agent.bot.memory_server, agent.bot.username.lower()) or\n"
            "            not isinstance(health, (int, float)) or not math.isfinite(health) or health <= 8 or state.get('nearby_hazards')):\n"
            "            return False, 'cancelled, identity changed, or unsafe state'\n"
            "        if any(not getattr(agent.store.settings.agent, flag) for flag in REQUIRED_FLAGS[step['action']]):\n"
            "            return False, 'required permissions disabled'\n"
            "        ok, detail = agent.execute(Decision(step['action'], step['params'], source='operator:draft'),\n"
            "                                   enforce_permissions=True)\n"
            "        if not ok:\n"
            "            return False, detail\n"
            "    return True, 'draft completed'\n")
