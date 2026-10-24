"""The mob database: ``pymc_bot/data/mobs.json``.

One entry per vanilla Java Edition mob with hostility, health, damage dealt to players (Easy /
Normal / Hard) and to other mobs, spawn conditions, the version it was added in, drops, taming
and a one-line tip. Values come from the Minecraft Wiki infoboxes.

It works offline and also covers mobs that mineflayer does not know yet (e.g. the 26.2 sulfur
cube), so the AI can still answer questions about them. Edit the JSON to add modded mobs: an
entry needs ``name``, ``category``, ``health``, ``added_in``, ``spawn``, ``attacks`` and ``fact``.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent / "data" / "mobs.json"

CATEGORIES = ("hostile", "neutral", "passive", "boss")
SPAWN_TIMES = ("dark", "night", "day", "any", "structure", "summoned", "built", "special", "event")
DIMENSIONS = ("overworld", "nether", "end")
DIFFICULTIES = ("easy", "normal", "hard")
SPAWN_TEXT = {
    "dark": "in darkness (light 0): at night on the surface, any time in caves",
    "night": "on the surface at night or in thunderstorms",
    "day": "with the world / in daylight on suitable blocks",
    "any": "at any time of day",
    "structure": "in structures",
    "summoned": "only when summoned",
    "built": "only when built by a player",
    "special": "only under special conditions",
    "event": "only during an event",
}

# Old versions before release 1.0.0, oldest first.
_EARLY = ("classic", "indev", "infdev", "alpha", "beta")
_STAT_WORDS = {
    "damage", "dmg", "hurt", "hit", "hits", "deal", "deals", "strong", "strength", "powerful", "health", "hp",
    "heart", "hearts", "hostile", "neutral", "passive", "friendly", "aggressive", "peaceful", "dangerous", "safe",
    "harmless", "attack", "attacks", "spawn", "spawns", "spawning", "live", "lives", "biome", "biomes", "time",
    "night", "day", "version", "versions", "added", "add", "exist", "exists", "new", "update", "release",
    "released", "drop", "drops", "loot", "xp", "experience", "tame", "taming", "breed", "breeding", "weak",
    "weakness", "weaknesses", "kill", "fight", "info", "about", "mob", "stats", "stat", "mineflayer", "supported",
    "support", "much", "many", "a", "an", "the", "is", "are", "does", "do", "it", "they", "what", "how", "when",
    "where", "which", "who", "can", "will", "there", "this", "that", "in", "on", "at", "of", "to", "by", "me",
    "tell", "minecraft", "java", "edition", "difficulty", "easy", "normal", "hard", "player", "players", "mobs",
    "other", "did", "come", "out", "first", "since", "yet", "still", "game", "find", "found", "get", "hostility",
    "and", "or", "you", "i", "my", "need", "know", "explain",
    "was", "were", "has", "have", "its", "their", "up", "if", "should", "be", "any", "all", "type", "kind",
}
_FIELD_WORDS = (({"tame", "taming"}, "tame"), ({"breed", "breeding"}, "breed"), ({"drop", "drops", "loot"}, "drops"),
                ({"xp", "experience"}, "xp"), ({"weak", "weakness", "weaknesses"}, "weaknesses"))
_VERSION_RE = re.compile(r"^(?:\d+\.\d+(?:\.\d+)?|\d{2}\.\d)$")

_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {}


def _norm(text: str) -> str:
    text = str(text or "").lower().replace("minecraft:", " ").replace("_", " ").replace("sulphur", "sulfur")
    return " ".join(re.findall(r"[a-z0-9.']+", text)).replace("'", "")


def version_key(version: str) -> tuple[int, ...]:
    """Sortable key: Classic < Indev < Infdev < Alpha < Beta < 1.0.0 < ... < 1.21.11 < 26.1 < 26.2."""
    text = str(version or "").strip().lower()
    for rank, word in enumerate(_EARLY):
        if text.startswith(word):
            numbers = [int(n) for n in re.findall(r"\d+", text)]
            return (0, rank, *numbers)
    numbers = [int(n) for n in re.findall(r"\d+", text)]
    if not numbers:
        raise ValueError(f"not a version: {version!r}")
    return (1, *numbers)


def version_at_least(version: str, minimum: str) -> bool:
    a, b = version_key(version), version_key(minimum)
    width = max(len(a), len(b))
    return a + (0,) * (width - len(a)) >= b + (0,) * (width - len(b))


def is_version(text: str) -> bool:
    return bool(_VERSION_RE.match(str(text or "").strip()))


class MobDatabase:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self.meta = {k: v for k, v in data.items() if k != "mobs"}
        self.mobs: dict[str, dict[str, Any]] = {k: {"id": k, **v} for k, v in data.get("mobs", {}).items()}
        self.names: dict[str, str] = {}
        for mob_id, mob in self.mobs.items():
            for alias in (mob_id, mob["name"], *mob.get("aliases", [])):
                for form in self._forms(_norm(alias)):
                    self.names.setdefault(form, mob_id)
        # longest names first, so "zombie horse" wins over "zombie" and "cave spider" over "spider"
        self._by_length = sorted(self.names, key=len, reverse=True)

    @staticmethod
    def _forms(name: str) -> list[str]:
        forms = [name]
        if name.endswith("man"):
            forms.append(name[:-3] + "men")
        elif name.endswith(("s", "sh", "ch")):
            forms.append(name + "es")
        elif name.endswith("y") and not name.endswith(("ay", "ey", "oy")):
            forms.append(name[:-1] + "ies")
        else:
            forms.append(name + "s")
        return forms

    @property
    def mineflayer_max(self) -> str:
        return str(self.meta.get("mineflayer_max_version") or "26.1")

    def __len__(self) -> int:
        return len(self.mobs)

    # ------------------------------------------------------------- lookups
    def get(self, name: str) -> dict[str, Any] | None:
        """A mob by id, name, alias or plural ("Sulphur Cubes", "minecraft:zombified_piglin", "zombie pigman")."""
        key = self.names.get(_norm(name))
        return self.mobs.get(key) if key else None

    def find_in_text(self, text: str) -> list[tuple[str, tuple[int, int]]]:
        """Mobs named in a sentence (longest match first, no overlaps): [(id, (start, end)), ...]."""
        words = _norm(text)
        found: list[tuple[str, tuple[int, int]]] = []
        taken: list[tuple[int, int]] = []
        for name in self._by_length:
            for match in re.finditer(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", words):
                span = match.span()
                if any(span[0] < end and start < span[1] for start, end in taken):
                    continue
                taken.append(span)
                found.append((self.names[name], span))
        found.sort(key=lambda item: item[1][0])
        return found

    def answerable(self, question: str) -> dict[str, Any] | None:
        """The mob a question is about when the database can answer it on its own.

        "how much damage does a warden do", "is the sulphur cube hostile", "when was the camel
        added" -> yes. "what drops blaze rods", "warden farm design" -> no (ask the web).
        """
        words = _norm(question)
        found = self.find_in_text(question)
        if not found:
            return None
        rest = words
        for _, (start, end) in sorted(found, key=lambda item: item[1][0], reverse=True):
            rest = rest[:start] + " " + rest[end:]
        leftover = [w for w in (w.strip(".") for w in rest.split())
                    if w and w not in _STAT_WORDS and not is_version(w) and not w.isdigit()]
        if leftover:
            return None
        mob = self.mobs[found[0][0]]
        asked = set(rest.split())
        for words, key in _FIELD_WORDS:
            if asked & words and not mob.get(key):
                return None  # "how to tame a fox": the entry cannot answer that, the wiki can
        return mob

    def supported_by_mineflayer(self, mob: dict[str, Any]) -> bool:
        return not version_at_least(mob["added_in"], self._after(self.mineflayer_max))

    @staticmethod
    def _after(version: str) -> str:
        parts = [int(n) for n in re.findall(r"\d+", version)]
        parts[-1] += 1
        return ".".join(map(str, parts))

    def exists_in(self, mob: dict[str, Any], version: str) -> bool:
        return version_at_least(version, mob["added_in"])

    def newer_than_mineflayer(self, server_version: str | None = None) -> list[dict[str, Any]]:
        """Mobs mineflayer cannot name, optionally limited to those a server version has."""
        result = [m for m in self.mobs.values() if not self.supported_by_mineflayer(m)]
        if server_version and is_version(server_version):
            result = [m for m in result if self.exists_in(m, server_version)]
        return result

    def search(self, *, category: str | None = None, hostile: bool | None = None, dimension: str | None = None,
               spawn_time: str | None = None, added_since: str | None = None, exists_in: str | None = None,
               query: str | None = None) -> list[dict[str, Any]]:
        result = []
        for mob in self.mobs.values():
            if category and mob["category"] != category:
                continue
            if hostile is not None and bool(mob["hostile"]) != hostile:
                continue
            if dimension and dimension not in mob["spawn"].get("dimensions", []):
                continue
            if spawn_time and mob["spawn"].get("when") != spawn_time:
                continue
            if added_since and not version_at_least(mob["added_in"], added_since):
                continue
            if exists_in and not self.exists_in(mob, exists_in):
                continue
            if query and _norm(query) not in _norm(" ".join([mob["id"], mob["name"], *mob.get("aliases", [])])):
                continue
            result.append(mob)
        return result

    # ------------------------------------------------------------- damage
    @staticmethod
    def to_mobs(attack: dict[str, Any]) -> Any:
        """Damage to other mobs: difficulty only scales damage to players, so it is the Normal value."""
        return attack.get("to_mobs", attack.get("normal"))

    @staticmethod
    def max_damage(mob: dict[str, Any], difficulty: str = "normal") -> float:
        best = 0.0
        for attack in mob.get("attacks", []):
            value = attack.get(difficulty)
            value = max(value) if isinstance(value, list) else value
            if isinstance(value, (int, float)):
                best = max(best, float(value))
        return best

    def with_derived(self, mob: dict[str, Any]) -> dict[str, Any]:
        """The entry plus computed fields (API/CLI output)."""
        out = dict(mob)
        out["mineflayer_supported"] = self.supported_by_mineflayer(mob)
        out["attacks"] = [{**a, "to_mobs": self.to_mobs(a)} for a in mob.get("attacks", [])]
        return out

    # ------------------------------------------------------------- text
    @staticmethod
    def _amount(value: Any) -> str:
        if isinstance(value, list):
            return "-".join(_fmt(v) for v in value)
        return _fmt(value)

    def attack_text(self, attack: dict[str, Any]) -> str:
        line = (f"{attack['type']} {self._amount(attack['easy'])}/{self._amount(attack['normal'])}/"
                f"{self._amount(attack['hard'])} HP (easy/normal/hard) to players, {self._amount(self.to_mobs(attack))}"
                " to mobs")
        if attack.get("effect"):
            line += f" + {attack['effect']}"
        if attack.get("note"):
            line += f" ({attack['note']})"
        return line

    def describe(self, mob: dict[str, Any], version: str | None = None) -> str:
        """A few sentences with everything the database knows about a mob (lookup answers)."""
        parts = [f"{mob['name']}: {mob['category']} mob" + (" - attacks players on sight" if mob["hostile"]
                                                               else ", does not attack on sight")]
        if mob.get("hostile_when"):
            parts.append(f"Turns hostile: {mob['hostile_when']}")
        health = f"Health {_fmt(mob['health'])} HP"
        if mob.get("health_note"):
            health += f" ({mob['health_note']})"
        if mob.get("mob_type"):
            health += f", {mob['mob_type']}"
        parts.append(health)
        if mob.get("attacks"):
            parts.append("Damage: " + "; ".join(self.attack_text(a) for a in mob["attacks"]))
        else:
            parts.append("Damage: none, it cannot attack")
        spawn = mob["spawn"]
        parts.append(f"Spawns {SPAWN_TEXT.get(spawn.get('when'), spawn.get('when'))}: {spawn['where']}"
                     + (f" (groups of {spawn['group']})" if spawn.get("group") else "")
                     + f" [{', '.join(spawn.get('dimensions', []))}]")
        added = f"Added in Java {mob['added_in']}"
        if not self.supported_by_mineflayer(mob):
            added += (f" - newer than mineflayer (supports up to {self.mineflayer_max}), so a bot sees it as an "
                      "unknown entity")
        parts.append(added)
        if version and is_version(version):
            exists = self.exists_in(mob, version)
            parts.append(f"In {version}: {'yes, it exists' if exists else 'no, it does not exist yet'}")
        for key, label in (("drops", "Drops"), ("weaknesses", "Weak to")):
            if mob.get(key):
                parts.append(f"{label}: {', '.join(mob[key])}")
        if mob.get("xp") is not None:
            parts.append(f"XP: {mob['xp']}")
        for key, label in (("tame", "Tame with"), ("breed", "Breed with"), ("notes", "Notes")):
            if mob.get(key):
                parts.append(f"{label}: {mob[key]}")
        parts.append(f"Tip: {mob['fact']}")
        return ". ".join(p.rstrip(".") for p in parts) + "."

    def prompt_line(self, mob: dict[str, Any]) -> str:
        """One short line for the turn prompt: "creeper (hostile, 20 HP, up to 43 dmg): <tip>"."""
        bits = [mob["category"], f"{_fmt(mob['health'])} HP"]
        top = self.max_damage(mob)
        if top:
            bits.append(f"up to {_fmt(top)} dmg")
        when = mob["spawn"].get("when")
        if mob["hostile"] and when in ("dark", "night"):
            bits.append("night/dark")
        return f"{mob['id'].replace('_', ' ')} ({', '.join(bits)}): {mob['fact']}"


def _fmt(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def load_mobs(path: Path | None = None) -> MobDatabase:
    """The mob database (read once per path and shared)."""
    path = Path(path or DB_PATH)
    key = str(path.resolve())
    with _LOCK:
        db = _CACHE.get(key)
        if db is None:
            db = _CACHE[key] = MobDatabase(json.loads(path.read_text(encoding="utf-8")))
        return db


def validate(data: dict[str, Any]) -> list[str]:
    """Problems in a mobs.json document (empty list = valid). Used by the tests and the CLI."""
    problems: list[str] = []
    mobs = data.get("mobs")
    if not isinstance(mobs, dict) or not mobs:
        return ["'mobs' must be a non-empty object"]
    for mob_id, mob in mobs.items():
        where = f"mobs.{mob_id}"
        if not re.fullmatch(r"[a-z0-9_]+", mob_id):
            problems.append(f"{where}: id must be lower_snake_case")
        for key in ("name", "category", "hostile", "behavior", "health", "added_in", "spawn", "attacks", "fact"):
            if key not in mob:
                problems.append(f"{where}: missing {key}")
        if problems and problems[-1].startswith(where + ": missing"):
            continue
        if mob["category"] not in CATEGORIES:
            problems.append(f"{where}: bad category {mob['category']!r}")
        if not isinstance(mob["hostile"], bool):
            problems.append(f"{where}: hostile must be true/false")
        if not isinstance(mob["health"], (int, float, str)) or (isinstance(mob["health"], (int, float)) and mob["health"] <= 0):
            problems.append(f"{where}: bad health")
        try:
            version_key(mob["added_in"])
        except ValueError:
            problems.append(f"{where}: bad added_in {mob['added_in']!r}")
        spawn = mob["spawn"]
        if spawn.get("when") not in SPAWN_TIMES or not spawn.get("where"):
            problems.append(f"{where}: spawn needs when (one of {SPAWN_TIMES}) and where")
        if not spawn.get("dimensions") or any(d not in DIMENSIONS for d in spawn["dimensions"]):
            problems.append(f"{where}: bad spawn.dimensions")
        for attack in mob["attacks"]:
            if not attack.get("type"):
                problems.append(f"{where}: attack without type")
            for level in DIFFICULTIES:
                value = attack.get(level)
                ok = isinstance(value, (int, float)) or (
                    isinstance(value, list) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value)
                    and value[0] <= value[1])
                if not ok:
                    problems.append(f"{where}: attack {attack.get('type')!r} has bad {level} damage")
        if not isinstance(mob["fact"], str) or not mob["fact"] or "{" in mob["fact"] or "}" in mob["fact"]:
            problems.append(f"{where}: fact must be a plain sentence")
        if not isinstance(mob.get("aliases", []), list):
            problems.append(f"{where}: aliases must be a list")
    return problems
