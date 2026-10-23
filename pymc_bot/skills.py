"""Skills: 44 extra things the bot can do so it can live on a server on its own.

Every skill has a name the AI (and the panel, and the mod's ``/pymcbot bot`` command) uses, a
group, a one-line summary, an example, and the permission switch that gates it for the AI. Most run
inside the Minecraft bridge (``minecraft_bridge.js`` has one command per skill, the simulated world
fakes them); a few are plain Python on top of the existing actions (homes, waypoints, guard...).

    survival   equip_weapon equip_tool totem shield drink surface extinguish respawn toss_junk
    gathering  find_block chop_tree mine_ore gather
    farming    harvest plant till bonemeal fish shear milk breed
    crafting   craft smelt deposit withdraw chest_contents trade
    building   place_torch light_up build_shelter pillar_up activate
    moving     set_home go_home set_waypoint goto_waypoint waypoints come ride dismount
    combat     guard hunt shoot
    social     report

Made with love by codero-sus - https://github.com/codero-sus - https://the-codero.web.app/
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pymc_bot.inventory import FOODS, Inventory, armor_slot_for, short, tool_kind

if TYPE_CHECKING:  # pragma: no cover
    from pymc_bot.bot import MinecraftBot

#: AI permission switch per skill permission (``none`` = always allowed).
PERMISSION_FLAGS: dict[str, str] = {
    "movement": "allow_movement",
    "mining": "allow_mining",
    "building": "allow_building",
    "inventory": "allow_inventory",
    "fighting": "allow_mob_fighting",
    "chat": "allow_chat",
}

#: What a player throws away when the inventory is full.
JUNK: frozenset[str] = frozenset({
    "rotten_flesh", "poisonous_potato", "spider_eye", "dead_bush", "wheat_seeds", "beetroot_seeds", "pumpkin_seeds",
    "melon_seeds", "gravel", "flint", "andesite", "diorite", "granite", "tuff", "netherrack", "cobbled_deepslate",
    "string", "bone", "feather", "egg", "kelp", "seagrass", "lily_pad", "stick", "short_grass", "tall_grass", "fern",
})
#: Kept even when it is junk-ish: one stack of building blocks is always useful.
KEEP_ONE_STACK: frozenset[str] = frozenset({"cobblestone", "dirt", "cobbled_deepslate", "netherrack"})


@dataclass(frozen=True)
class Param:
    kind: str = "str"  # str, int, float, bool
    default: Any = None
    low: float | None = None
    high: float | None = None


@dataclass(frozen=True)
class Skill:
    name: str
    group: str
    summary: str
    example: dict[str, Any] = field(default_factory=dict)
    permission: str = "none"
    bridge: str | None = None  # bridge command; None = Python skill (see run_skill)
    params: dict[str, Param] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    timeout: float = 30.0
    moves: bool = True  # holds the bot's motion lock while it runs

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "group": self.group, "summary": self.summary, "permission": self.permission,
            "example": {"action": self.name, **self.example}, "aliases": list(self.aliases),
            "params": {k: {"type": p.kind, "default": p.default} for k, p in self.params.items()},
        }


def _p(kind: str = "str", default: Any = None, low: float | None = None, high: float | None = None) -> Param:
    return Param(kind, default, low, high)


SKILLS: tuple[Skill, ...] = (
    # ---------------------------------------------------------------- survival
    Skill("equip_weapon", "survival", "hold your best weapon", {}, "inventory", moves=False, aliases=("best_weapon",)),
    Skill("equip_tool", "survival", "hold the right tool for a block", {"block": "stone"}, "inventory",
          params={"block": _p("str", "stone")}, moves=False, aliases=("best_tool",)),
    Skill("totem", "survival", "put a totem of undying in the off-hand", {}, "inventory", moves=False),
    Skill("shield", "survival", "shield to the off-hand and block for a moment", {"seconds": 2}, "fighting",
          params={"seconds": _p("float", 2.0, 0.5, 5.0)}, moves=False, aliases=("block_attack",)),
    Skill("drink", "survival", "drink a potion or milk", {"item": "potion"}, "inventory", "drink",
          {"item": _p("str", "")}, moves=False, timeout=12.0),
    Skill("surface", "survival", "swim up for air when under water", {}, "movement", "surface", timeout=15.0,
          aliases=("swim_up",)),
    Skill("extinguish", "survival", "put out fire: water bucket or the nearest water", {}, "movement", "extinguish",
          timeout=15.0),
    Skill("respawn", "survival", "respawn after dying", {}, "none", "respawn", moves=False, timeout=10.0),
    Skill("toss_junk", "survival", "throw away junk (rotten flesh, gravel, seeds...) to free space", {}, "inventory",
          moves=False, aliases=("clean_inventory",)),
    # --------------------------------------------------------------- gathering
    Skill("find_block", "gathering", "find the nearest block of a type and say where it is", {"block": "iron_ore"},
          "none", "find_block", {"block": _p("str", "oak_log"), "radius": _p("int", 64, 4, 128)}, moves=False,
          aliases=("locate",), timeout=15.0),
    Skill("chop_tree", "gathering", "chop down the nearest tree and pick up the wood", {"count": 16}, "mining",
          "chop_tree", {"count": _p("int", 16, 1, 64), "radius": _p("int", 32, 4, 64)}, timeout=120.0,
          aliases=("chop", "cut_tree")),
    Skill("mine_ore", "gathering", "mine the nearest ore (coal, iron, gold, diamond...)", {"ore": "iron", "count": 4},
          "mining", "mine_ore", {"ore": _p("str", ""), "count": _p("int", 4, 1, 32), "radius": _p("int", 24, 4, 48)},
          timeout=150.0, aliases=("mine_ores",)),
    Skill("gather", "gathering", "mine several blocks of one type", {"block": "sand", "count": 16}, "mining", "gather",
          {"block": _p("str", "dirt"), "count": _p("int", 8, 1, 64), "radius": _p("int", 24, 4, 48)}, timeout=150.0,
          aliases=("collect_blocks",)),
    # ----------------------------------------------------------------- farming
    Skill("harvest", "farming", "harvest ripe crops nearby and replant them", {}, "mining", "harvest",
          {"radius": _p("int", 12, 2, 32), "replant": _p("bool", True)}, timeout=120.0, aliases=("farm",)),
    Skill("plant", "farming", "plant seeds on empty farmland", {"seed": "wheat_seeds"}, "building", "plant",
          {"seed": _p("str", ""), "count": _p("int", 16, 1, 64), "radius": _p("int", 10, 2, 24)}, timeout=90.0,
          aliases=("sow",)),
    Skill("till", "farming", "turn dirt into farmland with a hoe", {"count": 8}, "building", "till",
          {"count": _p("int", 8, 1, 32), "radius": _p("int", 6, 2, 16)}, timeout=90.0, aliases=("hoe",)),
    Skill("bonemeal", "farming", "use bone meal on crops and saplings", {"count": 4}, "building", "bonemeal",
          {"count": _p("int", 4, 1, 32), "radius": _p("int", 8, 2, 16)}, timeout=60.0),
    Skill("fish", "farming", "fish with a fishing rod for a while", {"seconds": 60}, "inventory", "fish",
          {"seconds": _p("int", 60, 10, 300)}, timeout=320.0, aliases=("go_fishing",)),
    Skill("shear", "farming", "shear the sheep nearby", {}, "inventory", "shear",
          {"count": _p("int", 4, 1, 16), "radius": _p("int", 16, 2, 32)}, timeout=60.0),
    Skill("milk", "farming", "milk a cow with a bucket", {}, "inventory", "milk", {"radius": _p("int", 16, 2, 32)},
          timeout=30.0),
    Skill("breed", "farming", "feed two animals so they breed", {"animal": "cow"}, "inventory", "breed",
          {"animal": _p("str", ""), "radius": _p("int", 16, 2, 32)}, timeout=60.0),
    # ---------------------------------------------------------------- crafting
    Skill("craft", "crafting", "craft an item (uses or places a crafting table)", {"item": "stone_pickaxe", "count": 1},
          "inventory", "craft", {"item": _p("str", ""), "count": _p("int", 1, 1, 64)}, timeout=40.0),
    Skill("smelt", "crafting", "smelt or cook in a furnace (raw iron, beef, sand...)", {"item": "raw_iron", "count": 8},
          "inventory", "smelt", {"item": _p("str", ""), "count": _p("int", 8, 1, 64),
                                 "seconds": _p("int", 90, 10, 600)}, timeout=620.0, aliases=("cook",)),
    Skill("deposit", "crafting", "put items into the nearest chest (keeps tools, armour, food)", {}, "inventory",
          "deposit", {"item": _p("str", ""), "all": _p("bool", False)}, timeout=40.0, aliases=("store",)),
    Skill("withdraw", "crafting", "take items out of the nearest chest", {"item": "bread", "count": 8}, "inventory",
          "withdraw", {"item": _p("str", ""), "count": _p("int", 16, 1, 2304)}, timeout=40.0, aliases=("take",)),
    Skill("chest_contents", "crafting", "look into the nearest chest", {}, "none", "chest_contents", timeout=30.0,
          aliases=("open_chest", "check_chest")),
    Skill("trade", "crafting", "trade with a villager (no index = list the offers)", {"index": 0, "times": 1},
          "inventory", "trade", {"index": _p("int", None, 0, 30), "times": _p("int", 1, 1, 12)}, timeout=40.0),
    # ---------------------------------------------------------------- building
    Skill("place_torch", "building", "place a torch next to you", {}, "building", "place_torch", timeout=15.0,
          aliases=("torch",)),
    Skill("light_up", "building", "place torches on dark spots around you", {"radius": 8}, "building", "light_up",
          {"radius": _p("int", 8, 3, 16), "count": _p("int", 6, 1, 32)}, timeout=120.0, aliases=("light_area",)),
    Skill("build_shelter", "building", "wall yourself in with a roof (emergency night shelter)", {}, "building",
          "build_shelter", {"block": _p("str", "")}, timeout=90.0, aliases=("shelter",)),
    Skill("pillar_up", "building", "jump and place blocks under you to tower up", {"height": 4}, "building",
          "pillar_up", {"height": _p("int", 4, 1, 32), "block": _p("str", "")}, timeout=60.0, aliases=("tower",)),
    Skill("activate", "building", "use a door, button, lever, trapdoor or gate", {"block": "door"}, "movement",
          "activate", {"block": _p("str", ""), "x": _p("float"), "y": _p("float"), "z": _p("float")}, timeout=25.0,
          aliases=("open_door", "press", "use_block")),
    # ------------------------------------------------------------------ moving
    Skill("set_home", "moving", "remember this spot as home", {}, "none", moves=False, aliases=("sethome",)),
    Skill("go_home", "moving", "walk back home", {}, "movement", aliases=("home",), timeout=120.0),
    Skill("set_waypoint", "moving", "remember this spot under a name", {"name": "mine"}, "none",
          params={"name": _p("str", "spot")}, moves=False, aliases=("mark",)),
    Skill("goto_waypoint", "moving", "walk to a remembered spot", {"name": "mine"}, "movement",
          params={"name": _p("str", "")}, timeout=120.0, aliases=("waypoint",)),
    Skill("waypoints", "moving", "list remembered spots", {}, "none", moves=False),
    Skill("come", "moving", "walk over to a player", {"player": "Steve"}, "movement",
          params={"player": _p("str", "")}, timeout=60.0, aliases=("come_here", "tp_walk")),
    Skill("ride", "moving", "get into the nearest boat or minecart, or on a horse", {}, "movement", "ride",
          timeout=20.0, aliases=("mount",)),
    Skill("dismount", "moving", "get out / off", {}, "movement", "dismount", moves=False, timeout=8.0),
    # ------------------------------------------------------------------ combat
    Skill("guard", "combat", "stay here (or near a player) and fight off hostile mobs", {"seconds": 60}, "fighting",
          params={"seconds": _p("int", 60, 10, 600), "player": _p("str", ""), "radius": _p("int", 10, 4, 24)},
          timeout=620.0, aliases=("protect",)),
    Skill("hunt", "combat", "hunt an animal for food and pick up the drops", {"animal": "cow"}, "fighting",
          params={"animal": _p("str", "")}, timeout=60.0),
    Skill("shoot", "combat", "shoot a mob with a bow", {"mob": "skeleton"}, "fighting", "shoot",
          {"mob": _p("str", ""), "range": _p("int", 24, 4, 48), "shots": _p("int", 3, 1, 10)}, timeout=40.0,
          aliases=("bow",)),
    # ------------------------------------------------------------------ social
    Skill("report", "social", "say your health, food, position and inventory in chat", {}, "chat", moves=False,
          aliases=("status_report", "stats")),
)

SKILL_INDEX: dict[str, Skill] = {}
for _skill in SKILLS:
    SKILL_INDEX[_skill.name] = _skill
for _skill in SKILLS:
    for _alias in _skill.aliases:
        SKILL_INDEX.setdefault(_alias, _skill)

#: Bridge commands the skills add (the backends accept these next to the classic life commands).
SKILL_COMMANDS: tuple[str, ...] = tuple(sorted({s.bridge for s in SKILLS if s.bridge}))
GROUPS: tuple[str, ...] = ("survival", "gathering", "farming", "crafting", "building", "moving", "combat", "social")


def get(name: str) -> Skill | None:
    return SKILL_INDEX.get(str(name or "").strip().lower())


def catalogue() -> list[dict[str, Any]]:
    return [skill.as_dict() for skill in SKILLS]


def action_examples() -> dict[str, str]:
    """``{name: "summary: {json example}"}`` - merged into the agent's ACTIONS table."""
    return {s.name: f'{s.summary}: {json.dumps({"action": s.name, **s.example})}' for s in SKILLS}


#: Permission defaults of the prompt (same as the AI's switches out of the box: building is off).
_PROMPT_DEFAULTS = {"movement": True, "mining": True, "building": False, "inventory": True, "fighting": True, "chat": True}


def prompt_reference(allowed: dict[str, bool] | None = None) -> str:
    """The skills block of the LLM system prompt: one compact line per skill the switches allow."""
    allowed = {**_PROMPT_DEFAULTS, **(allowed or {})}
    lines = [" skills (bigger jobs that take a while - pick one and let it finish)"]
    for skill in SKILLS:
        if skill.permission != "none" and not allowed.get(skill.permission, True):
            continue
        example = json.dumps({"action": skill.name, **skill.example}, separators=(", ", ": "))
        lines.append(f"  {skill.name:<14} {example:<52} {skill.summary}")
    return "\n".join(lines)


def allowed(skill: Skill, agent: Any) -> bool:
    flag = PERMISSION_FLAGS.get(skill.permission)
    return True if flag is None else bool(getattr(agent, flag, True))


def clean_params(skill: Skill, params: dict[str, Any] | None) -> dict[str, Any]:
    """Coerce and clamp what the AI / a person typed (``"5"`` -> 5, ``count: 9999`` -> the maximum)."""
    params = dict(params or {})
    out: dict[str, Any] = {}
    synonyms = {"block": ("block", "item", "name", "type", "target"), "item": ("item", "name", "block", "type"),
                "ore": ("ore", "block", "type", "name"), "animal": ("animal", "mob", "type", "name", "target"),
                "mob": ("mob", "target", "type", "name"), "player": ("player", "name", "to", "target"),
                "seed": ("seed", "item", "name"), "name": ("name", "waypoint", "place", "label"),
                "count": ("count", "amount", "n", "number"), "seconds": ("seconds", "time", "duration")}
    for key, spec in skill.params.items():
        value = None
        for alias in synonyms.get(key, (key,)):
            if params.get(alias) not in (None, ""):
                value = params[alias]
                break
        if value is None:
            if spec.default is not None:
                out[key] = spec.default
            continue
        try:
            if spec.kind == "int":
                value = int(float(value))
            elif spec.kind == "float":
                value = float(value)
            elif spec.kind == "bool":
                value = value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "yes", "on")
            else:
                value = short(str(value)).replace(" ", "_")[:64]
        except (TypeError, ValueError):
            if spec.default is not None:
                out[key] = spec.default
            continue
        if spec.kind in ("int", "float"):
            if spec.low is not None:
                value = max(type(value)(spec.low), value)
            if spec.high is not None:
                value = min(type(value)(spec.high), value)
        out[key] = value
    return out


# ---------------------------------------------------------------------------- places (homes, waypoints)
class Places:
    """Homes and named waypoints per server and bot, kept in ``pymc_bot_places.json`` next to the config."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def book(self, server: str, username: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._load().get(server, {}).get(username.lower(), {}))

    def set(self, server: str, username: str, name: str, spot: dict[str, Any]) -> None:
        with self._lock:
            data = self._load()
            data.setdefault(server, {}).setdefault(username.lower(), {})[name.lower()] = spot
            self._save(data)

    def get(self, server: str, username: str, name: str) -> dict[str, Any] | None:
        return self.book(server, username).get(name.lower())


def places_for(bot: MinecraftBot) -> Places:
    return Places(Path(bot.store.path).parent / "pymc_bot_places.json")


def _server_key(bot: MinecraftBot) -> str:
    minecraft = bot.store.settings.minecraft
    return f"{minecraft.host}:{minecraft.port}".lower()


# ---------------------------------------------------------------------------- running a skill
def run_skill(bot: MinecraftBot, skill: Skill, params: dict[str, Any] | None = None) -> tuple[bool, str]:
    """Do one skill now. Returns ``(ok, what happened)`` in the words the AI and the panel show."""
    params = clean_params(skill, params)
    python = _PYTHON_SKILLS.get(skill.name)
    if python is not None:
        try:
            return python(bot, params)
        except Exception as exc:  # a skill never takes the bot down
            return False, f"{skill.name} failed: {exc}"
    if skill.bridge is None:  # pragma: no cover - every skill has one or the other
        return False, f"{skill.name} is not available"
    if skill.moves:
        with bot._motion_lock:  # noqa: SLF001 - skills are part of the bot
            ok, result = bot._perform(skill.bridge, params, timeout=skill.timeout, activity=skill.name, quiet=True)  # noqa: SLF001
    else:
        ok, result = bot._perform(skill.bridge, params, timeout=skill.timeout, activity=skill.name, quiet=True)  # noqa: SLF001
    if ok:
        message = str(result.get("message") or f"did {skill.name.replace('_', ' ')}")
        bot.log.add(message[:1].upper() + message[1:] + ("" if message.endswith((".", "!")) else "."), "info", "bot")
        bot.note_skill(skill.name, result)
        return True, message
    reason = bot.last_failure or f"could not {skill.name.replace('_', ' ')}"
    bot.log.add(f"Could not {skill.name.replace('_', ' ')}: {reason}", "info", "bot")
    return False, reason


def _here(bot: MinecraftBot) -> dict[str, Any] | None:
    snap = bot.snapshot()
    pos = snap.get("position")
    if not pos:
        return None
    return {"x": round(float(pos["x"]), 1), "y": round(float(pos["y"]), 1), "z": round(float(pos["z"]), 1),
            "dimension": snap.get("dimension") or "overworld"}


def _set_place(bot: MinecraftBot, name: str) -> tuple[bool, str]:
    spot = _here(bot)
    if spot is None:
        return False, "I don't know where I am yet"
    places_for(bot).set(_server_key(bot), bot.username, name, spot)
    bot.log.add(f"Remembered '{name}' at {spot['x']:.0f} {spot['y']:.0f} {spot['z']:.0f}.", "info", "bot")
    return True, f"remembered {name} at {spot['x']:.0f} {spot['y']:.0f} {spot['z']:.0f}"


def _goto_place(bot: MinecraftBot, name: str) -> tuple[bool, str]:
    spot = places_for(bot).get(_server_key(bot), bot.username, name)
    if spot is None:
        return False, f"I have no {name} yet" + (" - use set_home first" if name == "home" else "")
    ok = bot.walk_to(float(spot["x"]), float(spot["z"]), float(spot["y"]), tolerance=2.0, timeout=90)
    return ok, f"walked to {name}" if ok else (bot.last_failure or f"could not get to {name}")


def _skill_set_home(bot: MinecraftBot, params: dict[str, Any]) -> tuple[bool, str]:
    return _set_place(bot, "home")


def _skill_go_home(bot: MinecraftBot, params: dict[str, Any]) -> tuple[bool, str]:
    return _goto_place(bot, "home")


def _skill_set_waypoint(bot: MinecraftBot, params: dict[str, Any]) -> tuple[bool, str]:
    return _set_place(bot, str(params.get("name") or "spot"))


def _skill_goto_waypoint(bot: MinecraftBot, params: dict[str, Any]) -> tuple[bool, str]:
    name = str(params.get("name") or "")
    if not name:
        return False, "which waypoint? " + (", ".join(places_for(bot).book(_server_key(bot), bot.username)) or "none saved")
    return _goto_place(bot, name)


def _skill_waypoints(bot: MinecraftBot, params: dict[str, Any]) -> tuple[bool, str]:
    book = places_for(bot).book(_server_key(bot), bot.username)
    if not book:
        return True, "no saved places yet (set_home / set_waypoint)"
    return True, "places: " + ", ".join(f"{name} ({spot['x']:.0f} {spot['y']:.0f} {spot['z']:.0f})" for name, spot in sorted(book.items()))


def _skill_come(bot: MinecraftBot, params: dict[str, Any]) -> tuple[bool, str]:
    player = str(params.get("player") or "")
    if not player:
        near = bot.nearby_player(64)
        if not near:
            return False, "nobody is close enough to walk to"
        player = near["name"]
    target = bot.find_player(player)
    if target is None:
        return False, f"I can't see {player}"
    ok = bot.walk_to(float(target["x"]), float(target["z"]), tolerance=2.5, timeout=45)
    if ok:
        bot.look_at_player(player)
    return ok, f"came to {player}" if ok else (bot.last_failure or f"could not reach {player}")


def _skill_equip_weapon(bot: MinecraftBot, params: dict[str, Any]) -> tuple[bool, str]:
    inv = bot.inventory()
    index = inv.best_weapon()
    if index is None:
        return False, "I have no weapon"
    name = inv.slots[index].name
    return (True, f"holding {name}") if bot._hold_index(index) else (False, bot.last_failure or "could not hold it")  # noqa: SLF001


def _skill_equip_tool(bot: MinecraftBot, params: dict[str, Any]) -> tuple[bool, str]:
    block = str(params.get("block") or "stone")
    inv = bot.inventory()
    index = inv.best_tool(block)
    if index is None:
        return False, f"I have no tool for {block}"
    name = inv.slots[index].name
    return (True, f"holding {name} for {block}") if bot._hold_index(index) else (False, bot.last_failure or "could not hold it")  # noqa: SLF001


def _skill_totem(bot: MinecraftBot, params: dict[str, Any]) -> tuple[bool, str]:
    inv = bot.inventory()
    if inv.offhand and inv.offhand.name == "totem_of_undying":
        return True, "the totem is already in my off-hand"
    index = inv.find("totem_of_undying")
    if index is None:
        return False, "I have no totem of undying"
    ok, _ = bot._perform("equip", {"slot": index, "destination": "offhand"}, activity="totem")  # noqa: SLF001
    return ok, "totem in the off-hand" if ok else (bot.last_failure or "could not move the totem")


def _skill_shield(bot: MinecraftBot, params: dict[str, Any]) -> tuple[bool, str]:
    inv = bot.inventory()
    if not (inv.offhand and inv.offhand.name == "shield"):
        index = inv.find("shield")
        if index is None:
            return False, "I have no shield"
        ok, _ = bot._perform("equip", {"slot": index, "destination": "offhand"}, activity="shield")  # noqa: SLF001
        if not ok:
            return False, bot.last_failure or "could not take the shield"
    ok, _ = bot._perform("use_item", {"seconds": float(params.get("seconds") or 2.0), "offhand": True},  # noqa: SLF001
                         timeout=8.0, activity="shield")
    return ok, "blocked with the shield" if ok else (bot.last_failure or "could not raise the shield")


def junk_slots(inv: Inventory) -> list[int]:
    """Slots a player would empty first (one stack of plain building blocks is kept)."""
    kept: set[str] = set()
    out = []
    for index, stack in sorted(inv.items(), key=lambda item: -item[1].count):
        if stack.name in KEEP_ONE_STACK and stack.name not in kept:
            kept.add(stack.name)
            continue
        if stack.name in JUNK or stack.name in KEEP_ONE_STACK:
            out.append(index)
    return sorted(out)


def _skill_toss_junk(bot: MinecraftBot, params: dict[str, Any]) -> tuple[bool, str]:
    tossed: list[str] = []
    for _ in range(36):
        inv = bot.inventory()
        slots = junk_slots(inv)
        if not slots:
            break
        index = slots[0]
        name, count = inv.slots[index].name, inv.slots[index].count
        ok, _ = bot._perform("toss", {"slot": index}, activity="toss_junk", quiet=True)  # noqa: SLF001
        if not ok:
            break
        tossed.append(f"{count} {name}")
        bot._human_pause(0.15, 0.35)  # noqa: SLF001
    if not tossed:
        return False, "no junk to throw away"
    return True, "threw away " + ", ".join(tossed)


def _skill_guard(bot: MinecraftBot, params: dict[str, Any]) -> tuple[bool, str]:
    seconds = float(params.get("seconds") or 60)
    radius = float(params.get("radius") or 10)
    player = str(params.get("player") or "")
    anchor = _here(bot)
    if anchor is None:
        return False, "not in a world yet"
    deadline = time.monotonic() + seconds
    fights = 0
    bot.log.add(f"Guarding {'player ' + player if player else 'this spot'} for {seconds:.0f}s.", "info", "bot")
    with bot.activity("guard"):
        while time.monotonic() < deadline and not bot.cancelled and bot.connected:
            snap = bot.snapshot()
            threats = [e for e in snap.get("entities") or [] if e.get("hostile") and float(e.get("dist") or 99) <= radius]
            if threats:
                if bot.fight(short(min(threats, key=lambda e: e.get("dist", 99)).get("type"))):
                    fights += 1
                continue
            if player:
                target = bot.find_player(player)
                if target and float(target.get("distance") or 0) > 4:
                    bot.walk_to(float(target["x"]), float(target["z"]), tolerance=3.0, timeout=10)
            else:
                pos = snap.get("position") or {}
                if pos and math.dist((pos.get("x", 0), pos.get("z", 0)), (anchor["x"], anchor["z"])) > 4:
                    bot.walk_to(anchor["x"], anchor["z"], tolerance=2.0, timeout=10)
            time.sleep(0.6)
    return True, f"guarded for {seconds:.0f}s and fought {fights} mob(s)"


def _skill_hunt(bot: MinecraftBot, params: dict[str, Any]) -> tuple[bool, str]:
    animal = str(params.get("animal") or "")
    choices = [animal] if animal else ["cow", "pig", "sheep", "chicken", "rabbit"]
    snap = bot.snapshot()
    seen = {short(e.get("type")) for e in snap.get("entities") or [] if float(e.get("dist") or 99) <= 24}
    target = next((name for name in choices if name in seen), choices[0] if animal else None)
    if target is None:
        return False, "no animals around to hunt"
    if not bot.fight(target):
        return False, bot.last_failure or f"no {target} nearby"
    bot.collect_items(8.0)
    return True, f"hunted a {target}"


def _skill_report(bot: MinecraftBot, params: dict[str, Any]) -> tuple[bool, str]:
    snap = bot.snapshot()
    pos = snap.get("position") or {}
    inv = Inventory.from_snapshot(snap)
    counts: dict[str, int] = {}
    for _, stack in inv.items():
        counts[stack.name] = counts.get(stack.name, 0) + stack.count
    top = ", ".join(f"{n} x{c}" for n, c in sorted(counts.items(), key=lambda kv: -kv[1])[:4]) or "nothing"
    free = sum(1 for stack in inv.slots if stack is None)
    line = (f"hp {float(snap.get('health') or 0):.0f}/20, food {float(snap.get('food') or 0):.0f}/20, "
            f"at {float(pos.get('x', 0)):.0f} {float(pos.get('y', 0)):.0f} {float(pos.get('z', 0)):.0f}, "
            f"{free} free slots, carrying {top}")
    sent = bot.say(line)
    return bool(sent), line


_PYTHON_SKILLS = {
    "set_home": _skill_set_home,
    "go_home": _skill_go_home,
    "set_waypoint": _skill_set_waypoint,
    "goto_waypoint": _skill_goto_waypoint,
    "waypoints": _skill_waypoints,
    "come": _skill_come,
    "equip_weapon": _skill_equip_weapon,
    "equip_tool": _skill_equip_tool,
    "totem": _skill_totem,
    "shield": _skill_shield,
    "toss_junk": _skill_toss_junk,
    "guard": _skill_guard,
    "hunt": _skill_hunt,
    "report": _skill_report,
}


def has_food(inv: Inventory) -> bool:
    return any(stack.name in FOODS for _, stack in inv.items())


def is_tool(name: str) -> bool:
    return tool_kind(name) is not None or armor_slot_for(name) is not None
