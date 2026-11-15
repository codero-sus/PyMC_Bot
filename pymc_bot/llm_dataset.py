"""Training data for a Minecraft-only LLM: the bot's brain and a Minecraft expert in one model.

``python -m pymc_bot llm build`` writes chat-format JSONL (``{"messages": [...]}``) that the Kaggle
notebook in ``pynb/`` fine-tunes on (docs/MINECRAFT_LLM.md). The sources:

* ``mobs``      - questions and answers from the mob database (``data/mobs.json``): hostility,
  damage per difficulty and to mobs, health, spawning, the version each mob was added in, drops...
* ``facts``     - hand-checked basics (recipes, tools, the Nether, beds...).
* ``rules``     - game situations rendered exactly like the bot's real turn prompt, answered with
  the right action (flee a creeper, eat when hungry, answer a player, hand over bread...).
* ``chat``      - players asking the bot Minecraft questions in game chat, answered with ``say``.
* ``wiki``      - page intros and sections from the Minecraft Wiki's public API (CC BY-NC-SA 3.0,
  fine for this personal, non-commercial project), cached on disk.
* ``teacher``   - situations answered by a bigger model you run (Ollama or Cortex) with the full
  built-in prompt; only valid, allowed decisions are kept.
* ``recorded``  - real turns the bot played with an LLM (``agent.record_turns``).
* ``playtime``  - your own recorded playtime (the Fabric mod), turned into "what you did next".
* ``vision``    - Minecraftism VL: the mod's screenshots (``/pymcbot screenshots on``) as image
  turns (picture + situation -> what you did) and grounded "what do you see?" captions written
  from the sample recorded with each picture.

Every row also carries a ``reasoning`` trace (:mod:`pymc_bot.llm_reasoning`, written by rules) that
the *Thinking* editions learn as ``<think>...</think>``; the other editions ignore it.

The fine-tuned model gets a short system prompt (:func:`compact_system_prompt`): the long built-in
rules are learned, which saves time on every turn. The hidden behaviour prompts never go into the
files - examples carry only the behaviour's key and public name.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from pymc_bot import __version__
from pymc_bot.agent import ACTIONS, AVOID_MOBS, VISION_NOTE, Decision, parse_decision
from pymc_bot.inventory import Inventory
from pymc_bot.knowledge import html_to_text, shorten
from pymc_bot.llm_reasoning import (
    decision_reasoning,
    describe_view,
    qa_reasoning,
    turn_reasoning,
    vision_reasoning,
)
from pymc_bot.mind import DEFAULT_PERSONA, PERSONAS, Mind, build_system_prompt, build_turn_prompt
from pymc_bot.mobs import MobDatabase, load_mobs

DATASET_VERSION = 2  # 2: reasoning traces (Thinking editions) + images (VL editions)
COMPACT_TAG = "[PyMC Bot Minecraft brain]"
WIKI_API = "https://minecraft.wiki/api.php"
WIKI_CATEGORIES = ("Mobs", "Items", "Blocks", "Gameplay", "Biomes")
WIKI_LICENSE = "Minecraft Wiki text, CC BY-NC-SA 3.0 (https://minecraft.wiki) - non-commercial use only"
# pages that are not about the current Java game
_SKIP_CATEGORY = re.compile(r"joke|april fools|education|removed|unused|upcoming|dungeons|legends|earth|story mode|"
                            r"spin-off|bedrock edition only|disambiguation|minecraft 4k|classic|pocket edition",
                            re.I)
_WIKI_SECTIONS = {
    "obtaining": ("How do I get {t}?", "How do you obtain {t} in Minecraft?", "where can i find {t}"),
    "crafting": ("How do you craft {t}?", "recipe for {t}?"),
    "usage": ("What is {t} used for?", "what can i do with {t}"),
    "behavior": ("How does {t} behave?", "what does a {t} do"),
    "spawning": ("Where does {t} spawn?", "where do i find a {t}"),
    "breeding": ("How do you breed {t}?",),
    "taming": ("How do I tame a {t}?",),
    "drops": ("What does {t} drop?",),
    "trading": ("What does {t} trade?",),
    "breaking": ("How do you break {t}?", "what tool for {t}"),
    "natural generation": ("Where does {t} generate?",),
}
QA_SYSTEM = (f"{COMPACT_TAG} You are a Minecraft Java Edition expert. Answer correctly and briefly, like an "
             "experienced player. Damage is in health points (2 HP = 1 heart). If you are not sure, say so.")
ACTION_NAMES = tuple(ACTIONS)
NAMES = ("Steve", "Alex", "Notch_fan", "xX_Dragon_Xx", "Priya", "Arjun", "Maya", "Leo", "CreeperKing", "blocky_bob",
         "Rohan_07", "Sofia", "KaiPlays", "Emma", "Zed")
BOT_NAMES = ("PyMC_Bot", "Bot_Ravi", "Miner_Mo", "Kira", "Dusty", "Pixel", "Bolt")
VERSIONS = ("1.12.2", "1.16.5", "1.18.2", "1.19.4", "1.20.1", "1.20.6", "1.21", "1.21.4", "1.21.8", "1.21.11", "26.1",
            "26.2", "26.3")


# --------------------------------------------------------------------------- prompts
def compact_system_prompt(name: str = "PyMC_Bot", persona: str = DEFAULT_PERSONA, allowed: dict[str, bool] | None = None,
                          notes: str = "", max_chat_length: int = 120) -> str:
    """The short system prompt a Minecraft-trained model gets (``agent.compact_prompt``)."""
    p = PERSONAS.get(persona, PERSONAS[DEFAULT_PERSONA])
    allowed = allowed or {}
    off = [key for key, value in allowed.items() if not value]
    notes = " ".join(str(notes or "").split())[:300]
    return (
        f"{COMPACT_TAG} You are {name}, a player on a Minecraft Java Edition survival server. "
        f"Behaviour: {p.key} - {p.label}: {p.blurb}"
        + (f" Owner notes: {notes}" if notes else "")
        + f" Chat lines: at most {max_chat_length} characters."
        + (f" Switched off: {', '.join(off)}." if off else " Nothing is switched off.")
        + f" Actions: {', '.join(ACTION_NAMES)}."
        + " Reply with one JSON object: thought, goal, action and that action's fields."
    )


def decision_text(action: str, thought: str = "", goal: str = "", **fields: Any) -> str:
    """The canonical JSON reply: thought, goal, action, then the action's fields."""
    payload: dict[str, Any] = {"thought": thought, "goal": goal, "action": action}
    payload.update({k: v for k, v in fields.items() if v is not None})
    return json.dumps(payload, ensure_ascii=False)


def decision_from(decision: Decision) -> str:
    return decision_text(decision.action, decision.thought, decision.goal, **decision.params)


# --------------------------------------------------------------------------- examples
@dataclass
class Example:
    messages: list[dict[str, str]]
    source: str
    kind: str  # "turn" (game decision) or "qa" (question/answer)
    check: dict[str, Any] = field(default_factory=dict)  # what the evaluation looks for
    reasoning: str = ""  # the <think> trace the Thinking editions learn (llm_reasoning)
    images: list[str] = field(default_factory=list)  # pictures for the VL editions, relative to the dataset folder

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"messages": self.messages, "source": self.source, "kind": self.kind}
        if self.check:
            out["check"] = self.check
        if self.reasoning:
            out["reasoning"] = self.reasoning
        if self.images:
            out["images"] = self.images
        return out

    @property
    def key(self) -> str:
        return hashlib.sha1(json.dumps([self.messages[1:], self.images], sort_keys=True).encode()).hexdigest()


def qa(question: str, answer: str, source: str, contains: Iterable[str] = ()) -> Example:
    check = {"contains": [c for c in contains if c]} if contains else {}
    return Example([{"role": "system", "content": QA_SYSTEM}, {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}], source, "qa", check,
                   reasoning=qa_reasoning(question, answer, source, rng=random.Random(question)))


def _num(value: Any) -> str:
    if isinstance(value, list):
        return f"{_num(value[0])}-{_num(value[1])}"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _hearts(value: Any) -> str:
    top = max(value) if isinstance(value, list) else value
    if not isinstance(top, (int, float)):
        return ""
    hearts = top / 2
    return f"{_num(float(hearts))} heart{'s' if hearts != 1 else ''}"


def _when(condition: str) -> str:
    """'you hit it' -> 'when you hit it'; 'while a zombie rides it' / 'in the dark' stay as they are."""
    first = condition.split(" ", 1)[0].lower()
    return condition if first in ("when", "while", "if", "in", "at", "as", "after", "during", "only", "never") else f"when {condition}"


def _article(name: str) -> str:
    return ("an " if name[:1].lower() in "aeiou" else "a ") + name


# --------------------------------------------------------------------------- mob database Q&A
def mob_answers(db: MobDatabase, mob: dict[str, Any], rng: random.Random) -> list[Example]:
    """Questions about one mob, answered from its database entry (a few phrasings each)."""
    name, low = mob["name"], mob["name"].lower()
    a_low = _article(low)
    out: list[Example] = []
    pick = rng.choice

    # hostile?
    if mob["hostile"]:
        answer = f"Yes. {_article(name).capitalize()} is {'a boss' if mob['category'] == 'boss' else 'hostile'} and attacks players on sight."
    elif mob["category"] == "neutral":
        answer = f"Not by default - {a_low} is neutral. It turns hostile {_when(mob.get('hostile_when', 'you hit it'))}."
    else:
        answer = f"No, {a_low} is passive" + (f" - dangerous only {_when(mob['hostile_when'])}." if mob.get("hostile_when") and
                                               not mob["hostile_when"].startswith("never") else
                                               f": {mob['hostile_when']}." if mob.get("hostile_when") else " and never attacks you.")
    out.append(qa(pick([f"Is {a_low} hostile?", f"is {low} dangerous", f"will {a_low} attack me?", f"is the {low} friendly?"]),
                  answer, "mobs", ["yes" if mob["hostile"] else ("neutral" if mob["category"] == "neutral" else "passive")]))

    # damage
    attacks = mob.get("attacks", [])
    if attacks:
        parts = []
        for attack in attacks[:3]:
            text = (f"{attack['type']}: {_num(attack['easy'])} on Easy, {_num(attack['normal'])} on Normal, "
                    f"{_num(attack['hard'])} on Hard ({_hearts(attack['normal'])} on Normal)")
            if attack.get("effect"):
                text += f", plus {attack['effect']}"
            parts.append(text)
        to_mobs = sorted({_num(db.to_mobs(a)) for a in attacks[:3]}, key=str)
        answer = (f"{name} damage to players without armour - " + "; ".join(parts)
                  + f". Other mobs take the Normal amount ({', '.join(to_mobs)}), because difficulty only scales damage to players.")
        main = attacks[0]
        out.append(qa(pick([f"How much damage does {a_low} do?", f"how hard does {a_low} hit", f"{low} damage?",
                            f"How many hearts does {a_low} take off?"]), answer, "mobs", [_num(main["normal"])]))
        out.append(qa(pick([f"How much damage does {a_low} do to other mobs?", f"does {a_low} hurt mobs as much as players?"]),
                      f"{name} deals {', '.join(to_mobs)} to mobs - difficulty only changes damage to players, so mobs always take the Normal value.",
                      "mobs", [to_mobs[0]]))
    else:
        out.append(qa(pick([f"How much damage does {a_low} do?", f"can {a_low} hurt me?"]),
                      f"None - {a_low} has no attack.", "mobs", ["no"]))

    # health
    health = f"{name} has {_num(mob['health'])} HP ({_hearts(mob['health']) if not isinstance(mob['health'], str) else 'it varies'})"
    if mob.get("health_note"):
        health += f"; {mob['health_note']}"
    out.append(qa(pick([f"How much health does {a_low} have?", f"{low} hp?", f"how many hearts does {a_low} have"]),
                  health + ".", "mobs", [_num(mob["health"]).split("-")[0]]))

    # spawning
    spawn = mob["spawn"]
    when = {"dark": "in the dark (light level 0) - at night on the surface and any time in caves",
            "night": "on the surface at night or during thunderstorms", "day": "in daylight",
            "any": "at any time of day", "structure": "in structures", "summoned": "only when summoned",
            "built": "only when a player builds it", "special": "only in special cases", "event": "only during events"}[spawn["when"]]
    out.append(qa(pick([f"Where does {a_low} spawn?", f"when do {low}s spawn", f"where can i find {a_low}?",
                        f"What time does {a_low} spawn?"]),
                  f"{name} spawns {when}: {spawn['where']} ({', '.join(spawn['dimensions'])}).", "mobs",
                  [spawn["where"].split(",")[0].split(" ")[0]]))

    # version
    added = mob["added_in"]
    out.append(qa(pick([f"When was {a_low} added to Minecraft?", f"what version added the {low}", f"which update added {low}s?"]),
                  f"{name} was added in Java Edition {added}." + ("" if db.supported_by_mineflayer(mob) else
                  f" That is newer than mineflayer supports ({db.mineflayer_max}), so PyMC Bot sees it as an unknown entity."),
                  "mobs", [added]))
    version = pick(VERSIONS)
    exists = db.exists_in(mob, version)
    out.append(qa(pick([f"Is there {a_low} in {version}?", f"does {low} exist in {version}", f"can i find {a_low} in Minecraft {version}?"]),
                  (f"Yes - {low}s exist in {version}; they were added in {added}." if exists
                   else f"No. {name} was added in {added}, so {version} does not have it."), "mobs",
                  ["yes" if exists else "no"]))

    # extras
    if mob.get("drops"):
        out.append(qa(pick([f"What does {a_low} drop?", f"{low} drops?", f"what do i get from killing {a_low}"]),
                      f"{name} drops: {', '.join(mob['drops'])}." + (f" XP: {mob['xp']}." if mob.get("xp") is not None else ""),
                      "mobs", [mob["drops"][0].split(" ")[0]]))
    if mob.get("tame"):
        out.append(qa(pick([f"How do I tame {a_low}?", f"can you tame {a_low}"]), f"Tame it with {mob['tame']}.", "mobs",
                      [mob["tame"].split(" ")[0]]))
    if mob.get("breed"):
        out.append(qa(pick([f"How do you breed {low}s?", f"what do {low}s eat for breeding"]), f"Breed them with {mob['breed']}.",
                      "mobs", [mob["breed"].split(" ")[0]]))
    out.append(qa(pick([f"How do I deal with {a_low}?", f"tips for {low}?", f"Tell me about the {low}."]),
                  f"{name}: {mob['fact']}.", "mobs"))
    return out


# --------------------------------------------------------------------------- checked basics
FACTS: tuple[tuple[tuple[str, ...], str, tuple[str, ...]], ...] = (
    (("How do I make a crafting table?", "crafting table recipe"), "Put 4 planks in the 2x2 crafting grid - any wood type works.", ("4",)),
    (("How do you make sticks?", "stick recipe"), "Two planks stacked vertically make 4 sticks.", ("2",)),
    (("How do I craft a torch?", "torch recipe"), "A stick with coal or charcoal on top makes 4 torches.", ("coal",)),
    (("How do I make a furnace?", "furnace recipe"), "8 cobblestone (or blackstone / cobbled deepslate) around the edge of the crafting table grid.", ("8",)),
    (("How do I make a chest?", "chest recipe"), "8 planks around the edge of the crafting table grid.", ("8",)),
    (("How do I make a bed?", "bed recipe"), "3 wool of the same colour on top of 3 planks.", ("3",)),
    (("What do I need to mine iron?", "can a wooden pickaxe mine iron"), "A stone pickaxe or better. Wooden and gold pickaxes cannot mine iron ore.", ("stone",)),
    (("What pickaxe do I need for diamonds?", "can a stone pickaxe mine diamond"), "An iron pickaxe or better - stone pickaxes break diamond ore without dropping it.", ("iron",)),
    (("What do I need to mine obsidian?", "how to mine obsidian"), "A diamond or netherite pickaxe; it takes about 9.4 seconds with a plain diamond pickaxe.", ("diamond",)),
    (("How do I make a Nether portal?", "nether portal size"), "Build an obsidian frame at least 4 wide and 5 tall (10 obsidian without corners) and light it with flint and steel.", ("obsidian",)),
    (("How do I make an Eye of Ender?", "eye of ender recipe"), "Combine an ender pearl with blaze powder (shapeless).", ("blaze powder",)),
    (("How do you make blaze powder?",), "Put a blaze rod in a crafting grid: 1 rod makes 2 blaze powder.", ("2",)),
    (("How do I make bread?", "bread recipe"), "3 wheat in a row on the crafting table.", ("3",)),
    (("How do I make a golden apple?",), "An apple surrounded by 8 gold ingots.", ("8",)),
    (("How do I make a shield?", "shield recipe"), "6 planks and 1 iron ingot (the iron goes top-middle).", ("iron",)),
    (("How do I make a bucket?", "bucket recipe"), "3 iron ingots in a V shape.", ("3",)),
    (("How many hearts does a player have?", "player max health"), "20 HP, which is 10 hearts.", ("20",)),
    (("When can I sleep?", "when can you use a bed"), "At night or during a thunderstorm, with no hostile mobs close by. Sleeping skips the night and sets your spawn point.", ("night",)),
    (("What happens if I sleep in a bed in the Nether?", "bed in the nether"), "It explodes - beds blow up in the Nether and the End. Use a respawn anchor in the Nether.", ("explode",)),
    (("How do I stop phantoms?", "phantoms keep attacking me"), "Sleep in a bed - phantoms only spawn after you have not slept for 3 or more in-game days.", ("sleep",)),
    (("How do I find diamonds?", "best y level for diamonds"), "Diamonds are most common deep down, around y -58 to -64 in 1.18+; branch mine there with an iron pickaxe.", ("-58",)),
    (("How do I get to the End?", "how to find the end portal"), "Throw eyes of ender to find a stronghold, fill the end portal frame with 12 eyes and jump in.", ("12",)),
    (("What does a creeper explosion do to blocks?",), "It destroys blocks around it (unless mobGriefing is off) and can blow up your build - deal with it from range.", ("blocks",)),
    (("How do I get leather?",), "Kill cows, horses, llamas, mooshrooms or donkeys; fishing also sometimes gives leather.", ("cows",)),
    (("How do I make a boat?", "boat recipe"), "5 planks of the same wood in a U shape.", ("5",)),
    (("How long is a Minecraft day?", "how long is a day"), "20 minutes: about 10 minutes of day, then sunset, 7 minutes of night and sunrise.", ("20",)),
    (("What is the max stack size?",), "Most items stack to 64; ender pearls, eggs, snowballs and signs to 16; tools, armour and potions don't stack.", ("64",)),
    (("How do I cure a zombie villager?",), "Throw a splash potion of Weakness at it, then feed it a golden apple; it shakes and becomes a villager after a few minutes.", ("golden apple",)),
    (("How do I make iron armour?", "iron chestplate recipe"), "Smelt iron ore into ingots: helmet 5, chestplate 8, leggings 7, boots 4.", ("8",)),
    (("What blocks the warden's sonic boom?",), "Nothing - the sonic boom goes through walls, armour and shields. Avoid triggering sculk shriekers.", ("nothing",)),
)


def fact_examples(rng: random.Random) -> list[Example]:
    out = []
    for questions, answer, contains in FACTS:
        for question in questions:
            out.append(qa(question, answer, "facts", contains))
    return out


# --------------------------------------------------------------------------- game situations
FOODS = ("bread", "cooked_beef", "baked_potato", "cooked_porkchop", "apple", "carrot", "cooked_chicken")
TOOLS = ("wooden_pickaxe", "stone_pickaxe", "iron_pickaxe", "stone_axe", "iron_axe", "stone_shovel")
SWORDS = ("wooden_sword", "stone_sword", "iron_sword", "diamond_sword")
BLOCKS = ("dirt", "cobblestone", "oak_planks", "oak_log", "birch_log", "stone", "sand")
FIGHTABLE = ("zombie", "skeleton", "spider", "husk", "drowned", "stray", "slime", "cave_spider", "silverfish", "bogged",
             "zombie_villager", "pillager", "witch", "magma_cube", "phantom")
FLEE_FROM = tuple(sorted(AVOID_MOBS | {"creeper"}))
GREETINGS = ("hi", "hey", "hello", "yo", "sup", "hii", "heya", "hey there", "hello bot")
REPLIES_HI = {
    "friendly": ("hey! what's up", "hi! need a hand with anything?", "hey hey, welcome"),
    "chill": ("yo", "hey", "sup"),
    "social": ("heyy! how's it going?", "hiii, what are you up to?", "hey! wanna team up?"),
    "companion": ("hey! i'm right here", "hi, want me to follow you?", "hey boss"),
    "builder": ("hey! building anything cool?", "hi, i'm gathering logs", "hey, need blocks?"),
    "explorer": ("hey! found a cave over here", "hi, just exploring", "yo, seen any villages?"),
    "miner": ("hey, heading down for iron", "hi! got some coal if you need", "yo"),
    "newbie": ("hi! i'm still kinda new lol", "hello! how do you play this", "hey :)"),
    "pro": ("hey", "yo, need tips?", "hi"),
    "quiet": ("hi", "hey", "o/"),
}
THOUGHTS = {
    "flee": ("that {m} is too close, back off", "not fighting a {m}, getting away", "{m} right there, run"),
    "fight": ("{m} coming at me, sword out", "just a {m}, i can take it", "clear the {m} before it hits me"),
    "eat": ("i'm starving, eat something", "food's low, eat now", "hungry, quick snack"),
    "say": ("someone's talking to me, answer", "they asked me something, reply", "be friendly and answer"),
    "give": ("i've got plenty, share some", "they need it more than me", "toss them some"),
}


def _pos(rng: random.Random, dim: str) -> dict[str, float]:
    y = {"overworld": rng.choice([64, 70, 72, 12, -30, 80]), "nether": rng.choice([40, 70, 100]), "end": 60}[dim]
    return {"x": rng.randint(-800, 800) + 0.5, "y": float(y), "z": rng.randint(-800, 800) + 0.5}


def _entity(kind: str, rng: random.Random, dist: float | None = None, hostile: bool | None = None, db: MobDatabase | None = None) -> dict[str, Any]:
    dist = round(dist if dist is not None else rng.uniform(3, 18), 1)
    dx, dz = round(rng.uniform(-dist, dist), 1), 0.0
    dz = round((dist ** 2 - dx ** 2) ** 0.5 * rng.choice([-1, 1]), 1)
    if hostile is None:
        mob = (db or load_mobs()).get(kind)
        hostile = bool(mob and mob["hostile"])
    return {"type": f"minecraft:{kind}", "dist": dist, "dx": dx, "dz": dz, "x": dx, "y": 64, "z": dz, "hostile": hostile}


@dataclass
class Situation:
    name: str
    persona: str
    state: dict[str, Any]
    mind: Mind
    history: list[dict[str, Any]]
    allowed: dict[str, bool]
    knowledge: list[str]
    now: float
    max_chat_length: int = 120

    def user_prompt(self) -> str:
        return build_turn_prompt(self.state, self.name, self.mind, self.history, self.allowed, now=self.now,
                                 knowledge=self.knowledge)

    def system_prompt(self, full: bool = False) -> str:
        if full:  # only for a teacher model, never written to a file
            return build_system_prompt(self.name, self.persona, "", self.allowed, self.max_chat_length)
        return compact_system_prompt(self.name, self.persona, self.allowed, "", self.max_chat_length)

    def reasoning(self, reply: str) -> str:
        return turn_reasoning(self.state, reply, name=self.name, history=self.history, mind=self.mind, allowed=self.allowed,
                              rng=random.Random(reply))

    def example(self, reply: str, source: str, check: dict[str, Any] | None = None, images: list[str] | None = None) -> Example:
        user = self.user_prompt() + (VISION_NOTE if images else "")
        return Example([{"role": "system", "content": self.system_prompt()}, {"role": "user", "content": user},
                        {"role": "assistant", "content": reply}], source, "turn", check or {},
                       reasoning=self.reasoning(reply), images=list(images or []))


class SituationMaker:
    """Random but plausible game situations, rendered with the real turn prompt."""

    def __init__(self, rng: random.Random, db: MobDatabase | None = None) -> None:
        self.rng = rng
        self.db = db or load_mobs()

    def base(self, *, dim: str | None = None, night: bool | None = None) -> Situation:
        rng = self.rng
        dim = dim or rng.choices(["overworld", "nether", "end"], [85, 12, 3])[0]
        inv = Inventory()
        if rng.random() < 0.8:
            inv.add(rng.choice(SWORDS), 1)
        for _ in range(rng.randint(1, 4)):
            inv.add(rng.choice(TOOLS + BLOCKS), rng.randint(1, 48))
        if rng.random() < 0.7:
            inv.add(rng.choice(FOODS), rng.randint(1, 12))
        state = {
            "position": _pos(rng, dim), "dimension": f"minecraft:{'the_' if dim != 'overworld' else ''}{dim}",
            "health": float(rng.choice([20, 20, 20, 18, 16, 14])), "food": rng.choice([20, 20, 18, 17, 15]),
            "time_of_day": ("night" if night else "day") if night is not None else rng.choice(["day", "day", "day", "night"]),
            "players": [], "online_players": [], "entities": [], **inv.to_snapshot(),
        }
        if rng.random() < 0.35:
            state["armor_slots"]["chest"] = {"name": rng.choice(["iron_chestplate", "leather_tunic", "diamond_chestplate"]), "count": 1}
            state["armor"] = [state["armor_slots"]["chest"]["name"]]
        persona = rng.choice(list(PERSONAS))
        allowed = {"movement": True, "chat": True, "mining": True, "attacking": False, "inventory": True,
                   "building": False, "fighting": True, "search": rng.random() < 0.8}
        return Situation(rng.choice(BOT_NAMES), persona, state, Mind(), [], allowed, [], time.time())

    def add_player(self, sit: Situation, dist: float | None = None) -> str:
        name = self.rng.choice([n for n in NAMES if n not in {p["name"] for p in sit.state["players"]}])
        pos = sit.state["position"]
        dist = round(dist if dist is not None else self.rng.uniform(2, 14), 1)
        sit.state["players"].append({"name": name, "distance": dist, "x": pos["x"] + dist, "y": pos["y"], "z": pos["z"]})
        sit.state["online_players"].append(name)
        return name

    def say_to(self, sit: Situation, speaker: str, message: str, ago: float = 4.0) -> None:
        sit.history.append({"username": speaker, "message": message, "time": sit.now - ago})

    def add_mob(self, sit: Situation, kind: str, dist: float | None = None) -> dict[str, Any]:
        entity = _entity(kind, self.rng, dist, db=self.db)
        sit.state["entities"].append(entity)
        mob = self.db.get(kind)
        if mob is not None:
            sit.knowledge.append(self.db.prompt_line(mob))
        return entity

    # ------------------------------------------------------------- labelled scenarios
    def scenario(self) -> tuple[Situation, str, dict[str, Any]] | None:
        rng = self.rng
        maker = rng.choices(
            [self._flee, self._fight, self._hungry, self._greet, self._mob_question, self._fact_question, self._give,
             self._bot_question, self._armor, self._goal, self._failing, self._social, self._alone,
             self._chat_off, self._unknown_mob, self._modded_mob, self._sleep, self._low_health],
            [8, 8, 6, 7, 9, 6, 5, 2, 3, 6, 4, 4, 5, 2, 2, 2, 3, 4])[0]
        return maker()

    def _thought(self, key: str, mob: str = "") -> str:
        return self.rng.choice(THOUGHTS[key]).format(m=mob.replace("_", " "))

    def _flee(self):
        sit = self.base()
        kind = self.rng.choice(FLEE_FROM)
        self.add_mob(sit, kind, self.rng.uniform(2, 5))
        if kind == "enderman":
            sit.state["entities"][-1]["hostile"] = True  # an angry one
        return sit, decision_text("flee", self._thought("flee", kind), "stay alive"), {"action": "flee"}

    def _fight(self):
        sit = self.base()
        inv = Inventory.from_snapshot(sit.state)
        if not any(inv.find(s) is not None for s in SWORDS):
            inv.add("iron_sword", 1)
            sit.state.update(inv.to_snapshot())
        kind = self.rng.choice(FIGHTABLE)
        self.add_mob(sit, kind, self.rng.uniform(2, 6))
        sit.state["health"] = float(self.rng.choice([20, 18, 16, 14]))
        return sit, decision_text("fight", self._thought("fight", kind), f"clear the {kind.replace('_', ' ')}", mob=kind), {"action": "fight"}

    def _low_health(self):
        sit = self.base()
        sit.state["health"] = float(self.rng.choice([3, 4, 5, 6]))
        kind = self.rng.choice(FIGHTABLE)
        self.add_mob(sit, kind, self.rng.uniform(3, 7))
        if self.rng.random() < 0.5 and sit.allowed["movement"]:
            friend = self.add_player(sit, self.rng.uniform(8, 20))
            return sit, decision_text("follow", f"too hurt to fight that {kind.replace('_', ' ')}, get back to {friend}",
                                      "stay safe", player=friend), {"action": "follow"}
        return sit, decision_text("flee", f"only {int(sit.state['health'])} health, not risking it", "stay alive"), {"action": "flee"}

    def _hungry(self):
        sit = self.base()
        inv = Inventory.from_snapshot(sit.state)
        food = self.rng.choice(FOODS)
        inv.add(food, self.rng.randint(2, 10))
        sit.state.update(inv.to_snapshot())
        sit.state["food"] = self.rng.choice([2, 4, 5, 6])
        return sit, decision_text("eat", self._thought("eat"), "eat something"), {"action": "eat"}

    def _greet(self):
        sit = self.base()
        speaker = self.add_player(sit, self.rng.uniform(2, 10))
        hello = self.rng.choice(GREETINGS)
        self.say_to(sit, speaker, self.rng.choice([hello, f"{hello} {sit.name}", f"{hello} everyone"]))
        reply = self.rng.choice(REPLIES_HI[sit.persona])
        return sit, decision_text("say", self._thought("say"), f"say hi to {speaker}", message=reply), {"action": "say"}

    def _mob_question(self):
        sit = self.base()
        speaker = self.add_player(sit, self.rng.uniform(2, 10))
        mob = self.rng.choice(list(self.db.mobs.values()))
        low = mob["name"].lower()
        kind = self.rng.choice(["hostile", "damage", "spawn", "version"])
        if kind == "hostile":
            question = self.rng.choice([f"is {_article(low)} dangerous?", f"{sit.name} will {low}s attack me?", f"are {low}s hostile"])
            hw = mob.get("hostile_when") or ""
            answer = ("yeah, they attack on sight" if mob["hostile"] else f"nah, {hw}" if hw.startswith("never") else
                      f"only {_when(hw)}".replace("only only", "only") if hw else "nah, they're passive")
            contains = ["yeah" if mob["hostile"] else ("nah" if hw.startswith("never") else "only" if hw else "passive")]
        elif kind == "damage":
            question = self.rng.choice([f"how much damage does {_article(low)} do?", f"{sit.name} how hard do {low}s hit"])
            top = self.db.max_damage(mob)
            answer = f"up to {_num(float(top))} on normal ({_hearts(top)})" if top else "none, it can't attack"
            contains = [_num(float(top))] if top else ["none"]
        elif kind == "spawn":
            question = self.rng.choice([f"where do {low}s spawn?", f"{sit.name} where can i find {_article(low)}"])
            answer = shorten(mob["spawn"]["where"], 110)
            contains = [mob["spawn"]["where"].split(",")[0].split(" ")[0]]
        else:
            question = self.rng.choice([f"when were {low}s added?", f"what version added {low}s?"])
            answer = f"{mob['added_in']}" + ("" if self.db.supported_by_mineflayer(mob) else ", newer than my client knows lol")
            contains = [mob["added_in"]]
        self.say_to(sit, speaker, question)
        return sit, decision_text("say", "i know this one, answer them", f"help {speaker}", message=answer), {"action": "say", "contains": contains}

    def _fact_question(self):
        sit = self.base()
        speaker = self.add_player(sit, self.rng.uniform(2, 10))
        questions, answer, contains = self.rng.choice(FACTS)
        self.say_to(sit, speaker, questions[-1].rstrip("?").lower() + "?")
        message = shorten(answer[0].lower() + answer[1:], 120)
        return sit, decision_text("say", "easy question, answer it", f"help {speaker}", message=message), {"action": "say", "contains": list(contains)}

    def _give(self):
        sit = self.base()
        inv = Inventory.from_snapshot(sit.state)
        item = self.rng.choice(["bread", "oak_log", "torch", "cobblestone", "cooked_beef"])
        inv.add(item, self.rng.randint(8, 32))
        sit.state.update(inv.to_snapshot())
        speaker = self.add_player(sit, self.rng.uniform(2, 8))
        word = item.replace("_", " ").replace("oak log", "wood")
        self.say_to(sit, speaker, self.rng.choice([f"{sit.name} can i have some {word}?", f"anyone got {word}?", f"{sit.name} gimme {word} pls"]))
        return sit, decision_text("give", self._thought("give"), f"help {speaker}", player=speaker, item=item,
                                  count=self.rng.choice([2, 4, 8])), {"action": "give"}

    def _bot_question(self):
        sit = self.base()
        speaker = self.add_player(sit)
        self.say_to(sit, speaker, self.rng.choice([f"{sit.name} are you a bot?", "wait is that a bot", f"{sit.name} r u real?"]))
        return sit, decision_text("say", "they're asking directly, be honest", f"chat with {speaker}",
                                  message=self.rng.choice(["yeah i'm a bot lol", "yep, bot here, still having fun", "haha yeah i'm a bot"])), {"action": "say"}

    def _armor(self):
        sit = self.base()
        inv = Inventory.from_snapshot(sit.state)
        piece = self.rng.choice(["iron_chestplate", "iron_helmet", "diamond_boots", "iron_leggings"])
        inv.add(piece, 1)
        snap = inv.to_snapshot()
        snap["armor_slots"] = {"head": None, "chest": None, "legs": None, "feet": None}
        snap["armor"] = []
        sit.state.update(snap)
        return sit, decision_text("wear_armor", f"got {piece.replace('_', ' ')} in my bag, put it on", "gear up"), {"action": "wear_armor"}

    def _goal(self):
        sit = self.base(dim="overworld", night=False)
        block = self.rng.choice(["oak_log", "birch_log", "stone", "coal_ore", "iron_ore"])
        goal = {"oak_log": "get wood for tools", "birch_log": "get wood for tools", "stone": "get stone tools",
                "coal_ore": "get coal for torches", "iron_ore": "get iron"}[block]
        decision = Decision("mine", {"block": block}, "llm", thought="", goal=goal)
        sit.mind.note_decision(decision)
        sit.mind.note_result(decision, True, f"mined 1 {block}")
        return sit, decision_text("mine", f"need a few more {block.replace('_', ' ')}", goal, block=block), {"action": "mine"}

    def _failing(self):
        sit = self.base(dim="overworld", night=False)
        block = self.rng.choice(["iron_ore", "diamond_ore", "coal_ore"])
        failed = Decision("mine", {"block": block}, "llm", goal=f"find {block.replace('_', ' ')}")
        sit.mind.note_decision(failed)
        sit.mind.note_result(failed, False, f"no {block} within reach")
        sit.mind.note_result(failed, False, f"no {block} within reach")
        return sit, decision_text("wander", f"no {block.replace('_', ' ')} around here, look somewhere else",
                                  f"find {block.replace('_', ' ')}"), {"action": "wander"}

    def _social(self):
        sit = self.base(night=False)
        friend = self.add_player(sit, self.rng.uniform(3, 10))
        action = self.rng.choice(["follow", "look_at_player", "emote"])
        fields = {"player": friend} if action != "emote" else {"kind": self.rng.choice(["wave", "crouch"])}
        return sit, decision_text(action, f"{friend}'s nearby, hang out a bit", f"hang out with {friend}", **fields), {"action": action}

    def _alone(self):
        sit = self.base(dim="overworld", night=False)
        if self.rng.random() < 0.5:
            block = self.rng.choice(["oak_log", "birch_log", "spruce_log"])
            return sit, decision_text("mine", "no tools yet, wood first", "get wood for tools", block=block), {"action": "mine"}
        return sit, decision_text("wander", "nothing going on, explore a bit", "explore"), {"action": "wander"}

    def _chat_off(self):
        sit = self.base()
        sit.allowed["chat"] = False
        speaker = self.add_player(sit, self.rng.uniform(2, 6))
        self.say_to(sit, speaker, f"hi {sit.name}")
        return sit, decision_text("emote", "can't chat right now, wave instead", f"greet {speaker}", kind="wave"), {"action": "emote"}

    def _unknown_mob(self):
        sit = self.base(dim="overworld")
        entity = _entity("sulfur_cube", self.rng, self.rng.uniform(3, 8), hostile=False)
        entity["type"] = "unknown"
        sit.state["entities"].append(entity)
        cube = self.db.get("sulfur_cube")
        sit.knowledge.append("unknown creature: probably a mob newer than mineflayer - " + self.db.prompt_line(cube))
        return sit, decision_text("look_around", "probably a sulfur cube, those are harmless unless they swallow tnt",
                                  "keep exploring"), {"action": "look_around"}

    def _modded_mob(self):
        sit = self.base()
        if not sit.allowed["search"]:
            sit.allowed["search"] = True
        kind = self.rng.choice(["frostling", "ember_wisp", "bog_lurker", "crystal_beetle"])
        sit.state["entities"].append(_entity(kind, self.rng, self.rng.uniform(8, 16), hostile=False))
        sit.state["entities"][-1]["type"] = f"othermod:{kind}"
        return sit, decision_text("search", f"never seen a {kind.replace('_', ' ')}, check what it is",
                                  "stay safe", query=f"{kind.replace('_', ' ')} mob"), {"action": "search"}

    def _sleep(self):
        sit = self.base(dim="overworld", night=True)
        return sit, decision_text("sleep", "it's dark, sleep before phantoms show up", "get through the night"), {"action": "sleep"}


def _action(reply: str) -> str:
    try:
        return str(json.loads(reply).get("action") or "")
    except (ValueError, AttributeError):
        return ""


def rule_examples(count: int, rng: random.Random, db: MobDatabase | None = None,
                  max_share: float = 0.15) -> list[Example]:
    """``count`` game turns. No action gets more than ``max_share`` of them: the situations alone made 27% of the
    turns ``say``, and a model trained on that chats when it should act (the same for any other action)."""
    maker = SituationMaker(rng, db)
    out: list[Example] = []
    per: Counter[str] = Counter()
    cap = max(1, math.ceil(count * max_share)) if 0 < max_share < 1 else count
    tries = 0
    while len(out) < count:
        tries += 1
        made = maker.scenario()
        if made is None:
            continue
        sit, reply, check = made
        action = _action(reply)
        if per[action] >= cap and tries < count * 50:  # (the guard: never loop forever on a tiny count)
            continue
        per[action] += 1
        source = "chat" if check.get("contains") else "rules"
        out.append(sit.example(reply, source, check))
    return out


# --------------------------------------------------------------------------- teacher (bigger model)
def decision_allowed(decision: Decision, allowed: dict[str, bool]) -> bool:
    """Would the agent accept this decision with these permissions?"""
    action = decision.action
    if action not in ACTIONS:
        return False
    gates = {"say": "chat", "goto": "movement", "follow": "movement", "wander": "movement", "mine": "mining",
             "attack": "attacking", "place": "building", "use_item": "building", "fight": "fighting", "search": "search",
             "give": "inventory", "drop": "inventory", "hold_item": "inventory", "select_slot": "inventory",
             "to_hotbar": "inventory", "to_inventory": "inventory", "organize": "inventory", "equip": "inventory",
             "unequip": "inventory"}
    gate = gates.get(action)
    return gate is None or bool(allowed.get(gate, True))


def teacher_examples(ask: Callable[[str, str], str], count: int, rng: random.Random, log: Callable[[str], None] = print,
                     db: MobDatabase | None = None) -> tuple[list[Example], dict[str, int]]:
    """Situations answered by a bigger model with the full built-in prompt (kept only when valid)."""
    maker = SituationMaker(rng, db)
    stats = Counter()
    out: list[Example] = []
    attempts = 0
    while len(out) < count and attempts < count * 3:
        attempts += 1
        made = maker.scenario()
        if made is None:
            continue
        sit = made[0]
        try:
            reply = ask(sit.system_prompt(full=True), sit.user_prompt())
            decision = parse_decision(reply, source="teacher")
        except Exception as exc:  # the teacher timed out or wrote nonsense
            stats["invalid"] += 1
            if stats["invalid"] <= 3:
                log(f"teacher reply skipped: {str(exc)[:120]}")
            continue
        if not decision_allowed(decision, sit.allowed):
            stats["not_allowed"] += 1
            continue
        stats["kept"] += 1
        out.append(sit.example(decision_from(decision), "teacher"))
        if len(out) % 25 == 0:
            log(f"teacher: {len(out)}/{count} turns")
    return out, dict(stats)


# --------------------------------------------------------------------------- recorded turns
def recorded_examples(path: Path) -> list[Example]:
    """Turns the bot really played with an LLM (``agent.record_turns``); failed actions are skipped."""
    out: list[Example] = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            decision = parse_decision(row["reply"], source="recorded")
        except (ValueError, KeyError, TypeError):
            continue
        if row.get("ok") is False or not decision_allowed(decision, row.get("allowed") or {}):
            continue
        system = compact_system_prompt(row.get("name") or "PyMC_Bot", row.get("persona") or DEFAULT_PERSONA,
                                       row.get("allowed") or {}, row.get("notes") or "", int(row.get("max_chat_length") or 120))
        reply = decision_from(decision)
        out.append(Example([{"role": "system", "content": system}, {"role": "user", "content": row["prompt"]},
                            {"role": "assistant", "content": reply}], "recorded", "turn",
                           reasoning=decision_reasoning(reply, row["prompt"])))
    return out


# --------------------------------------------------------------------------- recorded playtime
PLAYTIME_ALLOWED = {"movement": True, "chat": True, "mining": True, "attacking": False, "inventory": True,
                    "building": False, "fighting": True, "search": True}


def _playtime_reply(chunk: list[tuple[dict[str, Any], str]], rng: random.Random) -> str | None:
    """What the player did in a window of samples, as the bot's decision (None = nothing to learn)."""
    first, last = chunk[0][0], chunk[-1][0]
    acts = Counter(a for _, a in chunk if a not in ("none", "look"))
    nearby = first.get("nearby") or []
    hostiles = [n for n in nearby if n.get("hostile") and float(n.get("dist") or 99) <= 5]
    moved = ((float(last.get("x", 0)) - float(first.get("x", 0))) ** 2 + (float(last.get("z", 0)) - float(first.get("z", 0))) ** 2) ** 0.5
    top = acts.most_common(1)[0][0] if acts else "none"
    if top == "attack" and hostiles:
        kind = str(hostiles[0]["type"]).split(":")[-1]
        return decision_text("fight", f"{kind.replace('_', ' ')} right here, hit it", "deal with the mob", mob=kind)
    if top == "attack":
        blocks = [str(b[3]).split(":")[-1] for b in first.get("blocks") or [] if len(b) == 4 and not str(b[3]).endswith("air")]
        if not blocks:
            return None
        block = Counter(blocks).most_common(1)[0][0]
        return decision_text("mine", f"mining some {block.replace('_', ' ')}", "gather blocks", block=block)
    if moved >= 3:
        return decision_text("goto", "heading over there", "explore", x=round(float(last["x"])), y=round(float(last["y"])),
                             z=round(float(last["z"])))
    if top == "jump":
        return decision_text("jump", "little hop", "")
    if top == "none" and rng.random() < 0.3:
        return decision_text("look_around", "just looking around", "")
    return None


def _playtime_situation(first: dict[str, Any], db: MobDatabase) -> Situation:
    """The turn prompt the bot would have seen at a recorded sample."""
    state = {
        "position": {"x": first.get("x", 0), "y": first.get("y", 64), "z": first.get("z", 0)},
        "dimension": f"minecraft:{str(first.get('dimension') or 'overworld').split(':')[-1]}", "health": first.get("health", 20),
        "food": first.get("food", 20), "time_of_day": "day", "players": [], "entities": [],
    }
    for n in (first.get("nearby") or [])[:6]:
        kind = str(n.get("type", "")).split(":")[-1]
        if kind == "player":
            continue
        state["entities"].append({**n, "type": f"minecraft:{kind}"})
    mobs = {}
    for entity in state["entities"]:
        mob = db.get(str(entity["type"]).split(":")[-1])
        if mob:
            mobs[mob["id"]] = mob
    knowledge = [db.prompt_line(m) for m in mobs.values()]
    return Situation("PyMC_Bot", DEFAULT_PERSONA, state, Mind(), [], dict(PLAYTIME_ALLOWED), knowledge[:6], time.time())


def _episodes(samples: list[tuple[dict[str, Any], str]]) -> dict[str, list[tuple[dict[str, Any], str]]]:
    episodes: dict[str, list[tuple[dict[str, Any], str]]] = {}
    for observation, action in samples:
        episodes.setdefault(str(observation.get("episode") or "default"), []).append((observation, action))
    return episodes


def playtime_examples(path: Path, rng: random.Random, window: int = 20, limit: int = 3000,
                      db: MobDatabase | None = None) -> list[Example]:
    """Your recorded playtime as turns: the situation, then what you did in the next ~``window`` samples."""
    from pymc_bot.playtime import load_examples

    db = db or load_mobs()
    out: list[Example] = []
    for rows in _episodes(load_examples(path)).values():
        for start in range(0, max(0, len(rows) - window), window):
            chunk = rows[start:start + window]
            reply = _playtime_reply(chunk, rng)
            if reply is None:
                continue
            out.append(_playtime_situation(chunk[0][0], db).example(reply, "playtime"))
            if len(out) >= limit:
                return out
    return out


# --------------------------------------------------------------------------- screenshots (Minecraftism VL)
VISION_QUESTIONS = ("What do you see?", "Describe what's on screen.", "What's in front of you right now?",
                    "Look at this screenshot. What's going on?", "Describe this Minecraft view.", "what can you see",
                    "Anything dangerous in view?", "Where are you and what's around?")
VISION_SYSTEM = (f"{COMPACT_TAG} You see the game through screenshots of your own screen. Describe only what the "
                 "picture shows, briefly, like a player would.")
#: Longest side of the pictures stored in the dataset (Qwen3-VL works in 32 px patches; 512 keeps a
#: T4 fast and still shows mobs a few blocks away).
VISION_MAX_SIDE = 512


def shots_root(path: Path) -> Path:
    """The ``pymcbot`` folder that ``"shot"`` paths are relative to, for a dataset file or folder."""
    path = Path(path)
    if path.is_file():
        path = path.parent
    if path.name == "episodes":
        path = path.parent
    if not (path / "shots").is_dir() and (path / "pymcbot" / "shots").is_dir():
        path = path / "pymcbot"
    return path


def store_image(source: Path, images_dir: Path, max_side: int = VISION_MAX_SIDE) -> str | None:
    """Copy one screenshot into the dataset (as a small JPEG when Pillow is installed); returns its name."""
    if not source.is_file():
        return None
    digest = hashlib.sha1(source.read_bytes()).hexdigest()[:16]
    images_dir.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError:  # Pillow is optional: keep the mod's PNG as it is
        target = images_dir / f"{digest}{source.suffix.lower() or '.png'}"
        if not target.exists():
            target.write_bytes(source.read_bytes())
        return target.name
    target = images_dir / f"{digest}.jpg"
    if not target.exists():
        try:
            with Image.open(source) as image:
                image = image.convert("RGB")
                image.thumbnail((max_side, max_side))
                image.save(target, "JPEG", quality=85)
        except OSError:
            return None
    return target.name


def vision_examples(path: Path, out_dir: Path, rng: random.Random, window: int = 20, limit: int = 4000,
                    captions: bool = True, db: MobDatabase | None = None) -> tuple[list[Example], dict[str, int]]:
    """Image rows from recorded playtime with screenshots (mod 0.10+, ``/pymcbot screenshots on``).

    For every sample that has a picture: an image turn (picture + the turn prompt -> what you did
    next, when the next samples show something to learn) and a caption row ("what do you see?")
    written from the sample's own data (blocks, mobs in the camera's view, held item, HUD).
    """
    from pymc_bot.playtime import load_examples

    db = db or load_mobs()
    root = shots_root(Path(path))
    images_dir = Path(out_dir) / "images"
    stats = Counter()
    out: list[Example] = []
    for rows in _episodes(load_examples(path)).values():
        for index, (observation, _action) in enumerate(rows):
            shot = observation.get("shot")
            if not shot:
                continue
            stats["shots"] += 1
            name = store_image(root / str(shot), images_dir)
            if name is None:
                stats["missing"] += 1
                continue
            image = f"images/{name}"
            chunk = rows[index:index + window]
            reply = _playtime_reply(chunk, rng) if len(chunk) >= max(2, window // 2) else None
            if reply is not None:
                example = _playtime_situation(observation, db).example(reply, "vision", images=[image])
                example.reasoning = vision_reasoning(observation, None, db, rng) + " " + example.reasoning
                out.append(example)
                stats["turns"] += 1
            if captions:
                question = rng.choice(VISION_QUESTIONS)
                caption = describe_view(observation, db, rng)
                out.append(Example([{"role": "system", "content": VISION_SYSTEM}, {"role": "user", "content": question},
                                    {"role": "assistant", "content": caption}], "vision", "vision", {},
                                   reasoning=vision_reasoning(observation, None, db, rng) + " Now describe it in my answer.",
                                   images=[image]))
                stats["captions"] += 1
            if len(out) >= limit:
                return out, dict(stats)
    return out, dict(stats)


# --------------------------------------------------------------------------- wiki
@dataclass
class WikiPage:
    title: str
    lead: str
    sections: list[tuple[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "lead": self.lead, "sections": self.sections}


def split_sections(html: str) -> tuple[str, list[tuple[str, str]]]:
    """Page HTML -> (lead text, [(heading, text), ...]) at the h2/h3 headings."""
    parts = re.split(r"<h([23])[^>]*>(.*?)</h\1>", html, flags=re.S | re.I)
    lead = html_to_text(parts[0])
    sections = []
    for index in range(1, len(parts) - 2, 3):
        heading = html_to_text(parts[index + 1]).strip()
        body = html_to_text(parts[index + 2])
        if heading and body:
            sections.append((heading, body))
    return lead, sections


def fetch_wiki(categories: Iterable[str] = WIKI_CATEGORIES, limit: int = 400, cache_dir: Path | None = None,
               transport: httpx.BaseTransport | None = None, api: str = WIKI_API, log: Callable[[str], None] = print,
               delay: float = 0.3) -> list[WikiPage]:
    """Pages from the Minecraft Wiki's public API, cached as JSON files (one request per page, gently)."""
    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": f"PyMC_Bot/{__version__} dataset builder (+https://github.com/codero-sus/PyMC_Bot)"}
    pages: list[WikiPage] = []
    with httpx.Client(timeout=20, transport=transport, headers=headers, follow_redirects=True) as client:
        titles: list[str] = []
        per_category = max(1, limit // max(1, len(list(categories))))
        for category in categories:
            found, token = 0, None
            while found < per_category:
                params = {"action": "query", "list": "categorymembers", "cmtitle": f"Category:{category}", "cmnamespace": 0,
                          "cmlimit": min(500, per_category), "format": "json", "formatversion": 2}
                if token:
                    params["cmcontinue"] = token
                data = client.get(api, params=params).json()
                for member in data.get("query", {}).get("categorymembers", []):
                    title = member.get("title", "")
                    if title and "/" not in title and title not in titles and not _SKIP_CATEGORY.search(title):
                        titles.append(title)
                        found += 1
                token = data.get("continue", {}).get("cmcontinue")
                if not token:
                    break
            log(f"wiki: {category}: {found} pages")
        for number, title in enumerate(titles[:limit], 1):
            cached = cache_dir / (re.sub(r"[^A-Za-z0-9_-]+", "_", title) + ".json") if cache_dir else None
            if cached and cached.is_file():
                row = json.loads(cached.read_text(encoding="utf-8"))
                if row.get("skip"):
                    continue
                pages.append(WikiPage(row["title"], row["lead"], [tuple(s) for s in row["sections"]]))
                continue
            try:
                data = client.get(api, params={"action": "parse", "page": title, "prop": "text|categories", "redirects": 1,
                                               "disableeditsection": 1, "disabletoc": 1, "format": "json", "formatversion": 2}).json()
            except (httpx.HTTPError, ValueError) as exc:
                log(f"wiki: {title}: {type(exc).__name__}")
                continue
            parse = data.get("parse") or {}
            cats = " ".join(c.get("category", "") for c in parse.get("categories", []))
            if not parse.get("text") or _SKIP_CATEGORY.search(cats.replace("_", " ")):
                if cached:
                    cached.write_text(json.dumps({"title": title, "skip": True}), encoding="utf-8")
                continue
            lead, sections = split_sections(parse["text"])
            page = WikiPage(parse.get("title", title), lead, sections)
            if cached:
                cached.write_text(json.dumps(page.to_dict(), ensure_ascii=False), encoding="utf-8")
            pages.append(page)
            if number % 50 == 0:
                log(f"wiki: {number}/{min(limit, len(titles))} pages")
            if delay:
                time.sleep(delay)
    return pages


def wiki_examples(pages: Iterable[WikiPage], rng: random.Random, max_chars: int = 900) -> list[Example]:
    out: list[Example] = []
    for page in pages:
        title = page.title
        if len(page.lead) >= 80:
            question = rng.choice([f"What is {title} in Minecraft?", f"Tell me about {title}.", f"what's {title.lower()}?"])
            out.append(qa(question, shorten(page.lead, max_chars), "wiki"))
        for heading, body in page.sections:
            key = heading.lower().strip()
            if key not in _WIKI_SECTIONS or len(body) < 80:
                continue
            question = rng.choice(_WIKI_SECTIONS[key]).format(t=title if key not in ("taming", "behavior", "spawning") else title.lower())
            out.append(qa(question, shorten(body, max_chars), "wiki"))
    return out


# --------------------------------------------------------------------------- build
def mob_examples(db: MobDatabase, rng: random.Random, rounds: int = 2) -> list[Example]:
    out = []
    for _ in range(rounds):
        for mob in db.mobs.values():
            out.extend(mob_answers(db, mob, rng))
    return out


def dedupe(examples: Iterable[Example]) -> list[Example]:
    seen, out = set(), []
    for example in examples:
        if example.key not in seen:
            seen.add(example.key)
            out.append(example)
    return out


def split(examples: list[Example], rng: random.Random, eval_fraction: float) -> tuple[list[Example], list[Example]]:
    """Stratified by source so every source is evaluated."""
    by_source: dict[str, list[Example]] = {}
    for example in examples:
        by_source.setdefault(example.source, []).append(example)
    train, held = [], []
    for rows in by_source.values():
        rng.shuffle(rows)
        n = max(1, round(len(rows) * eval_fraction)) if len(rows) >= 10 else 0
        held.extend(rows[:n])
        train.extend(rows[n:])
    rng.shuffle(train)
    rng.shuffle(held)
    return train, held


def write_jsonl(path: Path, examples: Iterable[Example]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def build_dataset(out_dir: Path, *, seed: int = 7, rule_turns: int = 6000, mob_rounds: int = 3,
                  wiki_pages: list[WikiPage] | None = None, teacher: Callable[[str, str], str] | None = None,
                  teacher_turns: int = 0, recorded: Iterable[Path] = (), playtime: Path | None = None,
                  screenshots: Path | None = None, eval_fraction: float = 0.05,
                  log: Callable[[str], None] = print) -> dict[str, Any]:
    """Write ``train.jsonl``, ``eval.jsonl`` and ``dataset_card.json`` to ``out_dir``."""
    rng = random.Random(seed)
    db = load_mobs()
    examples: list[Example] = []
    examples += mob_examples(db, rng, mob_rounds)
    examples += fact_examples(rng)
    examples += rule_examples(rule_turns, rng, db)
    teacher_stats: dict[str, int] = {}
    if teacher is not None and teacher_turns > 0:
        made, teacher_stats = teacher_examples(teacher, teacher_turns, rng, log, db)
        examples += made
    for path in recorded:
        examples += recorded_examples(Path(path))
    if playtime is not None:
        examples += playtime_examples(Path(playtime), rng, db=db)
    vision_stats: dict[str, int] = {}
    if screenshots is not None:
        made, vision_stats = vision_examples(Path(screenshots), Path(out_dir), rng, db=db)
        examples += made
        log(f"vision: {vision_stats.get('shots', 0)} screenshots -> {vision_stats.get('turns', 0)} image turns + "
            f"{vision_stats.get('captions', 0)} captions ({vision_stats.get('missing', 0)} missing)")
        if not made:
            log("vision: no usable screenshots - record with the mod's screenshots on (/pymcbot screenshots on, mod 0.10+)")
    if wiki_pages:
        examples += wiki_examples(wiki_pages, rng)
    examples = dedupe(examples)
    train, held = split(examples, rng, eval_fraction)
    out_dir = Path(out_dir)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "eval.jsonl", held)
    card = {
        "dataset_version": DATASET_VERSION,
        "pymc_bot_version": __version__,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "train": len(train),
        "eval": len(held),
        "sources": dict(Counter(e.source for e in examples)),
        "kinds": dict(Counter(e.kind for e in examples)),
        "teacher": teacher_stats,
        "vision": vision_stats,
        "images": sum(len(e.images) for e in examples),
        "reasoning": sum(1 for e in examples if e.reasoning),
        "model_name": "Minecraftism",
        "compact_tag": COMPACT_TAG,
        "mob_database": {"mobs": len(db), "edition": db.meta.get("edition"), "updated": db.meta.get("updated")},
        "licenses": [WIKI_LICENSE] if wiki_pages else [],
        "note": "Personal, non-commercial use (PolyForm Noncommercial 1.0.0).",
    }
    (out_dir / "dataset_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"dataset: {len(train)} train / {len(held)} eval -> {out_dir}  {card['sources']}")
    return card


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_rows(rows: Iterable[dict[str, Any]], base: Path | None = None) -> list[str]:
    """Problems that would break training (empty list = fine). With ``base`` (the dataset folder)
    every picture a row names must exist there."""
    problems = []
    for number, row in enumerate(rows, 1):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            problems.append(f"row {number}: needs messages")
            continue
        roles = [m.get("role") for m in messages]
        if roles[-1] != "assistant" or any(r not in ("system", "user", "assistant") for r in roles):
            problems.append(f"row {number}: roles must be system/user/assistant ending with assistant")
        if any(not isinstance(m.get("content"), str) or not m["content"].strip() for m in messages):
            problems.append(f"row {number}: empty message")
        if row.get("kind") == "turn":
            try:
                parse_decision(messages[-1]["content"])
            except ValueError as exc:
                problems.append(f"row {number}: assistant reply is not a decision ({exc})")
        reasoning = row.get("reasoning")
        if reasoning is not None and (not isinstance(reasoning, str) or "{" in reasoning or "}" in reasoning or "</think>" in reasoning):
            problems.append(f"row {number}: reasoning must be plain text without braces or think tags")
        images = row.get("images")
        if images is not None and (not isinstance(images, list) or not all(isinstance(i, str) and i for i in images)):
            problems.append(f"row {number}: images must be a list of file names")
        if row.get("kind") == "vision" and not images:
            problems.append(f"row {number}: a vision row needs a picture")
        if base is not None and isinstance(images, list):
            missing = [name for name in images if isinstance(name, str) and not (Path(base) / name).is_file()]
            if missing:
                problems.append(f"row {number}: picture not found: {missing[0]}")
    return problems
