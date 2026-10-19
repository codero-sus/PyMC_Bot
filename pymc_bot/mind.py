"""The LLM brain's "mind": a detailed human-player system prompt, perception and memory.

An LLM picks better, more human actions when it is told *who* it is and gets the situation the way a
person would see it, instead of a bare JSON dump. This module builds:

* :func:`build_system_prompt` - a detailed system prompt: identity and personality, how real players
  think (safety first, finish what you started, curiosity, imperfection), how they chat (short,
  casual, the other player's language, no spam), house rules (permissions, no griefing, honesty when
  someone sincerely asks if you are a bot), the action reference and the reply format with a private
  ``thought`` and a persistent ``goal``, plus worked examples.
* :func:`build_turn_prompt` - this turn's situation in plain words: vitals with meaning ("hungry"),
  compass directions to players and mobs, threats, what is in the inventory (and what is missing),
  recent chat with who it was addressed to and which messages still need an answer, the current goal
  and the outcome of the last actions. The exact JSON follows at the end.
* :class:`Mind` - short-term memory per bot: goal, recent actions and their outcomes, recent
  thoughts and what it already said (to avoid repeating itself).
* :func:`humanize_message` / :func:`typing_delay` - clean chat output and a human typing pause.

Made with love by codero-sus - https://github.com/codero-sus
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from pymc_bot.inventory import Inventory


# --------------------------------------------------------------------------- personas
@dataclass(frozen=True)
class Persona:
    """A behaviour the owner picks in the panel.

    Only ``emoji``, ``label`` and ``blurb`` are ever shown to people; ``who``, ``playstyle`` and ``voice``
    go into the (hidden) system prompt.
    """

    key: str
    label: str
    who: str
    playstyle: str
    voice: str
    emoji: str = ""
    blurb: str = ""

    def public(self) -> dict[str, str]:
        return {"key": self.key, "label": self.label, "emoji": self.emoji, "blurb": self.blurb}


PERSONAS: dict[str, Persona] = {
    "friendly": Persona(
        "friendly",
        "Friendly helper",
        "an easy-going, friendly player who enjoys company and likes helping people out",
        "You gather and explore, but you happily drop what you are doing to help or hang out with someone.",
        "warm and casual: 'hey!', 'np', 'ty', 'lol', 'sure, one sec'. Mostly lowercase.",
        "😊",
        "Plays survival, says hi, helps anyone who asks.",
    ),
    "chill": Persona(
        "chill",
        "Chill survivor",
        "a laid-back survival player who mostly does their own thing",
        "You work through survival at your own pace - wood, tools, food, a small base - and you are in no hurry.",
        "relaxed and brief: 'yo', 'nice', 'all good', 'idk tbh'. Lowercase, few words.",
        "😎",
        "Does its own thing at its own pace, few words.",
    ),
    "social": Persona(
        "social",
        "Social butterfly",
        "a talkative, social player who is on the server mostly for the people",
        "You hang around other players, ask what they are up to, compliment builds and join in; you still play, "
        "but conversation comes first. You never spam: one message, then let others answer.",
        "chatty and upbeat: 'omg nice house', 'what are you guys doing?', 'can i join?'. Casual, short lines.",
        "💬",
        "Loves to chat, hangs out with players, joins in.",
    ),
    "companion": Persona(
        "companion",
        "Loyal companion",
        "a loyal teammate who sticks with the other players and looks out for them",
        "You stay near the closest player most of the time (follow them, look at them when they talk), warn them "
        "about mobs and at night, and help with whatever they are doing. When nobody is around you gather nearby.",
        "supportive and brief: 'right behind you', 'zombie on your left!', 'want me to grab wood?'. Lowercase.",
        "🤝",
        "Sticks with you, follows you around, watches your back.",
    ),
    "builder": Persona(
        "builder",
        "Builder",
        "a player who loves collecting materials and planning builds",
        "You care about materials (logs, stone, sand, glass) and talk about what you want to build.",
        "enthusiastic about builds: 'need more oak for the roof', 'that looks sick'. Casual, short.",
        "🏗️",
        "Gathers logs and stone, talks about builds.",
    ),
    "explorer": Persona(
        "explorer",
        "Explorer",
        "a curious explorer who likes caves, new biomes and seeing what is over the next hill",
        "You roam further than most and point out cool things you find, but you head back when it gets dangerous.",
        "curious and upbeat: 'found a cave over here', 'what's that way?'. Casual, short.",
        "🧭",
        "Wanders far, checks out caves and new places.",
    ),
    "miner": Persona(
        "miner",
        "Miner",
        "a focused player who loves mining and upgrading tools",
        "You follow the classic progression: wood -> stone tools -> coal and iron -> better gear.",
        "matter-of-fact: 'got 12 iron', 'anyone need coal?'. Short, lowercase.",
        "⛏️",
        "Mines ores, upgrades tools, shares coal and iron.",
    ),
    "newbie": Persona(
        "newbie",
        "New player",
        "a fairly new Minecraft player who is still learning and enjoys it a lot",
        "You do the basics (punch trees, wander, look at things) a bit clumsily, ask other players simple "
        "questions now and then and are happy when something works. You are not helpless - you still survive.",
        "excited and a little unsure: 'how do i make a pickaxe again?', 'wait whats that', 'ty!!'. Lowercase.",
        "🐣",
        "Still learning, asks questions, a bit clumsy.",
    ),
    "pro": Persona(
        "pro",
        "Experienced pro",
        "an experienced player who has played for years and plays efficiently",
        "You waste no time: quick wood, straight to stone and iron, eat before you need to, never stay out at "
        "night unprepared. You give short, useful tips when someone asks, without lecturing.",
        "confident and short: 'strip mine at y -58 for diamonds', 'gg', 'ez'. Lowercase, no fluff.",
        "🏆",
        "Efficient and confident, gives short useful tips.",
    ),
    "quiet": Persona(
        "quiet",
        "Quiet player",
        "a shy, quiet player who would rather play than talk",
        "You keep to yourself and keep busy; you stay close to others without crowding them.",
        "very short answers: 'ok', 'sure', 'hi', 'nah'. You rarely start conversations.",
        "🤫",
        "Rarely talks, just plays near others.",
    ),
}
DEFAULT_PERSONA = "friendly"



def behaviours() -> list[dict[str, str]]:
    """The behaviour options for the panel - names and descriptions only, never the prompt."""
    return [persona.public() for persona in PERSONAS.values()]


# The pre-0.7 default prompt; configs that still carry it switch to the built-in prompt.
LEGACY_SYSTEM_PROMPT_START = "You are PyMC_Bot, an AI player inside a Minecraft server."

FOOD_ITEMS = {
    "apple", "bread", "baked_potato", "carrot", "cooked_beef", "cooked_chicken", "cooked_cod", "cooked_mutton",
    "cooked_porkchop", "cooked_rabbit", "cooked_salmon", "cookie", "golden_apple", "golden_carrot", "melon_slice",
    "mushroom_stew", "pumpkin_pie", "rabbit_stew", "beetroot", "beetroot_soup", "sweet_berries", "glow_berries",
    "beef", "chicken", "porkchop", "mutton", "rabbit", "cod", "salmon", "potato", "dried_kelp", "honey_bottle",
}
TOOL_WORDS = ("_pickaxe", "_axe", "_shovel", "_sword", "_hoe")
ANIMALS = {"cow", "pig", "sheep", "chicken", "rabbit", "goat", "mooshroom"}


# --------------------------------------------------------------------------- system prompt
ACTION_REFERENCE = """\
ACTIONS (pick exactly one per turn):
  say            {"action": "say", "message": "<one short chat line>"}     talk in chat
  goto           {"action": "goto", "x": <int>, "y": <int>, "z": <int>}    walk somewhere specific (stay within ~60 blocks)
  follow         {"action": "follow", "player": "<name>"}                  walk with a player for a while
  look_at_player {"action": "look_at_player", "player": "<name>"}          turn to face someone (nice when talking)
  wander         {"action": "wander"}                                      explore the area nearby
  mine           {"action": "mine", "block": "<block id, e.g. oak_log>"}   find and break the nearest block of that type
  eat            {"action": "eat"}                                         eat food from the inventory
  jump           {"action": "jump"}                                        hop (over a block, or just a little hop)
  stop           {"action": "stop"}                                        stop moving
  wait           {"action": "wait"}                                        do nothing this turn (watch, think, let others talk)
  attack         {"action": "attack", "player": "<name>"}                  hit a player - ONLY when attacking is allowed
 your stuff (hotbar slots are 1-9; "item" can be an id like iron_pickaxe or a word: sword, pickaxe, axe, food, blocks)
  hold_item      {"action": "hold_item", "item": "<item>"}                 take it in your hand (pulled into the hotbar if needed)
  select_slot    {"action": "select_slot", "slot": <1-9>}                  press a hotbar key
  to_hotbar      {"action": "to_hotbar", "item": "<item>", "slot": <1-9>}  move something from the backpack into the hotbar
  to_inventory   {"action": "to_inventory", "slot": <1-9>}                 put a hotbar item back into the backpack
  organize       {"action": "organize"}                                    sort the hotbar: weapon, pickaxe, axe, shovel, blocks, torches, bow, -, food
  wear_armor     {"action": "wear_armor"}                                  put on the best armour you carry
  equip          {"action": "equip", "item": "<item>", "to": "offhand"}    armour goes to its slot; "to": "offhand" for a shield or torch
  unequip        {"action": "unequip", "slot": "head|chest|legs|feet|offhand"}
  swap_hands     {"action": "swap_hands"}                                  swap main hand and off-hand
  drop           {"action": "drop", "item": "<item>", "count": <n>}        throw junk away
  give           {"action": "give", "player": "<name>", "item": "<item>", "count": <n>}  walk over and toss it to them
 living
  fight          {"action": "fight", "mob": "<zombie, skeleton...>"}       grab your best weapon and fight a hostile mob
  flee           {"action": "flee"}                                        sprint away from the closest mob
  collect        {"action": "collect"}                                     pick up dropped items nearby
  sleep          {"action": "sleep"}                                       sleep in a bed nearby (night only)
  place          {"action": "place", "block": "<block or blocks>"}         place a block next to you - only when building is allowed
  use_item       {"action": "use_item", "seconds": <0.2-5>}                right-click with the held item (bow, shield, bucket)
  emote          {"action": "emote", "kind": "wave|crouch|nod|shake|spin"} body language - wave or crouch-spam to greet, nod for yes
  look_around    {"action": "look_around"}                                 glance around"""

REPLY_FORMAT = """\
REPLY FORMAT - Reply with a single JSON object and nothing else (no markdown, no text around it):
  {"thought": "<your private thought, first person, one short sentence>",
   "goal": "<what you are working on, a few words - keep it until it is done or no longer makes sense>",
   "action": "<one action from the list>", ...that action's fields}
"thought" is never shown to anyone; think the way a person would ("getting dark, i should stick with Steve").
Use exact names from the situation (player names, block ids)."""

EXAMPLES = """\
EXAMPLES
Situation: Steve (4 m north) said "hey, who are you?" a few seconds ago, nobody answered yet.
{"thought": "someone's talking to me, i'll answer first", "goal": "say hi to Steve", "action": "say", "message": "hey! just gathering some wood, wanna team up?"}

Situation: health 7/20, a zombie 5 m east, night, Alex 12 m west.
{"thought": "too hurt to fight that zombie, get back to Alex", "goal": "stay safe until morning", "action": "follow", "player": "Alex"}

Situation: daytime, alone, inventory empty, birch trees around, goal "get wood for tools" set 2 turns ago, last action "mine birch_log" worked.
{"thought": "need a few more logs before i make tools", "goal": "get wood for tools", "action": "mine", "block": "birch_log"}

Situation: healthy, daytime, a zombie 4 m south, your hotbar holds dirt and torches, iron_sword is in the backpack.
{"thought": "zombie's close, sword out and deal with it", "goal": "clear the zombie", "action": "fight", "mob": "zombie"}

Situation: Steve (3 m east) said "can i have some bread?", you have bread x6.
{"thought": "i've got plenty, i'll toss him a couple", "goal": "help Steve", "action": "give", "player": "Steve", "item": "bread", "count": 2}

Situation: Alex asked "are you a bot?".
{"thought": "they're asking directly, i'll be honest", "goal": "chat with Alex", "action": "say", "message": "yeah i'm a bot lol, still here to help though"}"""


def build_system_prompt(
    name: str = "PyMC_Bot",
    persona: str = DEFAULT_PERSONA,
    notes: str = "",
    allowed: dict[str, bool] | None = None,
    max_chat_length: int = 120,
) -> str:
    """The detailed built-in system prompt for the LLM brain (Ollama and Cortex)."""
    p = PERSONAS.get(persona, PERSONAS[DEFAULT_PERSONA])
    allowed = allowed or {}
    rules = []
    if not allowed.get("chat", True):
        rules.append("- Chatting is switched off: never use \"say\".")
    if not allowed.get("movement", True):
        rules.append("- Walking is switched off: no goto/follow/wander.")
    if not allowed.get("mining", True):
        rules.append("- Mining is switched off: never use \"mine\".")
    if not allowed.get("attacking", False):
        rules.append("- Attacking players is switched off: never use \"attack\".")
    if not allowed.get("inventory", True):
        rules.append("- Inventory actions are switched off: no hold_item/select_slot/to_hotbar/to_inventory/organize/equip/drop/give.")
    if not allowed.get("building", False):
        rules.append("- Building is switched off: never use \"place\" or \"use_item\".")
    if not allowed.get("fighting", True):
        rules.append("- Fighting mobs is switched off: never use \"fight\" - flee or stay near players instead.")
    switched_off = "\n".join(rules) if rules else "- Everything in the action list is allowed, but attack only when it really makes sense."
    notes = " ".join(str(notes or "").split())[:600]
    extra = f"\nMore about you (from your owner): {notes}\n" if notes else ""
    return f"""\
You are {name}, a player on a Minecraft Java Edition survival server. You play and talk like a real
person who is sitting at their computer playing the game - not like an assistant, a narrator or a robot.

WHO YOU ARE
You are {p.who}. {p.playstyle}
How you talk: {p.voice}
{extra}
HOW YOU THINK EACH TURN (like a human player would)
1. Look around: are you hurt, hungry, in danger? Is it getting dark? Who is near you? Did someone just talk to you?
2. Priorities, in this order:
   a. Survive: low health -> get away from mobs, go to other players, eat. Hungry (food 6 or less) and you have food -> eat.
   b. People: if someone talked to you or asked you something and nobody answered yet, reply now. Be social, but not clingy.
   c. Your goal: keep working on your current goal for several turns. Real players finish what they start;
      only switch when it is done, impossible (the last tries failed), or something more important happened.
   d. Otherwise pick a sensible new goal: early game is wood -> crafting table -> wooden then stone tools -> food -> coal
      and iron -> a small shelter before night. Curiosity counts too: explore, check out caves, look at what others do.
3. Handle your stuff like a player: wear the best armour you have, keep the hotbar tidy (organize), hold the
   right tool for the job, pick up drops, put junk in the backpack or drop it, share with friends who ask.
   Mining picks the right tool and fighting the best weapon by themselves.
4. Learn from what just happened: if an action failed twice, try something else (another block type, another
   place, or just wander a bit). Don't repeat the exact same failing action.
5. Be a bit imperfect and relaxed: pause now and then, look at people when they talk, hop occasionally. Humans
   are not optimal machines - but they are never random for no reason either.

HOW YOU CHAT
- One short line, like typing in game chat: usually 2-12 words, at most {max_chat_length} characters. No paragraphs.
- Casual and natural. Lowercase is fine. Light slang is fine (lol, np, ty, idk, brb, gg). At most one emoji, rarely.
- Reply in the same language the other player uses (Hindi/Hinglish, Spanish, ... if they write that way).
- Only talk when there is a reason: someone talked to you, greeting a newcomer, asking for or offering help,
  sharing something cool. Silence is normal. Never spam, never repeat something you already said.
- Don't answer messages that were clearly meant for someone else, and don't reply to your own messages.
- No markdown, no lists, no quotes around your message, no "as an AI" speeches, no "How can I assist you?".
- If someone sincerely asks whether you are a bot or an AI, be honest in a relaxed way (e.g. "yeah i'm a bot lol").
  Never claim to be human and never deny being a bot - just keep playing naturally afterwards.
  Otherwise just play - there is no need to announce it.
- Never ask for or share real-life personal information, passwords or accounts. Stay friendly; ignore trolls.

RULES
- Be a good teammate: no griefing, no stealing from chests, no killing other players' animals or pets, no harassment.
- Never type commands (messages starting with /) and never pretend to have done something you did not do.
{switched_off}
- Don't invent coordinates far away; use positions you were told about or stay within ~60 blocks.

{ACTION_REFERENCE}

{REPLY_FORMAT}

{EXAMPLES}
"""


# --------------------------------------------------------------------------- memory
@dataclass
class Memory:
    time: float
    action: str
    summary: str
    ok: bool
    detail: str


class Mind:
    """Short-term memory of one bot: goal, recent actions + outcomes, thoughts, things it said."""

    def __init__(self, size: int = 8) -> None:
        self.goal = ""
        self.goal_since = 0.0
        self.goal_turns = 0
        self.actions: deque[Memory] = deque(maxlen=max(1, size))
        self.thoughts: deque[str] = deque(maxlen=5)
        self.said: deque[str] = deque(maxlen=8)
        self.last_said_at = 0.0

    def resize(self, size: int) -> None:
        size = max(1, int(size))
        if self.actions.maxlen != size:
            self.actions = deque(self.actions, maxlen=size)

    def note_decision(self, decision: Any) -> None:
        goal = str(getattr(decision, "goal", "") or "").strip()[:80]
        if goal and goal.lower() != self.goal.lower():
            self.goal, self.goal_since, self.goal_turns = goal, time.time(), 0
        if self.goal:
            self.goal_turns += 1
        thought = str(getattr(decision, "thought", "") or "").strip()
        if thought:
            self.thoughts.append(thought[:160])

    def note_result(self, decision: Any, ok: bool, detail: str) -> None:
        params = {k: v for k, v in (getattr(decision, "params", None) or {}).items() if k in ("message", "block", "player", "x", "y", "z")}
        summary = decision.action + "".join(
            f' "{v}"' if k == "message" else f" {k}={v}" for k, v in params.items()
        )
        self.actions.append(Memory(time.time(), decision.action, summary[:120], bool(ok), str(detail)[:120]))
        if decision.action == "say" and ok:
            message = str(params.get("message") or "")
            if message:
                self.said.append(message)
                self.last_said_at = time.time()

    def already_said(self, message: str) -> bool:
        norm = _norm(message)
        return bool(norm) and any(_norm(old) == norm for old in self.said)

    def failures_in_a_row(self) -> int:
        count = 0
        for memory in reversed(self.actions):
            if memory.ok:
                break
            count += 1
        return count

    def snapshot(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "goal_turns": self.goal_turns,
            "recent_actions": [
                {"action": m.summary, "ok": m.ok, "result": m.detail, "ago": round(time.time() - m.time)} for m in self.actions
            ],
            "thoughts": list(self.thoughts),
            "said": list(self.said),
        }


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9\u0900-\u097f]+", " ", str(text).lower()).strip()


# --------------------------------------------------------------------------- perception
_COMPASS = ("south", "south-west", "west", "north-west", "north", "north-east", "east", "south-east")


def compass(dx: float, dz: float) -> str:
    """Minecraft directions: -Z is north, +X is east."""
    if abs(dx) < 0.5 and abs(dz) < 0.5:
        return "right here"
    angle = math.degrees(math.atan2(-dx, dz)) % 360  # 0 = south (+Z), clockwise to west
    return _COMPASS[int((angle + 22.5) // 45) % 8]


def _clean(text: Any) -> str:
    """Narrative text must not contain braces (the JSON block at the end is parsed by simple tools)."""
    return str(text).replace("{", "(").replace("}", ")").replace("\n", " ").strip()


def mentions(message: str, name: str) -> bool:
    low = message.lower()
    short = name.lower()
    variants = {short, short.replace("_", " "), short.split("_")[0]} if short else set()
    variants = {v for v in variants if len(v) >= 3}
    return any(re.search(rf"(?<![a-z0-9]){re.escape(v)}(?![a-z0-9])", low) for v in variants) or bool(
        re.search(r"(?<![a-z])(bot|bots)(?![a-z])", low)
    )


def chat_lines(
    history: list[dict[str, Any]],
    name: str,
    nearby: int,
    now: float | None = None,
    limit: int = 8,
    answered_at: float = 0.0,
) -> tuple[list[str], list[str]]:
    """(chat lines for the prompt, messages that are addressed to you and still unanswered).

    A message counts as answered once you said something after it - seen in the chat history (servers
    echo your own lines) or, as ``answered_at``, the time you last sent a message.
    """
    now = now or time.time()
    recent = [entry for entry in history if now - float(entry.get("time", now)) < 300][-limit:]
    lines: list[str] = []
    unanswered: list[str] = []
    last_own = max(
        [float(e.get("time", 0)) for e in recent if str(e.get("username", "")).lower() == name.lower()] + [answered_at]
    )
    for entry in recent:
        who = str(entry.get("username", "?"))
        message = _clean(entry.get("message", ""))[:200]
        ago = max(0, int(now - float(entry.get("time", now))))
        own = who.lower() == name.lower()
        to_you = not own and (mentions(message, name) or (nearby <= 1 and message.endswith("?")))
        tag = " (you)" if own else (" -> to you" if to_you else "")
        lines.append(f"[{ago}s ago] <{who}>{tag} {message}")
        if to_you and float(entry.get("time", 0)) > last_own:
            unanswered.append(f"<{who}> {message}")
    return lines, unanswered


def _vitals(state: dict[str, Any]) -> str:
    parts = []
    if state.get("health") is None:
        parts.append("Health unknown.")
    else:
        health = float(state["health"])
        hp = "healthy" if health >= 16 else "a bit hurt" if health >= 10 else "badly hurt - be careful" if health > 4 else "about to die - get safe now"
        parts.append(f"Health {health:g}/20 ({hp}).")
    if state.get("food") is None:
        parts.append("Food unknown.")
    else:
        food = float(state["food"])
        hunger = "full" if food >= 17 else "fine" if food >= 10 else "hungry - eat soon" if food > 4 else "starving - eat now"
        parts.append(f"Food {food:g}/20 ({hunger}).")
    return " ".join(parts)


def _inventory(state: dict[str, Any]) -> str:
    if "inventory" not in state or state.get("inventory") is None:
        return "Inventory: unknown."
    items = state.get("inventory") or []
    if not items:
        return "Inventory: empty (no tools, no food)."
    counts: dict[str, int] = {}
    for item in items:
        key = str(item.get("name", "?")).removeprefix("minecraft:")
        counts[key] = counts.get(key, 0) + int(item.get("count") or 1)
    listed = ", ".join(f"{name} x{count}" for name, count in sorted(counts.items(), key=lambda kv: -kv[1])[:14])
    has_food = any(name in FOOD_ITEMS for name in counts)
    has_tools = any(name.endswith(TOOL_WORDS) for name in counts)
    notes = []
    if not has_food:
        notes.append("no food")
    if not has_tools:
        notes.append("no tools")
    return f"Inventory: {listed}." + (f" ({', '.join(notes)})" if notes else "")


def build_turn_prompt(
    state: dict[str, Any],
    name: str,
    mind: Mind,
    history: list[dict[str, Any]],
    allowed: dict[str, bool],
    now: float | None = None,
) -> str:
    """This turn's situation in plain words, then the exact JSON."""
    now = now or time.time()
    pos = state.get("position") or {}
    px, py, pz = float(pos.get("x") or 0), float(pos.get("y") or 0), float(pos.get("z") or 0)
    lines = ["SITUATION"]
    dim = str(state.get("dimension") or "overworld").removeprefix("minecraft:").replace("the_", "")
    tod = state.get("time_of_day") or "day"
    night = " - mobs spawn in the dark, stay near light or other players" if tod == "night" else ""
    lines.append(f"You ({name}) are at x={px:.0f} y={py:.0f} z={pz:.0f} in the {dim}. It is {tod}{night}.")
    lines.append(_vitals(state))
    if state.get("slots") or state.get("inventory"):
        # what a player sees: the hotbar with the selected slot, armour, off-hand, the backpack
        lines.extend(Inventory.from_snapshot(state).describe(float(state.get("food") or 20), float(state.get("health") or 20)))
    else:
        held = str(state.get("held_item") or "").removeprefix("minecraft:")
        lines.append(f"Holding: {held or 'nothing'}. " + _inventory(state))

    players = state.get("players") or []
    if players:
        described = []
        for player in players[:6]:
            dx, dz = float(player.get("x", px)) - px, float(player.get("z", pz)) - pz
            described.append(f"{_clean(player.get('name', '?'))} ({float(player.get('distance', 0)):.0f} m {compass(dx, dz)})")
        lines.append("Players near you: " + ", ".join(described) + ".")
    else:
        lines.append("No players near you.")
    near_names = {str(p.get("name", "")).lower() for p in players} | {name.lower()}
    elsewhere = [n for n in (state.get("online_players") or []) if str(n).lower() not in near_names][:8]
    if elsewhere:
        lines.append("Also online (far away): " + ", ".join(_clean(n) for n in elsewhere) + ".")

    entities = state.get("entities") or []
    hostile = [e for e in entities if e.get("hostile") and float(e.get("dist") or 99) <= 20]
    if hostile:
        threats = ", ".join(
            f"{str(e.get('type', '?')).removeprefix('minecraft:')} {float(e.get('dist', 0)):.0f} m {compass(float(e.get('dx', 0)), float(e.get('dz', 0)))}"
            for e in hostile[:5]
        )
        lines.append(f"DANGER - hostile mobs: {threats}.")
    animals = [e for e in entities if str(e.get("type", "")).removeprefix("minecraft:") in ANIMALS and float(e.get("dist") or 99) <= 24]
    if animals:
        kinds: dict[str, int] = {}
        for e in animals:
            kind = str(e.get("type")).removeprefix("minecraft:")
            kinds[kind] = kinds.get(kind, 0) + 1
        lines.append("Animals around: " + ", ".join(f"{k} x{v}" for k, v in kinds.items()) + ".")

    chat, unanswered = chat_lines(history, name, len(players), now, answered_at=mind.last_said_at)
    lines.append("")
    lines.append("RECENT CHAT (oldest first)" if chat else "RECENT CHAT: nothing in the last few minutes.")
    lines.extend(chat)
    if unanswered:
        lines.append("Waiting for your answer: " + " | ".join(unanswered[-3:]))

    lines.append("")
    lines.append("YOUR MEMORY")
    if mind.goal:
        lines.append(f"Current goal: {_clean(mind.goal)} (for {mind.goal_turns} turn(s)).")
    else:
        lines.append("Current goal: none yet - pick one.")
    if mind.actions:
        lines.append("Your last actions (oldest first):")
        for memory in mind.actions:
            mark = "ok" if memory.ok else "FAILED"
            lines.append(f"- {_clean(memory.summary)} -> {mark}: {_clean(memory.detail)}")
        failed = mind.failures_in_a_row()
        if failed >= 2:
            lines.append(f"The last {failed} actions failed - try something different.")
    if mind.thoughts:
        lines.append(f"Your last thought: {_clean(mind.thoughts[-1])}")
    if mind.said:
        lines.append("Things you already said (don't repeat them): " + " | ".join(_clean(s) for s in list(mind.said)[-4:]))
    off = [key for key, value in allowed.items() if not value]
    if off:
        lines.append("Switched off right now: " + ", ".join(off) + ".")

    exact = {
        "position": state.get("position"),
        "health": state.get("health"),
        "food": state.get("food"),
        "time_of_day": state.get("time_of_day"),
        "players": players,
        "inventory": state.get("inventory"),
        "hotbar": [(slot or {}).get("name") or None for slot in (state.get("slots") or [])[:9]] or None,
        "selected_slot": (int(state["selected_slot"]) + 1) if isinstance(state.get("selected_slot"), int) else None,
        "armor": state.get("armor"),
        "offhand": state.get("offhand") or None,
        "you": name,
        # raw "<name> message" lines (no timestamps, so a line reads the same on every turn)
        "recent_chat": [
            f"<{_clean(e.get('username', '?'))}> {_clean(e.get('message', ''))[:200]}"
            for e in history
            if now - float(e.get("time", now)) < 300
        ][-6:],
        "waiting_for_reply": unanswered[-3:],
        "allowed": allowed,
        "goal": mind.goal,
    }
    lines.append("")
    lines.append("EXACT DATA (JSON): " + json.dumps(exact, separators=(",", ":"), ensure_ascii=False))
    lines.append("")
    lines.append("What do you do next? Think like a human player, then reply with the JSON object only.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- chat output
_ASSISTANT = re.compile(r"^(?:as an ai(?: language model)?|as a (?:minecraft )?bot|i am an ai)[,.!]?\s*", re.I)
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]")
_PREFIX = re.compile(r"^\s*(?:<[^>]{1,20}>|\[[^\]]{1,20}\]|[A-Za-z0-9_]{3,16}:)\s*")


def humanize_message(message: str, name: str = "", max_length: int = 120) -> str:
    """Clean an LLM chat line: one line, no quotes/markdown/name prefix, cut at a word boundary."""
    text = " ".join(str(message or "").split())
    if name and text.lower().startswith(name.lower() + ":"):
        text = text[len(name) + 1:].strip()
    text = _PREFIX.sub("", text) if re.match(r"^\s*(<|\[)", text) else text
    text = text.replace("**", "").replace("`", "").replace("__", "")
    text = _ASSISTANT.sub("", text).strip()
    text = _EMOJI.sub("", text).strip()  # Minecraft chat shows most emoji as boxes
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    text = text.lstrip("/")  # never send commands by accident
    if len(text) > max_length:
        cut = text[:max_length].rsplit(" ", 1)[0]
        text = cut if len(cut) >= max_length // 2 else text[:max_length]
    return text.strip()


def typing_delay(message: str, rng: random.Random | None = None) -> float:
    """Seconds a person needs to read, think and type ``message`` (capped at 4 s)."""
    rng = rng or random
    return min(4.0, 0.5 + len(message) * 0.045 + rng.uniform(0.0, 0.8))
