"""Built-in thinking traces for the Minecraftism *Thinking* editions (no teacher model needed).

A thinking model is trained on ``<think>reasoning</think>`` followed by the answer. Fine-tuning a
thinking model on answers alone teaches it to skip thinking, so every training row gets a trace
here - written by rules from the same game state the turn prompt shows:

1. how the bot is doing (health, food, armour),
2. what is around it (hostile mobs with their distance and the mob-database facts, players,
   creatures it does not know),
3. what was said in chat and whether it is meant for the bot,
4. its goal and recent failures,
5. what it is not allowed to do, when that matters,
6. why the chosen action is the right one.

Traces are short (the bot plays in real time), plain sentences, and never contain ``{`` or ``}`` so
the JSON answer after ``</think>`` is the only object in a reply. The non-thinking editions
ignore the ``reasoning`` field.
"""

from __future__ import annotations

import json
import math
import random
import re
from typing import Any

from pymc_bot.inventory import Inventory
from pymc_bot.knowledge import shorten
from pymc_bot.llm_train import answer_plan
from pymc_bot.mobs import MobDatabase, load_mobs

#: Sentences longer than this are cut (a trace is a few lines, not an essay).
MAX_TRACE = 900


def clean(text: Any) -> str:
    """One line, no braces (they would look like a JSON answer inside the trace)."""
    text = " ".join(str(text or "").split())
    return text.replace("{", "(").replace("}", ")")


def _name(kind: Any) -> str:
    return str(kind or "").split(":")[-1].replace("_", " ").strip() or "something"


def _hp(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 20.0


def _dist(entity: dict[str, Any]) -> float:
    for key in ("dist", "distance"):
        try:
            return float(entity[key])
        except (KeyError, TypeError, ValueError):
            continue
    return 99.0


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


# --------------------------------------------------------------------------- pieces
def status_line(state: dict[str, Any], rng: random.Random) -> str:
    health, food = _hp(state.get("health", 20)), _hp(state.get("food", 20))
    parts = [f"Health {_fmt(health)}/20, food {_fmt(food)}/20."]
    if health <= 6:
        parts.append(rng.choice(["That is low, one or two hits could kill me.", "I'm badly hurt, so no risky fights.",
                                 "Low health: staying alive comes first."]))
    elif health <= 12:
        parts.append("I'm a bit hurt.")
    if food <= 6:
        parts.append(rng.choice(["Food is low, I won't heal and soon I start starving.", "I'm hungry and need to eat.",
                                 "Hunger is low, eating is a priority."]))
    elif health > 12 and food > 14:
        parts.append(rng.choice(["I'm fine.", "Healthy and fed.", "No problems there."]))
    armour = [a for a in state.get("armor") or [] if a]
    if armour:
        parts.append("Wearing " + ", ".join(_name(a) for a in armour[:4]) + ".")
    return " ".join(parts)


def place_line(state: dict[str, Any]) -> str:
    dimension = _name(state.get("dimension") or "overworld").replace("the ", "")
    time_of_day = str(state.get("time_of_day") or "").strip()
    pos = state.get("position") or {}
    where = f"It's {time_of_day} in the {dimension}" if time_of_day and dimension == "overworld" else f"I'm in the {dimension}"
    try:
        y = float(pos.get("y"))
    except (TypeError, ValueError):
        y = None
    if y is not None and dimension == "overworld" and y < 40:
        where += f", underground at y {int(y)}"
    return where + "."


def threats_line(state: dict[str, Any], db: MobDatabase, limit: int = 2) -> tuple[str, list[dict[str, Any]]]:
    entities = sorted((e for e in state.get("entities") or [] if isinstance(e, dict)), key=_dist)
    hostile = [e for e in entities if e.get("hostile")]
    lines: list[str] = []
    for entity in hostile[:limit]:
        mob = db.get(_name(entity.get("type")))
        fact = ""
        if mob is not None:
            fact = " " + clean(shorten(mob.get("fact") or mob.get("behavior") or "", 150))
            damage = db.max_damage(mob)
            if damage:
                fact += f" It hits for up to {_fmt(float(damage))} on normal."
        lines.append(f"A {_name(entity.get('type'))} is {_fmt(round(_dist(entity), 1))} blocks away.{fact}")
    others = [e for e in entities if not e.get("hostile")]
    for entity in others[:2]:
        kind = str(entity.get("type") or "")
        if kind == "unknown":
            lines.append(f"There's a creature my client can't name {_fmt(round(_dist(entity), 1))} blocks away, "
                         "probably a mob newer than mineflayer.")
        elif ":" in kind and not kind.startswith("minecraft:"):
            lines.append(f"A {_name(kind)} from a mod is nearby. It's not in my mob list, so I don't know if it's dangerous.")
        elif not hostile and db.get(_name(kind)) is not None and _dist(entity) <= 8:
            lines.append(f"A {_name(kind)} is close, harmless.")
    if not lines:
        return "No hostile mobs around.", hostile
    return " ".join(lines), hostile


def players_line(state: dict[str, Any]) -> str:
    players = [p for p in state.get("players") or [] if isinstance(p, dict) and p.get("name")]
    if not players:
        online = [str(n) for n in state.get("online_players") or []][:3]
        return f"Nobody is near me{' (online: ' + ', '.join(online) + ')' if online else ''}."
    near = sorted(players, key=_dist)[:3]
    return " ".join(f"{p['name']} is {_fmt(round(_dist(p), 1))} blocks away." for p in near)


_QUESTION_WORDS = ("how", "what", "where", "when", "why", "who", "which", "is", "are", "can", "do", "does", "will",
                   "should", "could", "would", "r", "u")


def chat_line(history: list[dict[str, Any]], name: str, db: MobDatabase | None = None) -> str:
    if not history:
        return ""
    last = history[-1]
    speaker, message = clean(last.get("username")), clean(last.get("message"))[:140]
    if not speaker or not message:
        return ""
    low = message.lower()
    words = [w for w in re.findall(r"[a-z0-9_']+", low) if w != name.lower()]
    question = "?" in message or bool(words and words[0] in _QUESTION_WORDS)
    to_me = name.lower() in low or "bot" in low or question or len(history) == 1
    kind = "a question" if question else "a message"
    line = f"{speaker} said: '{message}'. " + (f"That's {kind} for me." if to_me else "Probably not aimed at me.")
    found = (db or load_mobs()).find_in_text(message) if question else []
    if found:
        mob = (db or load_mobs()).mobs[found[0][0]]
        line += f" What I know about the {mob['name'].lower()}: {clean(shorten((db or load_mobs()).prompt_line(mob), 200))}."
    return line


def memory_line(mind: Any) -> str:
    if mind is None:
        return ""
    parts = []
    goal = clean(getattr(mind, "goal", ""))
    if goal:
        parts.append(f"My goal: {goal}.")
    actions = list(getattr(mind, "actions", []) or [])
    failures = 0
    for memory in reversed(actions):
        if getattr(memory, "ok", True):
            break
        failures += 1
    if failures:
        detail = clean(getattr(actions[-1], "detail", ""))
        parts.append(f"The last {failures} tr{'ies' if failures > 1 else 'y'} failed ({detail}), repeating it won't help.")
    elif actions:
        last = actions[-1]
        parts.append(f"Last thing I did: {clean(getattr(last, 'summary', ''))}, it worked.")
    return " ".join(parts)


def rules_line(allowed: dict[str, bool], action: str, hostile: list[dict[str, Any]], history: list[dict[str, Any]]) -> str:
    off = []
    if history and allowed.get("chat") is False:
        off.append("chat is switched off, so I can't type")
    if hostile and allowed.get("fighting") is False:
        off.append("fighting is switched off")
    if allowed.get("movement") is False:
        off.append("I'm not allowed to move")
    return ("Rules: " + "; ".join(off) + ".") if off else ""


def _held_or_best(state: dict[str, Any], kind: str) -> str:
    inv = Inventory.from_snapshot(state)
    index = inv.best_weapon() if kind == "weapon" else inv.best_food(_hp(state.get("food", 20)), _hp(state.get("health", 20)))
    if index is None:
        return ""
    stack = inv.slots[index]
    return _name(stack.name) if stack else ""


def action_line(decision: dict[str, Any], state: dict[str, Any], hostile: list[dict[str, Any]], rng: random.Random) -> str:
    action = str(decision.get("action") or "").lower()
    threat = _name(hostile[0].get("type")) if hostile else "mob"
    player = clean(decision.get("player"))
    item = _name(decision.get("item")) if decision.get("item") else ""
    block = _name(decision.get("block")) if decision.get("block") else ""
    if action == "flee":
        return rng.choice([f"Fighting the {threat} isn't worth it right now, so I get distance.",
                           f"Best move: back away from the {threat} and stay alive."])
    if action == "fight":
        weapon = _held_or_best(state, "weapon")
        mob = _name(decision.get("mob") or threat)
        return f"I can take the {mob}" + (f" with my {weapon}" if weapon else "") + ", better to deal with it before it hits me."
    if action == "eat":
        food = _held_or_best(state, "food")
        return f"Eat now{' - I have ' + food if food else ''}, so I can heal."
    if action == "say":
        return rng.choice(["Answer in chat: short, friendly, in character.", "Reply in chat, one short line.",
                           "I'll answer them in chat, briefly."])
    if action == "give":
        return f"I have spare {item or 'items'}, so I'll give {player or 'them'} {decision.get('count') or 'some'}."
    if action in ("wear_armor", "equip"):
        return "There's armour in my bag that I'm not wearing; putting it on protects me."
    if action == "mine":
        return f"Mine {block or 'the block'}" + (" - it moves my goal forward." if decision.get("goal") else ".")
    if action == "wander":
        return "Nothing useful here, so I'll walk somewhere new and look."
    if action == "follow":
        return f"Stick with {player or 'them'}" + (" - safer than being alone." if hostile else ".")
    if action == "look_at_player":
        return f"Turn to {player or 'them'} so it feels like I'm paying attention."
    if action == "emote":
        return f"A quick {decision.get('kind') or 'wave'} says enough without typing."
    if action == "look_around":
        return "Nothing urgent, so I'll look around and check my surroundings."
    if action == "search":
        return f"I don't know this one, so I'll look it up first ({clean(decision.get('query'))})."
    if action == "sleep":
        return "Sleeping skips the night, resets the phantom timer and sets my spawn."
    if action == "goto":
        return "I'll walk over there."
    if action == "jump":
        return "A little hop, like players do."
    return f"So: {action}."


# --------------------------------------------------------------------------- traces
def _limit(text: str) -> str:
    text = clean(text)
    if len(text) <= MAX_TRACE:
        return text
    cut = text[:MAX_TRACE].rsplit(". ", 1)[0]
    return cut + "."


def turn_reasoning(state: dict[str, Any], decision: dict[str, Any] | str, *, name: str = "PyMC_Bot",
                   history: list[dict[str, Any]] | None = None, mind: Any = None, allowed: dict[str, bool] | None = None,
                   db: MobDatabase | None = None, rng: random.Random | None = None) -> str:
    """The thinking trace for one game turn, ending with why ``decision`` is the move."""
    rng = rng or random.Random(0)
    db = db or load_mobs()
    history = history or []
    if isinstance(decision, str):
        try:
            decision = json.loads(decision)
        except ValueError:
            decision = {}
    status, place = status_line(state, rng), place_line(state)
    threats, hostile = threats_line(state, db)
    players, chat, memory = players_line(state), chat_line(history, name, db), memory_line(mind)
    # what matters most comes first, like a player would notice it - never the same opener on every turn (all
    # 6,000 traces used to start with "Health X/20, food Y/20.", even with a creeper 3 blocks away)
    health, food = _hp(state.get("health", 20)), _hp(state.get("food", 20))
    near = min((_dist(e) for e in hostile), default=99.0)
    first: list[str] = []
    if health <= 6:
        first.append(status)
    if hostile and near <= 8:
        first.append(threats)
    if chat:
        first.append(chat)
    if food <= 6 and status not in first:
        first.append(status)
    rest = [p for p in (status, place, threats, players, memory) if p and p not in first]
    rng.shuffle(rest)
    tail = " ".join(p for p in (rules_line(allowed or {}, str(decision.get("action") or ""), hostile, history),
                                 action_line(decision, state, hostile, rng)) if p)
    body: list[str] = []
    for part in first + rest:  # the least important context goes when it is too long - never the decision
        if part and len(" ".join([*body, part, tail])) <= MAX_TRACE:
            body.append(part)
    return _limit(" ".join([*body, tail]))


def decision_reasoning(decision: dict[str, Any] | str, prompt: str = "") -> str:
    """A trace when only the prompt text is known (recorded turns): what the prompt says, then why."""
    if isinstance(decision, str):
        try:
            decision = json.loads(decision)
        except ValueError:
            decision = {}
    lines = [line.strip() for line in str(prompt or "").splitlines() if line.strip()]
    keep = [line for line in lines if re.match(r"(?i)(health|food|you are|near|entities|players|chat|goal|mobs?|time)\b", line)]
    summary = " ".join(keep[:4])[:300]
    thought = clean(decision.get("thought"))
    parts = [f"What I see: {summary}" if summary else "", f"{thought[:1].upper()}{thought[1:]}." if thought else "",
             action_line(decision, {}, [], random.Random(0))]
    return _limit(" ".join(p for p in parts if p))


def qa_reasoning(question: str, answer: str, source: str = "", db: MobDatabase | None = None,
                 rng: random.Random | None = None) -> str:
    """A trace for a knowledge question: a plan for the answer it needs (``llm_train.answer_plan``), plus the
    mob-database notes when the question really is about a mob.

    Earlier traces were one fixed form ("The question: ... What I know: ... Key point: <the answer's first
    sentence>. Answer directly and briefly."). A model trained on thousands of them writes that form for every
    question and fills the slots with invented facts (a stat card for a piston, a "key point" that is wrong),
    always "briefly", even for "explain how" questions. So: no copied key point, no fixed closing line, and the
    mob notes only for mobs."""
    del rng  # kept for callers; the plan is chosen from the question itself
    db = db or load_mobs()
    question = clean(question)
    parts = [clean(answer_plan(question, answer))]
    found = db.find_in_text(question)
    if found:
        mob = db.mobs[found[0][0]]
        name = mob["name"].lower()
        line = clean(shorten(db.prompt_line(mob), 220))
        line = line[len(name):].strip() if line.lower().startswith(name) else line
        card = re.match(r"\((.*?)\):?\s*(.*)", line)  # "(hostile, 30 HP): text" -> "hostile, 30 HP; text"
        line = f"{card.group(1)}; {card.group(2)}" if card else line
        parts.append(f"My notes on the {name}: {line}.")
    elif source == "wiki":
        parts.append("The Minecraft Wiki covers this; I'll put it in plain words.")
    if re.search(r"\b(damage|hit|hp|health|hearts?)\b", question, re.I):
        parts.append("Damage is in health points, 2 HP is one heart.")
    return _limit(" ".join(parts))


# --------------------------------------------------------------------------- what is in view (VL)
def relative_direction(dx: float, dz: float, yaw: float) -> tuple[str, float]:
    """Where (dx, dz) is for a player looking along ``yaw`` (Minecraft: 0 = south/+Z, 90 = west/-X).

    Returns a word ("ahead", "ahead-left", "left", "behind"...) and the angle in degrees
    (negative = left, positive = right).
    """
    forward = (-math.sin(math.radians(yaw)), math.cos(math.radians(yaw)))
    # Facing south (+Z) west (-X) is on your right; facing north (-Z) east (+X) is. With
    # forward f, (dz*f.x - dx*f.z) is positive exactly for the right-hand side.
    angle = math.degrees(math.atan2(dz * forward[0] - dx * forward[1], dx * forward[0] + dz * forward[1]))
    if abs(angle) <= 15:
        word = "ahead"
    elif abs(angle) <= 50:
        word = "ahead-right" if angle > 0 else "ahead-left"
    elif abs(angle) <= 120:
        word = "right" if angle > 0 else "left"
    else:
        word = "behind"
    return word, angle


#: Half of Minecraft's default horizontal field of view at 16:9 (about 70 vertical -> ~102 horizontal).
HALF_FOV = 50.0


def visible_entities(observation: dict[str, Any]) -> list[dict[str, Any]]:
    """Entities from a recorded sample that are inside the camera's horizontal view."""
    yaw = float(observation.get("yaw") or 0.0)
    seen: dict[tuple[str, int, int], dict[str, Any]] = {}
    for entity in (observation.get("entities") or []) + (observation.get("nearby") or []):
        if not isinstance(entity, dict):
            continue
        try:
            dx, dz = float(entity.get("dx")), float(entity.get("dz"))
        except (TypeError, ValueError):
            continue
        word, angle = relative_direction(dx, dz, yaw)
        if abs(angle) > HALF_FOV:
            continue
        key = (str(entity.get("type")), round(dx), round(dz))
        seen.setdefault(key, {**entity, "where": word, "angle": angle, "dist": _dist(entity) if _dist(entity) < 99 else math.hypot(dx, dz)})
    return sorted(seen.values(), key=_dist)


def describe_view(observation: dict[str, Any], db: MobDatabase | None = None, rng: random.Random | None = None) -> str:
    """A grounded caption of a recorded screenshot, written from the sample recorded with it."""
    rng = rng or random.Random(0)
    db = db or load_mobs()
    dimension = _name(observation.get("dimension") or "overworld").replace("the ", "")
    parts = [f"This is the {dimension}" + (" from first person." if rng.random() < 0.5 else ", first-person view.")]
    pitch = float(observation.get("pitch") or 0.0)
    if pitch > 45:
        parts.append("I'm looking down at the ground.")
    elif pitch < -45:
        parts.append("I'm looking up at the sky.")
    blocks = [b for b in observation.get("blocks") or [] if isinstance(b, list) and len(b) == 4]
    below = next((b[3] for b in blocks if b[0] == 0 and b[1] == -1 and b[2] == 0), None)
    if below and not str(below).endswith("air"):
        parts.append(f"I'm standing on {_name(below)}.")
    solid = [str(b[3]) for b in blocks if not str(b[3]).endswith("air")]
    if solid:
        top = sorted({_name(s) for s in solid}, key=lambda s: -sum(1 for x in solid if _name(x) == s))[:3]
        parts.append("Blocks right around me: " + ", ".join(top) + ".")
    entities = visible_entities(observation)
    if entities:
        described = []
        for entity in entities[:4]:
            kind = "player" if entity.get("player") else _name(entity.get("type"))
            mob = db.get(kind)
            danger = " (hostile)" if entity.get("hostile") or (mob and mob.get("hostile")) else ""
            described.append(f"a {kind}{danger} about {_fmt(round(float(entity['dist'])))} blocks {entity['where']}")
        parts.append("In view: " + "; ".join(described) + ".")
    else:
        parts.append("No mobs or players in view.")
    held = observation.get("held_item")
    if held:
        parts.append(f"I'm holding {_name(held)}.")
    health, food = _hp(observation.get("health", 20)), _hp(observation.get("food", 20))
    parts.append(f"The HUD shows {_fmt(health / 2)} hearts and {_fmt(food / 2)} food.")
    if observation.get("sprinting"):
        parts.append("I'm sprinting.")
    elif observation.get("sneaking"):
        parts.append("I'm sneaking.")
    return clean(" ".join(parts))


def vision_reasoning(observation: dict[str, Any], decision: dict[str, Any] | None = None,
                     db: MobDatabase | None = None, rng: random.Random | None = None) -> str:
    """Thinking trace for an image turn: read the picture first, then decide."""
    rng = rng or random.Random(0)
    parts = ["Looking at the screenshot.", describe_view(observation, db, rng)]
    if decision:
        parts.append(action_line(decision, {}, [e for e in visible_entities(observation) if e.get("hostile")], rng))
    return _limit(" ".join(parts))
