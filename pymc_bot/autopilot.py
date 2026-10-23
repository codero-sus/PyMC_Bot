"""Autopilot: the survival reflexes every bot has on its own.

A player does a lot without thinking about it: swims up for air, eats when hungry, puts
on better armour, hits the zombie that walks up, goes to bed at night, drops a torch in
a dark cave, picks up what fell on the floor. The autopilot gives every bot (the main
one and the whole fleet) those reflexes, so it can be left alone on a server.

:func:`evaluate` is a pure function - snapshot in, at most one :class:`Plan` out - which
keeps every rule easy to test. :class:`Autopilot` runs it for each bot every couple of
seconds (same shape as :class:`pymc_bot.antiafk.AntiAfkKeeper`) and carries plans out
through the bot's normal actions and skills.

Rules, most urgent first::

    respawn  surface  extinguish  totem  flee  eat          <- emergencies, run even when busy
    armor  weapon  defend  sleep  shelter  torch  collect  tidy  home  idle_job

Set ``PYMC_BOT_NO_AUTOPILOT=1`` to keep the background loop from starting (the tests do).
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pymc_bot.config import AutopilotSettings, ConfigStore
from pymc_bot.events import EventLog
from pymc_bot.inventory import INVENTORY_SIZE, Inventory, is_food, short, weapon_score

if TYPE_CHECKING:  # pragma: no cover
    from pymc_bot.bot import MinecraftBot

#: Rules in the order they are checked (the panel lists them like this too).
RULES: tuple[str, ...] = (
    "respawn", "surface", "extinguish", "totem", "flee", "eat",
    "armor", "weapon", "defend", "sleep", "shelter", "torch", "collect", "tidy", "home", "idle_job",
)
EMERGENCY_RULES = frozenset({"respawn", "surface", "extinguish", "totem", "flee", "eat"})

RULE_INFO: dict[str, str] = {
    "respawn": "Click respawn after dying.",
    "surface": "Swim up when the air bar runs low.",
    "extinguish": "Put out fire (water bucket, or jump into water).",
    "totem": "Totem of undying into the off-hand when health is low.",
    "flee": "Run from mobs at low health, and back off from creepers.",
    "eat": "Eat when hungry (best food for the moment).",
    "armor": "Put on better armour as soon as it is in the backpack.",
    "weapon": "Take the best weapon out when a mob comes close.",
    "defend": "Fight hostile mobs that come within range.",
    "sleep": "Go to bed at night (when no monster is close).",
    "shelter": "No bed at night and monsters around: wall itself in.",
    "torch": "Place a torch when standing in the dark underground.",
    "collect": "Pick up items lying nearby.",
    "tidy": "Toss junk when the backpack is nearly full.",
    "home": "Walk home when night falls (home = the set_home skill).",
    "idle_job": "A chore while idle and the AI is off: farm, fish, chop, mine or collect.",
}

#: Monsters that leave you alone until you hit them - never start a fight with these.
NEUTRAL_MOBS = frozenset({"enderman", "zombified_piglin", "piglin", "spider", "cave_spider", "wolf", "bee", "llama",
                          "panda", "polar_bear", "iron_golem", "dolphin", "goat", "trader_llama"})
SLEEP_START, SLEEP_END = 12542, 23460  # ticks in which a bed can be used (clear weather)
IDLE_SKILLS: dict[str, tuple[str, dict[str, Any]]] = {
    "farm": ("harvest", {}),
    "fish": ("fish", {"seconds": 60}),
    "chop": ("chop_tree", {"count": 8}),
    "mine": ("mine_ore", {"count": 4}),
}

TICK = 2.0


@dataclass
class Plan:
    """One thing the autopilot wants a bot to do now."""

    rule: str
    action: str  # "skill:<name>" or the name of a MinecraftBot method
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    cooldown: float = 5.0  # seconds before the same rule may fire again
    fail_cooldown: float | None = None  # longer wait after it failed (no bed, nothing to eat...)
    interrupt: bool = False  # stop the current action first (drowning, burning, fleeing)

    @property
    def emergency(self) -> bool:
        return self.rule in EMERGENCY_RULES

    def to_dict(self) -> dict[str, Any]:
        return {"rule": self.rule, "action": self.action, "params": dict(self.params), "reason": self.reason}


@dataclass
class Situation:
    """What a rule may look at (everything except the snapshot has a safe default)."""

    snap: dict[str, Any]
    now: float = 0.0
    busy: bool = False
    idle_seconds: float = 0.0
    ai_busy: bool = False
    home: dict[str, Any] | None = None
    cooldowns: dict[str, float] = field(default_factory=dict)  # rule -> monotonic time it may fire again
    failures: dict[str, float] = field(default_factory=dict)  # rule -> monotonic time of the last failure
    inv: Inventory | None = None

    def __post_init__(self) -> None:
        if self.inv is None:
            self.inv = Inventory.from_snapshot(self.snap)

    def ready(self, rule: str) -> bool:
        return self.now >= self.cooldowns.get(rule, 0.0)

    def failed_within(self, rule: str, seconds: float) -> bool:
        return rule in self.failures and self.now - self.failures[rule] <= seconds


# ------------------------------------------------------------------------------ snapshot helpers
def _num(snap: dict[str, Any], key: str, default: float) -> float:
    value = snap.get(key)
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def mob_kind(entity: dict[str, Any]) -> str:
    return short(str(entity.get("type") or ""))


def hostiles(snap: dict[str, Any], radius: float) -> list[dict[str, Any]]:
    """Hostile mobs within ``radius`` blocks, nearest first."""
    out = [
        entity for entity in snap.get("entities") or []
        if entity.get("hostile") and not entity.get("player") and float(entity.get("dist") or 99.0) <= radius
    ]
    return sorted(out, key=lambda entity: float(entity.get("dist") or 99.0))


def is_night(snap: dict[str, Any]) -> bool:
    ticks = snap.get("world_time")
    if isinstance(ticks, (int, float)):
        return SLEEP_START <= float(ticks) % 24000 < SLEEP_END
    return snap.get("time_of_day") == "night"


def free_slots(inv: Inventory) -> int:
    return sum(1 for index in range(INVENTORY_SIZE) if inv.slots[index] is None)


def _distance_home(snap: dict[str, Any], home: dict[str, Any] | None) -> float | None:
    pos = snap.get("position")
    if not pos or not home:
        return None
    if (home.get("dimension") or "overworld") != (snap.get("dimension") or "overworld"):
        return None
    return math.dist((float(pos["x"]), float(pos["z"])), (float(home["x"]), float(home["z"])))


# ------------------------------------------------------------------------------------- the rules
def evaluate(settings: AutopilotSettings, situation: Situation) -> Plan | None:
    """The single most urgent thing to do, or ``None``. Pure: never touches the bot."""
    if not settings.enabled:
        return None
    for rule in RULES:
        if not situation.ready(rule):
            continue
        if situation.busy and rule not in EMERGENCY_RULES:
            continue
        plan = RULE_FUNCS[rule](settings, situation)
        if plan is not None:
            return plan
    return None


def _rule_respawn(settings: AutopilotSettings, s: Situation) -> Plan | None:
    if settings.auto_respawn and (s.snap.get("dead") or _num(s.snap, "health", 20.0) <= 0):
        return Plan("respawn", "skill:respawn", reason="died", cooldown=5.0)
    return None


def _rule_surface(settings: AutopilotSettings, s: Situation) -> Plan | None:
    if settings.auto_surface and s.snap.get("in_water") and _num(s.snap, "oxygen", 20.0) < 8:
        return Plan("surface", "skill:surface", reason=f"air {int(_num(s.snap, 'oxygen', 20.0))}/20",
                    cooldown=3.0, interrupt=True)
    return None


def _rule_extinguish(settings: AutopilotSettings, s: Situation) -> Plan | None:
    if settings.auto_extinguish and s.snap.get("on_fire") and not s.snap.get("in_water"):
        return Plan("extinguish", "skill:extinguish", reason="on fire", cooldown=4.0, fail_cooldown=8.0,
                    interrupt=True)
    return None


def _rule_totem(settings: AutopilotSettings, s: Situation) -> Plan | None:
    inv = s.inv
    if not settings.auto_totem or _num(s.snap, "health", 20.0) > settings.totem_health:
        return None
    if inv.offhand and inv.offhand.name == "totem_of_undying":
        return None
    if inv.find("totem_of_undying") is None:
        return None
    return Plan("totem", "skill:totem", reason=f"health {_num(s.snap, 'health', 20.0):.0f}", cooldown=10.0)


def _rule_flee(settings: AutopilotSettings, s: Situation) -> Plan | None:
    if not settings.auto_flee:
        return None
    health = _num(s.snap, "health", 20.0)
    near = hostiles(s.snap, 10.0)
    if health <= settings.flee_health and near:
        return Plan("flee", "flee", {"distance": 14.0}, reason=f"health {health:.0f} and a {mob_kind(near[0])} close",
                    cooldown=6.0, interrupt=True)
    if settings.flee_creepers:
        creeper = next((e for e in hostiles(s.snap, 5.0) if mob_kind(e) == "creeper"), None)
        if creeper is not None:
            return Plan("flee", "flee", {"distance": 10.0}, reason="creeper", cooldown=4.0, interrupt=True)
    return None


def _rule_eat(settings: AutopilotSettings, s: Situation) -> Plan | None:
    if not settings.auto_eat:
        return None
    food = _num(s.snap, "food", 20.0)
    health = _num(s.snap, "health", 20.0)
    hungry = food <= settings.eat_below or (health < 12 and food < 18)
    if not hungry or not any(is_food(stack.name) for _, stack in s.inv.items()):
        return None
    if s.busy and food > 6:  # while busy only a nearly empty bar is an emergency
        return None
    return Plan("eat", "eat", reason=f"food {food:.0f}/20", cooldown=8.0, fail_cooldown=30.0)


def _rule_armor(settings: AutopilotSettings, s: Situation) -> Plan | None:
    if settings.auto_armor and s.inv.armor_upgrades():
        return Plan("armor", "wear_best_armor", reason="better armour in the backpack", cooldown=10.0,
                    fail_cooldown=60.0)
    return None


def _rule_weapon(settings: AutopilotSettings, s: Situation) -> Plan | None:
    if not settings.auto_weapon or not hostiles(s.snap, settings.defend_range + 6):
        return None
    best = s.inv.best_weapon()
    held = s.inv.held
    if best is None or best == s.inv.selected:
        return None
    if held and weapon_score(held.name) >= weapon_score(s.inv.slots[best].name):
        return None
    return Plan("weapon", "skill:equip_weapon", reason="a mob is coming", cooldown=10.0)


def _rule_defend(settings: AutopilotSettings, s: Situation) -> Plan | None:
    if not settings.auto_defend or _num(s.snap, "health", 20.0) <= settings.flee_health:
        return None
    for mob in hostiles(s.snap, settings.defend_range):
        kind = mob_kind(mob)
        if kind in NEUTRAL_MOBS or kind == "creeper":
            continue
        return Plan("defend", "fight", {"mob": kind}, reason=f"{kind} {float(mob.get('dist') or 0):.0f} blocks away",
                    cooldown=2.0)
    return None


def _rule_sleep(settings: AutopilotSettings, s: Situation) -> Plan | None:
    if not settings.auto_sleep or s.snap.get("sleeping") or not is_night(s.snap):
        return None
    if (s.snap.get("dimension") or "overworld").replace("minecraft:", "") != "overworld":
        return None  # beds explode in the nether and the end
    if hostiles(s.snap, 8.0):
        return None  # "you may not rest now, there are monsters nearby"
    return Plan("sleep", "sleep_in_bed", reason="night", cooldown=15.0, fail_cooldown=90.0)


def _rule_shelter(settings: AutopilotSettings, s: Situation) -> Plan | None:
    if not settings.auto_shelter or s.snap.get("sleeping") or not is_night(s.snap):
        return None
    if settings.auto_sleep and not s.failed_within("sleep", 180.0):
        return None  # try the bed first
    if not hostiles(s.snap, 16.0) or s.inv.building_block() is None:
        return None
    return Plan("shelter", "skill:build_shelter", reason="night, no bed and monsters around", cooldown=600.0,
                fail_cooldown=120.0)


def _rule_torch(settings: AutopilotSettings, s: Situation) -> Plan | None:
    if not settings.auto_torch or s.inv.count("torch") <= 0:
        return None
    sky = s.snap.get("sky_light")
    block = s.snap.get("block_light")
    if not isinstance(sky, (int, float)) or not isinstance(block, (int, float)):
        return None
    if sky < 8 and block <= settings.torch_light:
        return Plan("torch", "skill:place_torch", reason=f"dark here (light {int(block)})", cooldown=6.0,
                    fail_cooldown=20.0)
    return None


def _rule_collect(settings: AutopilotSettings, s: Situation) -> Plan | None:
    if not settings.auto_collect or free_slots(s.inv) == 0:
        return None
    near = [item for item in s.snap.get("ground_items") or [] if float(item.get("dist") or 99.0) <= settings.collect_radius]
    if not near:
        return None
    return Plan("collect", "collect_items", {"radius": settings.collect_radius + 2},
                reason=f"{len(near)} item(s) on the ground", cooldown=10.0, fail_cooldown=30.0)


def _rule_tidy(settings: AutopilotSettings, s: Situation) -> Plan | None:
    from pymc_bot.skills import junk_slots

    if not settings.auto_tidy or free_slots(s.inv) > settings.tidy_free_slots:
        return None
    if junk_slots(s.inv):
        return Plan("tidy", "skill:toss_junk", reason="backpack nearly full", cooldown=60.0, fail_cooldown=180.0)
    return Plan("tidy", "skill:deposit", reason="backpack full - into the nearest chest", cooldown=120.0,
                fail_cooldown=300.0)


def _rule_home(settings: AutopilotSettings, s: Situation) -> Plan | None:
    if not settings.auto_home or not is_night(s.snap) or s.snap.get("sleeping"):
        return None
    distance = _distance_home(s.snap, s.home)
    if distance is None or distance <= 24:
        return None
    return Plan("home", "skill:go_home", reason=f"night, home is {distance:.0f} blocks away", cooldown=300.0)


def _rule_idle_job(settings: AutopilotSettings, s: Situation) -> Plan | None:
    job = settings.idle_job
    if job == "none" or s.ai_busy or s.idle_seconds < settings.idle_after:
        return None
    if job == "gather_items":
        if not s.snap.get("ground_items"):
            return None
        return Plan("idle_job", "collect_items", {"radius": 16.0}, reason="idle", cooldown=settings.idle_every)
    if job == "fish" and s.inv.count("fishing_rod") <= 0:
        return None
    name, params = IDLE_SKILLS[job]
    return Plan("idle_job", f"skill:{name}", dict(params), reason=f"idle - {job}", cooldown=settings.idle_every,
                fail_cooldown=max(settings.idle_every, 300.0))


RULE_FUNCS: dict[str, Callable[[AutopilotSettings, Situation], Plan | None]] = {
    "respawn": _rule_respawn, "surface": _rule_surface, "extinguish": _rule_extinguish, "totem": _rule_totem,
    "flee": _rule_flee, "eat": _rule_eat, "armor": _rule_armor, "weapon": _rule_weapon, "defend": _rule_defend,
    "sleep": _rule_sleep, "shelter": _rule_shelter, "torch": _rule_torch, "collect": _rule_collect,
    "tidy": _rule_tidy, "home": _rule_home, "idle_job": _rule_idle_job,
}


# ----------------------------------------------------------------------------------- the keeper
@dataclass
class BotAutopilot:
    """Per-bot memory: cooldowns, failures and what happened last."""

    cooldowns: dict[str, float] = field(default_factory=dict)
    failures: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    in_flight: bool = False
    last_rule: str = ""
    last_message: str = ""
    last_at: float = 0.0
    paused: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": dict(self.counts), "last_rule": self.last_rule, "last_message": self.last_message,
            "last_at": self.last_at, "busy": self.in_flight, "paused": self.paused,
        }


class Autopilot:
    """Runs :func:`evaluate` for every bot and carries the plans out."""

    def __init__(
        self,
        bots: Callable[[], Sequence[MinecraftBot]],
        store: ConfigStore,
        log: EventLog | None = None,
        ai_busy: Callable[[MinecraftBot], bool] | None = None,
    ) -> None:
        self._bots = bots
        self.store = store
        self.log = log or EventLog()
        self._ai_busy = ai_busy or (lambda bot: False)
        self._memory: dict[int, BotAutopilot] = {}
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._workers: set[threading.Thread] = set()
        self.actions = 0

    # ------------------------------------------------------------- lifecycle
    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def settings(self) -> AutopilotSettings:
        return self.store.settings.autopilot

    def start(self) -> bool:
        if os.environ.get("PYMC_BOT_NO_AUTOPILOT"):
            return False
        with self._lock:
            if self.running:
                return False
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._loop, name="autopilot", daemon=True)
            self._thread.start()
        return True

    def stop(self, wait: bool = True) -> bool:
        with self._lock:
            if not self.running:
                return False
            self._stop_event.set()
            thread = self._thread
        if wait and thread is not None:
            thread.join(timeout=5.0)
        return True

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            settings = self.settings
            try:
                if settings.enabled:
                    self.tick(settings)
            except Exception as exc:  # pragma: no cover - the autopilot must never die
                self.log.add(f"Autopilot hiccup: {exc}", "warn", "autopilot")
            self._stop_event.wait(settings.tick_seconds)

    # ------------------------------------------------------------------ work
    def memory(self, bot: MinecraftBot) -> BotAutopilot:
        with self._lock:
            return self._memory.setdefault(id(bot), BotAutopilot())

    def situation(self, bot: MinecraftBot, now: float | None = None) -> Situation:
        memory = self.memory(bot)
        snap = bot.snapshot()
        return Situation(
            snap=snap,
            now=time.monotonic() if now is None else now,
            busy=bot.busy,
            idle_seconds=bot.idle_seconds,
            ai_busy=bool(self._ai_busy(bot)),
            home=_home_of(bot) if self.settings.auto_home else None,
            cooldowns=dict(memory.cooldowns),
            failures=dict(memory.failures),
        )

    def plan_for(self, bot: MinecraftBot) -> Plan | None:
        if not bot.connected:
            return None
        return evaluate(self.settings, self.situation(bot))

    def tick(self, settings: AutopilotSettings | None = None, wait: bool = False) -> list[tuple[str, str]]:
        """Plan for every bot; start the plans in worker threads. Returns ``[(bot, rule)]``."""
        settings = settings or self.settings
        if not settings.enabled:
            return []
        started: list[tuple[str, str]] = []
        workers: list[threading.Thread] = []
        for bot in list(self._bots()):
            if bot is None or not bot.connected:
                continue
            memory = self.memory(bot)
            if memory.in_flight or memory.paused:
                continue
            situation = self.situation(bot)
            plan = evaluate(settings, situation)
            if plan is None:
                continue
            memory.in_flight = True
            memory.cooldowns[plan.rule] = situation.now + plan.cooldown
            worker = threading.Thread(target=self._run_worker, args=(bot, plan, memory),
                                      name=f"autopilot-{bot.username}", daemon=True)
            worker.start()
            workers.append(worker)
            started.append((bot.username, plan.rule))
        if wait:
            for worker in workers:
                worker.join(timeout=60.0)
        return started

    def _run_worker(self, bot: MinecraftBot, plan: Plan, memory: BotAutopilot) -> None:
        try:
            self.run(bot, plan, memory)
        finally:
            memory.in_flight = False

    def run(self, bot: MinecraftBot, plan: Plan, memory: BotAutopilot | None = None) -> tuple[bool, str]:
        """Carry one plan out now."""
        memory = memory or self.memory(bot)
        if plan.interrupt and bot.busy:
            bot.cancel_actions()
            time.sleep(0.3)
            bot.clear_cancel()
        if self.settings.verbose or plan.emergency:
            self.log.add(f"[{bot.username}] autopilot: {plan.rule} ({plan.reason})", "info", "autopilot")
        try:
            if plan.action.startswith("skill:"):
                ok, message = bot.skill(plan.action.split(":", 1)[1], plan.params)
            else:
                ok = bool(getattr(bot, plan.action)(**plan.params))
                message = "done" if ok else (bot.last_failure or "failed")
        except Exception as exc:  # a reflex never takes the bot down
            ok, message = False, str(exc)
        now = time.monotonic()
        with self._lock:
            memory.last_rule, memory.last_message, memory.last_at = plan.rule, message, time.time()
            if ok:
                memory.counts[plan.rule] = memory.counts.get(plan.rule, 0) + 1
                memory.failures.pop(plan.rule, None)
                self.actions += 1
            else:
                memory.failures[plan.rule] = now
                if plan.fail_cooldown:
                    memory.cooldowns[plan.rule] = now + plan.fail_cooldown
        return ok, message

    def pause(self, bot: MinecraftBot, paused: bool = True) -> None:
        self.memory(bot).paused = paused

    def status(self, bots: Sequence[MinecraftBot] | None = None) -> dict[str, Any]:
        bots = list(self._bots() if bots is None else bots)
        settings = self.settings
        rules = []
        for rule in RULES:
            flag = _RULE_FLAGS.get(rule)
            on = bool(getattr(settings, flag)) if flag else settings.idle_job != "none"
            rules.append({"name": rule, "info": RULE_INFO[rule], "enabled": on, "emergency": rule in EMERGENCY_RULES,
                          "setting": flag or "idle_job"})
        return {
            "enabled": settings.enabled,
            "running": self.running,
            "actions": self.actions,
            "idle_job": settings.idle_job,
            "rules": rules,
            "bots": {bot.username: self.memory(bot).to_dict() for bot in bots if bot is not None},
        }


_RULE_FLAGS: dict[str, str] = {
    "respawn": "auto_respawn", "surface": "auto_surface", "extinguish": "auto_extinguish", "totem": "auto_totem",
    "flee": "auto_flee", "eat": "auto_eat", "armor": "auto_armor", "weapon": "auto_weapon", "defend": "auto_defend",
    "sleep": "auto_sleep", "shelter": "auto_shelter", "torch": "auto_torch", "collect": "auto_collect",
    "tidy": "auto_tidy", "home": "auto_home",
}


def _home_of(bot: MinecraftBot) -> dict[str, Any] | None:
    from pymc_bot import skills

    try:
        return skills.places_for(bot).get(skills._server_key(bot), bot.username, "home")  # noqa: SLF001
    except Exception:  # pragma: no cover - a broken places file only disables the rule
        return None
