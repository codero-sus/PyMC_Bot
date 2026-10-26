"""The AI player brain: an Ollama decision loop with a heuristic fallback.

Every ``decision_interval`` seconds the loop:
  1. snapshots the world (position, health, players, inventory, recent chat),
  2. asks Ollama for exactly one action as JSON,
  3. validates and executes it through :class:`pymc_bot.bot.MinecraftBot`.

If Ollama is disabled or unreachable the same loop runs a small deterministic
heuristic policy instead, so the bot always keeps playing.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pymc_bot import skills
from pymc_bot.bot import MinecraftBot
from pymc_bot.config import AppSettings, ConfigStore
from pymc_bot.cortex import CortexClient, client_from_settings
from pymc_bot.events import EventLog
from pymc_bot.inventory import HOTBAR_SIZE, Inventory, short
from pymc_bot.knowledge import KnowledgeBase, LookupResult, shared_knowledge
from pymc_bot.mind import (
    Mind,
    build_system_prompt,
    build_turn_prompt,
    humanize_message,
    typing_delay,
)
from pymc_bot.ollama import OllamaClient, OllamaError
from pymc_bot.train import list_models

ACTIONS: dict[str, str] = {
    "say": 'chat: {"action": "say", "message": "hello"}',
    "goto": 'walk: {"action": "goto", "x": 10, "y": 64, "z": -20}',
    "follow": 'follow a player: {"action": "follow", "player": "Steve"}',
    "wander": 'explore: {"action": "wander"}',
    "mine": 'dig: {"action": "mine", "block": "oak_log"}',
    "attack": 'hit a player: {"action": "attack", "player": "Steve"}',
    "jump": 'hop: {"action": "jump"}',
    "eat": 'eat food: {"action": "eat"}',
    "look_at_player": 'turn: {"action": "look_at_player", "player": "Steve"}',
    "stop": 'stand still: {"action": "stop"}',
    "wait": 'do nothing: {"action": "wait"}',
    # inventory - the way a player handles it (hotbar slots are 1-9)
    "select_slot": 'press a hotbar key: {"action": "select_slot", "slot": 3}',
    "hold_item": 'hold an item (pulls it into the hotbar): {"action": "hold_item", "item": "iron_pickaxe"}',
    "equip": 'put on armour / off-hand: {"action": "equip", "item": "iron_helmet"} or {"item": "shield", "to": "offhand"}',
    "unequip": 'take off: {"action": "unequip", "slot": "head"}',
    "wear_armor": 'put on the best armour you have: {"action": "wear_armor"}',
    "to_hotbar": 'backpack -> hotbar: {"action": "to_hotbar", "item": "cobblestone", "slot": 5}',
    "to_inventory": 'hotbar -> backpack: {"action": "to_inventory", "slot": 7}',
    "swap_hands": 'swap main hand and off-hand (F): {"action": "swap_hands"}',
    "organize": 'sort the hotbar (weapon, tools, blocks, torches, food): {"action": "organize"}',
    "drop": 'throw items away: {"action": "drop", "item": "dirt", "count": 16}',
    "give": 'hand items to a player: {"action": "give", "player": "Steve", "item": "bread", "count": 2}',
    # life
    "place": 'place a block next to you: {"action": "place", "block": "cobblestone"}',
    "use_item": 'right-click with the held item: {"action": "use_item", "seconds": 1}',
    "sleep": 'sleep in a nearby bed (night): {"action": "sleep"}',
    "collect": 'pick up dropped items nearby: {"action": "collect"}',
    "fight": 'fight the nearest hostile mob: {"action": "fight", "mob": "zombie"}',
    "flee": 'run away from danger: {"action": "flee"}',
    "emote": 'body language: {"action": "emote", "kind": "wave"} (wave, crouch, nod, shake, spin)',
    "look_around": 'glance around: {"action": "look_around"}',
    # knowledge
    "search": 'look something up (Minecraft Wiki): {"action": "search", "query": "how to tame a wolf"}',
}
# the 44 skills: chop_tree, mine_ore, craft, smelt, harvest, deposit, go_home, guard... (pymc_bot.skills)
ACTIONS.update(skills.action_examples())

SEARCH_ACTIONS = ("search", "lookup", "look_up", "web_search", "google", "wiki")

#: Mobs a sensible player does not pick a fight with.
AVOID_MOBS = {"warden", "enderman", "wither", "ender_dragon", "piglin_brute", "elder_guardian", "ravager", "evoker"}
GIVE_WORDS = re.compile(r"\b(give|gimme|toss|throw|spare|lend|can i (have|get)|could i (have|get)|need|share)\b", re.I)
GREET_WORDS = re.compile(r"\b(hi|hey|hello|yo|sup|hii+|heya)\b", re.I)
ITEM_WORDS = {"food": "food", "wood": "_log", "logs": "_log", "planks": "_planks", "blocks": "blocks", "torch": "torch",
              "torches": "torch", "sword": "sword", "pickaxe": "pickaxe", "pick": "pickaxe", "axe": "axe", "shovel": "shovel"}

MINING_TARGETS = ["oak_log", "stone", "coal_ore", "iron_ore", "birch_log", "dirt", "cobblestone"]


# ---------------------------------------------------------------------------
# decision parsing
# ---------------------------------------------------------------------------
@dataclass
class Decision:
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    source: str = "heuristic"
    raw: str = ""
    # LLM brains: a private one-sentence thought and the goal the model is working on.
    thought: str = ""
    goal: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "params": self.params,
            "source": self.source,
            "raw": self.raw,
            "thought": self.thought,
            "goal": self.goal,
        }


#: Appended to the turn prompt when a picture of the bot's view goes with it (the same words are
#: used in the Minecraftism VL training rows, so the model knows what the image is).
VISION_NOTE = "\n\nA screenshot of your current first-person view is attached; use what you see."

def parse_decision(text: str, source: str = "ollama") -> Decision:
    """Extract one action from a (possibly chatty) LLM response."""
    # reasoning models (Qwen3, DeepSeek-R1, Minecraftism Thinking...) may think out loud first
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    if re.search(r"</think>", cleaned, flags=re.I):
        # Qwen3 *Thinking* models start inside the think block: only the closing tag is printed
        cleaned = re.split(r"</think>", cleaned, flags=re.I)[-1].strip()
    if cleaned.lower().startswith("<think>"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object found in model output: {text[:200]!r}")
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON from model: {exc}; output was {text[:200]!r}") from exc
    if not isinstance(payload, dict):
        raise ValueError("model did not return a JSON object")

    action = payload.get("action") or payload.get("name") or payload.get("cmd") or payload.get("command")
    if not isinstance(action, str) or not action.strip():
        raise ValueError("model response has no 'action' field")
    action = action.strip().lower().removeprefix("minecraft:")

    params: dict[str, Any] = {}
    nested = payload.get("params") or payload.get("parameters") or payload.get("args")
    if isinstance(nested, dict):
        params.update(nested)
    for key, value in payload.items():
        if key in ("action", "name", "cmd", "command", "params", "parameters", "args"):
            continue
        params.setdefault(key, value)
    thought = next((str(params.pop(key)) for key in ("thought", "thinking", "reason", "reasoning") if key in params), "")
    goal = str(params.pop("goal", "") or "")
    return Decision(
        action=action, params=params, source=source, raw=text[:500], thought=thought.strip()[:240], goal=goal.strip()[:80]
    )


# ---------------------------------------------------------------------------
# heuristic policy (no LLM required)
# ---------------------------------------------------------------------------
class HeuristicPolicy:
    """Built-in brain (no LLM): looks after itself the way a player does, then plays.

    Priorities, like a person: get away when badly hurt, eat, deal with a mob that comes close,
    answer people (and hand over things they ask for), put on better armour, tidy the hotbar,
    pick up drops, sleep at night - and otherwise follow friends, mine, explore and fidget.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._step = 0
        self._greeted: set[str] = set()

    def decide(self, state: dict[str, Any], chat: list[str], allowed: dict[str, bool] | None = None) -> Decision:
        self._step += 1
        can = {"movement": True, "chat": True, "mining": True, "inventory": True, "building": False, "fighting": True}
        can.update(allowed or {})
        health = float(state.get("health") or 20)
        food = float(state.get("food") or 20)
        players = state.get("players") or []
        bot_name = str(state.get("username") or state.get("server", {}).get("username") or "PyMC_Bot")
        knows_items = bool(state.get("slots") or state.get("inventory"))
        inv = Inventory.from_snapshot(state)
        hostiles = sorted(
            (e for e in state.get("entities") or [] if e.get("hostile") and not e.get("player")),
            key=lambda e: float(e.get("dist") or 99),
        )
        threat = hostiles[0] if hostiles else None
        threat_dist = float(threat.get("dist") or 99) if threat else 99.0
        threat_kind = short(threat.get("type")) if threat else ""

        # 1. survive
        if threat and threat_dist <= 8 and health <= 8 and can["movement"]:
            return Decision("flee", {}, "heuristic")
        if health <= 8 or food <= 6:
            return Decision("eat", {}, "heuristic")
        if knows_items and food <= 14 and inv.best_food(food, health) is not None:
            return Decision("eat", {}, "heuristic")
        if threat and threat_dist <= 6:
            if threat_kind in AVOID_MOBS:
                if can["movement"] and threat_dist <= 5:
                    return Decision("flee", {}, "heuristic")
            elif can["fighting"] and health > 10:
                return Decision("fight", {"mob": threat_kind}, "heuristic")

        # 2. people
        mentioned = [line for line in chat if bot_name.lower() in line.lower()]
        if mentioned:
            line = mentioned[-1]
            speaker_match = re.match(r"\s*<([^>]+)>", line)
            speaker = speaker_match.group(1) if speaker_match else ""
            if knows_items and can["inventory"] and speaker and GIVE_WORDS.search(line):
                wanted = self._requested_item(line, inv)
                if wanted:
                    count = max(1, min(16, inv.count(wanted) // 2 or 1))
                    return Decision("give", {"player": speaker, "item": wanted, "count": count}, "heuristic")
            if can["chat"]:
                return Decision(
                    "say",
                    {"message": self._rng.choice(["on it!", "sure, coming", "hey", "yeah?", "one sec"])},
                    "heuristic",
                )

        # 3. look after the gear
        if knows_items and can["inventory"]:
            if inv.armor_upgrades():
                return Decision("wear_armor", {}, "heuristic")
            if self._step % 4 == 0 and inv.hotbar_is_messy(food, health):
                return Decision("organize", {}, "heuristic")
        drops = [d for d in state.get("ground_items") or [] if float(d.get("dist") or 99) <= 8]
        if drops and can["movement"] and self._step % 2 == 0:
            return Decision("collect", {}, "heuristic")
        if state.get("time_of_day") == "night" and not state.get("sleeping") and self._step % 6 == 0:
            return Decision("sleep", {}, "heuristic")

        # 4. friends
        if players:
            nearest = players[0]
            name = str(nearest.get("name") or "")
            if name and nearest.get("distance", 99) < 6 and name not in self._greeted:
                self._greeted.add(name)
                return Decision("emote", {"kind": self._rng.choice(["wave", "crouch"])}, "heuristic")
            if nearest.get("distance", 99) < 12 and self._step % 3 == 0:
                return Decision("follow", {"player": nearest["name"]}, "heuristic")

        # 5. play
        if self._step % 5 == 0:
            return Decision("mine", {"block": self._rng.choice(MINING_TARGETS)}, "heuristic")
        if self._step % 7 == 0:
            return Decision("say", {"message": self._rng.choice(["just exploring", "nice server", "anyone need wood?"])}, "heuristic")
        if state.get("time_of_day") == "night" and self._step % 4 == 0:
            return Decision("mine", {"block": "coal_ore"}, "heuristic")
        roll = self._rng.random()
        if roll < 0.12:
            return Decision("look_around", {}, "heuristic")
        if roll < 0.2 and knows_items and can["inventory"]:
            filled = [i for i in range(HOTBAR_SIZE) if inv.slots[i] and i != inv.selected]
            if filled:  # players scroll through the hotbar while they think
                return Decision("select_slot", {"slot": self._rng.choice(filled) + 1}, "heuristic")
        return Decision("wander", {}, "heuristic")

    @staticmethod
    def _requested_item(line: str, inv: Inventory) -> str | None:
        """The item a chat line asks for, if the bot has it ("can i have some bread", "gimme wood")."""
        text = line.split(">", 1)[-1].lower()
        for _, stack in sorted(inv.items(), key=lambda item: -len(item[1].name)):
            if stack.name in text.replace(" ", "_") or stack.name.replace("_", " ") in text:
                return stack.name
        for word, meaning in ITEM_WORDS.items():
            if re.search(rf"\b{word}\b", text):
                if meaning.startswith("_"):
                    match = next((st.name for _, st in inv.items() if st.name.endswith(meaning)), None)
                else:
                    index = inv.resolve(meaning, 12.0)
                    match = inv.slots[index].name if index is not None else None
                if match:
                    return match
        return None


# ---------------------------------------------------------------------------
# agent loop
# ---------------------------------------------------------------------------
class AgentLoop:
    """Background thread that turns world state into actions."""

    def __init__(
        self,
        bot: MinecraftBot,
        store: ConfigStore,
        log: EventLog | None = None,
        ollama: OllamaClient | None = None,
        heuristic: HeuristicPolicy | None = None,
        cortex: CortexClient | None = None,
        prefer_ollama: bool = True,
        brain: str | None = None,
    ) -> None:
        self.bot = bot
        self.store = store
        self.log = log or EventLog()
        # Fleet bots default to heuristics so 20 extra players do not hammer one GPU.
        self.prefer_ollama = prefer_ollama
        # Per-bot override of ``agent.mode`` (the fleet uses it so one bot can run the
        # model trained on playtime while the others stay on heuristics / Ollama).
        self._brain_override = brain if brain and brain != "auto" else None
        self._trained: Any | None = None
        self._trained_key: tuple[Any, ...] | None = None
        self._ollama = ollama
        self._cortex = cortex
        self._heuristic = heuristic or HeuristicPolicy()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._last_decision: dict[str, Any] | None = None
        self._decisions = 0
        self._record_lock = threading.Lock()
        self._last_exchange: tuple[str, str] | None = None
        self._vision_turn = 0
        self._vision_error = ""
        self._errors = 0
        self._last_error: str | None = None
        # Short-term memory for the LLM brains (goal, recent actions, things already said).
        self.mind = Mind(store.settings.agent.memory_turns)
        self._rng = random.Random()

    # ------------------------------------------------------------- lifecycle
    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> bool:
        with self._lock:
            if self.running:
                return False
            self.bot.clear_cancel()
            self._stop_event.clear()
            self.bot.llm_chat = self.uses_llm()
            self._thread = threading.Thread(target=self._run, name="agent-loop", daemon=True)
            self._thread.start()
        self.log.add("AI agent loop started.", "success", "agent")
        return True

    def stop(self, wait: bool = True) -> bool:
        with self._lock:
            if not self.running:
                return False
            self._stop_event.set()
            thread = self._thread
        # Abort any long movement action (walking, following, mining) so the
        # loop can exit promptly instead of finishing its 30s action.
        self.bot.cancel_actions()
        self.bot.llm_chat = False
        if wait and thread is not None:
            thread.join(timeout=10.0)
        self.bot.clear_cancel()
        self.log.add("AI agent loop stopped.", "info", "agent")
        return True

    # ----------------------------------------------------------------- brain
    def brain(self, settings: AppSettings | None = None) -> str:
        """Which brain this loop uses right now."""
        if self._brain_override:
            return self._brain_override
        settings = settings or self._settings()
        return settings.agent.mode

    def policy(self, settings: AppSettings | None = None) -> Any | None:
        """The trained playtime policy, loaded (and cached) on demand.

        Returns ``None`` when no checkpoint exists yet, so the agent can fall back to
        the heuristic instead of refusing to play.
        """
        settings = settings or self._settings()
        training = settings.training
        run = (training.active_run or "").strip()
        key = (training.models_dir, run, training.temperature, training.step_seconds)
        with self._lock:
            if self._trained_key == key:
                return self._trained
        policy = None
        try:
            from pymc_bot.local_model import TrainedPolicy

            if not run:
                runs = [
                    card
                    for card in list_models(training.models_dir)
                    if card.get("checkpoint_exists")
                ]
                if not runs:
                    raise FileNotFoundError(
                        f"no trained model in {training.models_dir} - train one with "
                        "'python -m pymc_bot train --dataset " + training.dataset + "'"
                    )
                run = str(runs[0]["run"])
            policy = TrainedPolicy.load(
                run,
                training.models_dir,
                temperature=training.temperature,
                step_seconds=training.step_seconds,
                include_blocks=bool(training.include_blocks),
            )
            self.log.add(
                f"Loaded trained player model '{policy.run.run}' "
                f"({policy.run.engine}, step {policy.run.card.get('step')}, "
                f"{policy.run.card.get('params')} params) from {policy.run.checkpoint}",
                "success",
                "agent",
            )
        except Exception as exc:
            self.log.add(f"Trained brain unavailable ({exc}); using built-in behaviour.", "warn", "agent")
            policy = None
        with self._lock:
            self._trained = policy
            self._trained_key = key
        return policy

    def allowed_trained_actions(self, settings: AppSettings | None = None) -> set[str]:
        """Low level actions the trained policy may pick, given the permission gates.

        Looking around and idling are always allowed; everything else follows the same
        switches as the high level brains, so a model trained on playtime cannot attack
        or mine when the config forbids it.
        """
        settings = settings or self._settings()
        agent = settings.agent
        allowed = {"look", "none"}
        if agent.allow_movement:
            allowed |= {"forward", "back", "left", "right", "jump", "sneak", "move"}
        if agent.allow_mining:
            allowed |= {"use", "hold"}
        if agent.allow_attacking:
            allowed.add("attack")
        return allowed

    def _decide(self, settings: AppSettings, state: dict[str, Any]) -> Decision:
        """Pick the next decision with the configured brain."""
        brain = self.brain(settings)
        if brain == "trained":
            policy = self.policy(settings)
            if policy is None:
                return self._heuristic.decide(state, self.bot.recent_chat())
            return policy.decide(state, allowed=self.allowed_trained_actions(settings))
        if brain == "heuristic":
            return self._heuristic.decide(state, self.bot.recent_chat())
        if brain == "ollama":
            return self._ask_ollama(settings, state)
        if brain == "cortex":
            return self._ask_cortex(settings, state)
        # auto: the first enabled LLM (Ollama, then Cortex), otherwise the built-in behaviour.
        resolved = self.resolved_brain(settings)
        if resolved == "ollama":
            return self._ask_ollama(settings, state)
        if resolved == "cortex":
            return self._ask_cortex(settings, state)
        return self._heuristic.decide(state, self.bot.recent_chat())

    def uses_llm(self, settings: AppSettings | None = None) -> bool:
        return self.resolved_brain(settings) in ("ollama", "cortex")

    def resolved_brain(self, settings: AppSettings | None = None) -> str:
        """The brain that really decides: ``auto`` resolved to ollama/cortex/heuristic."""
        settings = settings or self._settings()
        brain = self.brain(settings)
        if brain != "auto":
            return brain
        if self.prefer_ollama and settings.ollama.enabled:
            return "ollama"
        if self.prefer_ollama and settings.cortex.enabled:
            return "cortex"
        return "heuristic"

    # ----------------------------------------------------------------- stats
    @property
    def status(self) -> dict[str, Any]:
        with self._lock:
            trained = self._trained
        return {
            "running": self.running,
            "decisions": self._decisions,
            "errors": self._errors,
            "last_error": self._last_error,
            "last_decision": self._last_decision,
            "brain": self.brain(),
            "resolved_brain": self.resolved_brain(),
            "trained": trained.describe() if trained is not None else None,
            "persona": self._settings().agent.persona,
            "memory": self.mind.snapshot(),
        }

    # ------------------------------------------------------------ main logic
    def _settings(self) -> AppSettings:
        return self.store.settings

    def _ollama_client(self, settings: AppSettings) -> OllamaClient:
        if self._ollama is None:
            self._ollama = OllamaClient(
                base_url=settings.ollama.base_url,
                model=settings.ollama.model,
                temperature=settings.ollama.temperature,
                timeout=settings.ollama.request_timeout,
            )
        self._ollama.base_url = settings.ollama.base_url.rstrip("/")
        self._ollama.model = settings.ollama.model
        self._ollama.temperature = settings.ollama.temperature
        self._ollama.timeout = settings.ollama.request_timeout
        return self._ollama

    def _cortex_client(self, settings: AppSettings) -> CortexClient:
        cfg = settings.cortex
        if self._cortex is None:
            self._cortex = client_from_settings(cfg)
        # Follow live config edits from the panel without rebuilding the HTTP client.
        self._cortex.base_url = cfg.base_url.rstrip("/")
        self._cortex.api_path = cfg.api_path
        self._cortex.model = cfg.model
        self._cortex.api_key_env = cfg.api_key_env
        self._cortex.temperature = cfg.temperature
        self._cortex.max_tokens = cfg.max_tokens
        self._cortex.json_mode = cfg.json_mode
        self._cortex.timeout = cfg.request_timeout
        return self._cortex

    def decision_interval(self, settings: AppSettings | None = None) -> float:
        """Seconds between decisions for the brain in charge."""
        settings = settings or self._settings()
        brain = self.resolved_brain(settings)
        # The trained policy plays at the recording cadence (a few bursts a second),
        # the high level brains think once per decision_interval.
        if brain == "trained":
            return max(0.15, settings.training.step_seconds)
        if brain == "cortex":
            return settings.cortex.decision_interval
        return settings.ollama.decision_interval

    def _run(self) -> None:
        failures = 0
        while not self._stop_event.is_set():
            settings = self._settings()
            interval = self.decision_interval(settings)
            if not self.bot.connected:
                if self._wait(1.0):
                    break
                continue
            state = self.bot.snapshot()
            decision: Decision | None = None
            try:
                decision = self._decide(settings, state)
            except Exception as exc:
                failures += 1
                self._errors += 1
                self._last_error = str(exc)
                self.log.add(f"AI decision failed ({exc}); falling back to built-in behaviour.", "warn", "agent")
                decision = self._heuristic.decide(state, self.bot.recent_chat(), self.allowed(settings))
            if decision is None:
                decision = self._heuristic.decide(state, self.bot.recent_chat(), self.allowed(settings))
            failures = 0
            ok, detail = self._act(decision)
            if self.uses_llm(settings):
                # people don't act on a metronome: +-25% around the configured interval
                interval *= self._rng.uniform(0.75, 1.25)
            if self._wait(interval):
                break
        self.log.add("AI agent loop finished.", "debug", "agent")

    def _wait(self, seconds: float) -> bool:
        """Sleep in small slices. Returns True when a stop was requested."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self._stop_event.is_set():
                return True
            time.sleep(0.05)
        return self._stop_event.is_set()

    def vision_frame(self, settings: AppSettings) -> list[str] | None:
        """A picture of the bot's view for this turn (every ``vision_every``-th turn), or None.

        Problems (vision packages missing, renderer failure) are logged once and the turn goes
        ahead as text only.
        """
        agent = settings.agent
        if not agent.vision:
            return None
        self._vision_turn += 1
        if (self._vision_turn - 1) % agent.vision_every:
            return None
        backend = self.bot.backend
        if backend is None:
            return None
        try:
            frame = backend.view_image(agent.vision_width)
        except Exception as exc:  # noqa: BLE001 - any renderer problem means "no picture"
            message = str(exc)
            if message != self._vision_error:
                self._vision_error = message
                self.log.add(f"👁 no picture this turn: {message}", "warn", "agent")
            return None
        self._vision_error = ""
        image = str(frame.get("image") or "")
        return [image] if image else None

    def _ask_ollama(self, settings: AppSettings, state: dict[str, Any]) -> Decision:
        client = self._ollama_client(settings)
        system = self.system_prompt(settings, settings.ollama.system_prompt)
        images = self.vision_frame(settings)

        def ask(prompt: str) -> str:
            if images:
                return client.chat([{"role": "user", "content": prompt + VISION_NOTE}], system=system, images=images)
            try:
                return client.chat([{"role": "user", "content": prompt}], system=system)
            except OllamaError:
                # Some models/servers only implement /api/generate.
                return client.generate(prompt, system=system)

        return self._research_and_decide(settings, state, ask, lambda: f"ollama:{settings.ollama.model}")

    def _ask_cortex(self, settings: AppSettings, state: dict[str, Any]) -> Decision:
        """One decision from the model hosted by Cortex LLMHoster."""
        client = self._cortex_client(settings)
        system = self.system_prompt(settings, settings.cortex.system_prompt)
        images = self.vision_frame(settings)
        return self._research_and_decide(
            settings, state,
            lambda prompt: client.decide(prompt + VISION_NOTE if images else prompt, system=system, images=images),
            lambda: f"cortex:{client.last_model or settings.cortex.model or 'default'}",
        )

    # ------------------------------------------------------------- knowledge
    @property
    def knowledge(self) -> KnowledgeBase:
        """The shared web-lookup cache (one per process, see pymc_bot.knowledge)."""
        return shared_knowledge(self._settings().search, self.store.path.parent)

    def can_search(self, settings: AppSettings) -> bool:
        return settings.agent.allow_web_search and settings.search.enabled

    def look_up(self, query: str) -> LookupResult:
        """Run one lookup, remember it for the prompt and log it."""
        result = self.knowledge.lookup(query)
        self.mind.note_lookup(result.query, result.summary(self._settings().search.max_chars), result.ok)
        if result.ok:
            where = f" ({result.title}, {result.source}{', cached' if result.cached else ''})"
            self.log.add(f"🔎 looked up \"{result.query}\"{where}: {result.summary(160)}", "ai", "agent")
        else:
            self.log.add(f"🔎 lookup \"{result.query}\" found nothing: {result.error}", "info", "agent")
        return result

    def _research_and_decide(self, settings: AppSettings, state: dict[str, Any], ask: Any, source: Any) -> Decision:
        """Ask the LLM; if it wants to look something up first, search and ask again with the answer.

        ``search.max_per_turn`` lookups may happen inside one turn, so the model can read about a
        mob or mechanic and act on it right away instead of guessing.
        """
        raw_ask = ask

        def ask(prompt: str) -> str:  # remember the exchange so record_turns can save it
            reply = raw_ask(prompt)
            self._last_exchange = (prompt, reply)
            return reply

        self._last_exchange = None
        decision = parse_decision(ask(self._build_prompt(settings, state)), source=source())
        rounds = settings.search.max_per_turn if self.can_search(settings) else 0
        for _ in range(rounds):
            if decision.action not in SEARCH_ACTIONS:
                break
            query = str(decision.params.get("query") or decision.params.get("q") or decision.params.get("topic") or "")
            if not query.strip():
                break
            if decision.thought:
                self.log.add(f"💭 {decision.thought}", "ai", "agent")
            self.look_up(query)
            prompt = self._build_prompt(settings, state) + (
                "\n\nYou just looked that up (see WHAT YOU LOOKED UP). Now choose an action that uses it - "
                "do not search for the same thing again."
            )
            decision = parse_decision(ask(prompt), source=source())
        return decision

    def _prefetch_unknown_mobs(self, settings: AppSettings, state: dict[str, Any]) -> None:
        if not self.can_search(settings) or not settings.search.auto_lookup_mobs:
            return
        for entity in (state.get("entities") or [])[:12]:
            if not entity.get("player") and float(entity.get("dist") or 99) <= 24:
                self.knowledge.prefetch_mob(str(entity.get("type") or ""))

    def allowed(self, settings: AppSettings) -> dict[str, bool]:
        agent = settings.agent
        return {
            "movement": agent.allow_movement,
            "chat": agent.allow_chat,
            "mining": agent.allow_mining,
            "attacking": agent.allow_attacking,
            "inventory": agent.allow_inventory,
            "building": agent.allow_building,
            "fighting": agent.allow_mob_fighting,
            "search": agent.allow_web_search and settings.search.enabled,
        }

    def system_prompt(self, settings: AppSettings, custom: str = "") -> str:
        """The system prompt the LLM gets: the owner's own text, or the detailed built-in one."""
        if custom and custom.strip():
            return custom
        agent = settings.agent
        if agent.compact_prompt:
            from pymc_bot.llm_dataset import compact_system_prompt

            return compact_system_prompt(self.bot.username, agent.persona, self.allowed(settings), agent.persona_notes,
                                         min(agent.max_chat_length, 160))
        return build_system_prompt(
            name=self.bot.username,
            persona=agent.persona,
            notes=agent.persona_notes,
            allowed=self.allowed(settings),
            max_chat_length=min(agent.max_chat_length, 160),
        )

    def _build_prompt(self, settings: AppSettings, state: dict[str, Any]) -> str:
        """This turn's situation as a person would see it (plus the exact JSON at the end)."""
        self._prefetch_unknown_mobs(settings, state)
        kinds: list[str] = []
        for entity in state.get("entities") or []:
            kind = str(entity.get("type") or "").removeprefix("minecraft:")
            if kind and kind not in kinds and not entity.get("player") and float(entity.get("dist") or 99) <= 24:
                kinds.append(kind)
        facts = self.knowledge.facts_for(kinds[:6], server_version=settings.minecraft.version)
        return build_turn_prompt(state, self.bot.username, self.mind, self.bot.chat_history, self.allowed(settings),
                                 knowledge=facts)

    # -------------------------------------------------------------- executor
    def execute(self, decision: Decision, enforce_permissions: bool = True) -> tuple[bool, str]:
        """Run one decision against the bot.

        Permission flags from the config gate the AI's own choices.  Manual
        actions sent by a human through the web panel pass
        ``enforce_permissions=False`` (attacking stays gated either way).
        """
        settings = self._settings()
        agent = settings.agent
        action = decision.action
        params = decision.params or {}

        # A trained playtime model predicts low level controls, so it executes itself.
        if str(decision.source or "").startswith("trained:"):
            policy = self.policy(settings)
            if policy is None:
                return False, "the trained model is not available any more"
            if not agent.allow_movement and action in (
                "forward", "back", "left", "right", "jump", "sneak", "move", "look"
            ):
                return False, "movement is disabled in settings"
            if not agent.allow_attacking and action == "attack":
                return False, "attacking is disabled in settings"
            if not agent.allow_mining and action in ("use", "hold"):
                return False, "mining is disabled in settings"
            return policy.act(self.bot, decision)

        def blocked(flag: str, allowed: bool) -> bool:
            if allowed:
                return False
            return enforce_permissions or flag == "attacking"

        if action == "say":
            if blocked("chat", agent.allow_chat):
                return False, "chat is disabled in settings"
            message = str(params.get("message") or params.get("text") or "").strip()
            llm = str(decision.source or "").startswith(("ollama", "cortex"))
            if llm:
                message = humanize_message(message, self.bot.username, agent.max_chat_length)
            if not message:
                return False, "say had no message"
            if llm and self.mind.already_said(message):
                return False, "skipped: already said that"
            if llm and agent.human_chat:
                since = time.time() - self.mind.last_said_at
                if since < agent.chat_cooldown and not self._addressed_recently():
                    return False, f"held back chat ({since:.0f}s since the last message)"
                # read, think, type - like a person at the keyboard
                if self._wait(typing_delay(message, self._rng)):
                    return False, "stopped before sending"
            params["message"] = message
            return (True, f"said: {message}") if self.bot.say(message) else (False, "could not send chat")

        if action in ("goto", "walk", "move"):
            if blocked("movement", agent.allow_movement):
                return False, "movement is disabled in settings"
            try:
                x = float(params.get("x"))
                z = float(params.get("z"))
            except (TypeError, ValueError):
                return False, "goto needs numeric x/z"
            y = params.get("y")
            y = float(y) if isinstance(y, (int, float)) else None
            x = max(-30000.0, min(30000.0, x))
            z = max(-30000.0, min(30000.0, z))
            ok = self.bot.walk_to(x, z, y)
            return ok, f"{'walked' if ok else 'could not walk'} to ({x:.0f}, {z:.0f})"

        if action in ("follow", "follow_player"):
            if blocked("movement", agent.allow_movement):
                return False, "movement is disabled in settings"
            player = str(params.get("player") or params.get("name") or "")
            if not player:
                return False, "follow needs a player name"
            ok = self.bot.follow(player, seconds=min(30.0, max(6.0, self.decision_interval(settings) * 3)))
            return ok, f"{'followed' if ok else 'could not follow'} {player}"

        if action in ("wander", "explore"):
            if blocked("movement", agent.allow_movement):
                return False, "movement is disabled in settings"
            ok = self.bot.wander()
            return ok, "explored the area" if ok else "had nowhere to explore"

        if action in ("mine", "dig"):
            if blocked("mining", agent.allow_mining):
                return False, "mining is disabled in settings"
            block = str(params.get("block") or params.get("target") or "stone")
            ok = self.bot.mine(block)
            return ok, f"{'mined' if ok else 'could not mine'} {block}"

        if action in ("attack", "hit"):
            if blocked("attacking", agent.allow_attacking):
                return False, "attacking is disabled in settings"
            player = str(params.get("player") or params.get("target") or "")
            if not player:
                return False, "attack needs a player name"
            ok = self.bot.attack(player)
            return ok, f"{'attacked' if ok else 'could not attack'} {player}"

        if action == "look_at_player":
            player = str(params.get("player") or params.get("name") or "")
            ok = self.bot.look_at_player(player)
            return ok, f"looked at {player}" if ok else f"{player} is not visible"

        if action in ("jump", "hop"):
            ok = self.bot.jump()
            return ok, "jumped" if ok else "could not jump"

        if action == "eat":
            return True, "ate" if self.bot.eat() else "had nothing to eat"

        if action in ("stop", "idle"):
            self.bot.stop_moving()
            return True, "stopped moving"

        if action in SEARCH_ACTIONS:
            if blocked("search", self.can_search(settings)):
                return False, "web search is disabled in settings"
            query = str(params.get("query") or params.get("q") or params.get("topic") or "").strip()
            if not query:
                return False, "search needs a query"
            result = self.look_up(query)
            return (True, f"looked up {query}: {result.summary(240)}") if result.ok else (False, f"found nothing for {query}: {result.error}")

        life = self._execute_life(action, params, agent, blocked)
        if life is not None:
            return life

        skill = skills.get(action)
        if skill is not None:
            return self._execute_skill(skill, params, agent, blocked)

        if action in ("wait", "noop", "none", "nothing"):
            return True, "waited"

        return False, f"unsupported action: {action}"

    def _execute_life(self, action: str, params: dict[str, Any], agent: Any, blocked: Any) -> tuple[bool, str] | None:
        """Inventory, building, mob fighting and the other human things. None = not one of these."""
        bot = self.bot

        def why(fallback: str) -> str:
            return bot.last_failure or fallback

        def item_of(*keys: str) -> str:
            return str(next((params[k] for k in keys if params.get(k)), "")).strip()

        def number(key: str, default: Any = None) -> Any:
            value = params.get(key, default)
            try:
                return int(value) if value is not None and value != "" else default
            except (TypeError, ValueError):
                return default

        inventory_actions = {
            "select_slot", "hotbar", "hold_item", "hold", "equip", "wear", "unequip", "take_off", "wear_armor",
            "armor", "to_hotbar", "to_inventory", "swap_hands", "organize", "sort_hotbar", "drop", "toss", "give",
        }
        if action in inventory_actions and blocked("inventory", agent.allow_inventory):
            return False, "inventory actions are disabled in settings"
        if action in ("place", "place_block", "build", "use_item", "use") and blocked("building", agent.allow_building):
            return False, "building is disabled in settings"
        if action in ("fight", "attack_mob", "kill") and blocked("fighting", agent.allow_mob_fighting):
            return False, "fighting mobs is disabled in settings"
        if action in ("flee", "run", "collect", "pickup", "pick_up", "sleep") and blocked("movement", agent.allow_movement):
            return False, "movement is disabled in settings"

        if action in ("select_slot", "hotbar"):
            slot = number("slot")
            if slot is None:
                return False, "select_slot needs a slot (1-9)"
            ok = bot.select_slot(slot)
            return ok, f"switched to hotbar slot {slot}" if ok else why("could not switch slots")
        if action in ("hold_item", "hold"):
            item = item_of("item", "name", "block")
            if not item:
                return False, "hold_item needs an item"
            ok = bot.hold(item)
            return ok, f"holding {item}" if ok else why(f"could not hold {item}")
        if action in ("equip", "wear"):
            item = item_of("item", "name")
            if not item:
                return False, "equip needs an item"
            ok = bot.equip(item, str(params.get("to") or params.get("destination") or "") or None)
            return ok, f"equipped {item}" if ok else why(f"could not equip {item}")
        if action in ("unequip", "take_off"):
            slot = item_of("slot", "from", "item")
            ok = bot.unequip(slot)
            return ok, f"took off {slot}" if ok else why(f"could not take off {slot}")
        if action in ("wear_armor", "armor"):
            ok = bot.wear_best_armor()
            return ok, "put on better armour" if ok else why("no better armour")
        if action == "to_hotbar":
            item = item_of("item", "name", "block")
            if not item:
                return False, "to_hotbar needs an item"
            ok = bot.to_hotbar(item, number("slot"))
            return ok, f"moved {item} to the hotbar" if ok else why(f"could not move {item}")
        if action == "to_inventory":
            ok = bot.to_inventory(number("slot"), item_of("item", "name") or None)
            return ok, "moved it to the backpack" if ok else why("could not move it")
        if action == "swap_hands":
            ok = bot.swap_hands()
            return ok, "swapped hands" if ok else why("could not swap hands")
        if action in ("organize", "sort_hotbar"):
            ok = bot.organize_hotbar()
            return ok, "sorted the hotbar" if ok else why("the hotbar is already tidy")
        if action in ("drop", "toss"):
            item = item_of("item", "name")
            if not item:
                return False, "drop needs an item"
            ok = bot.drop(item, number("count"))
            return ok, f"dropped {item}" if ok else why(f"could not drop {item}")
        if action == "give":
            player, item = item_of("player", "to"), item_of("item", "name")
            if not player or not item:
                return False, "give needs a player and an item"
            target = bot.find_player(player)
            if target and target.get("distance", 99) > 3 and blocked("movement", agent.allow_movement):
                return False, f"{player} is too far and movement is disabled"
            ok = bot.give(player, item, number("count"))
            return ok, f"gave {item} to {player}" if ok else why(f"could not give {item} to {player}")
        if action in ("place", "place_block", "build"):
            block = item_of("block", "item", "name") or "blocks"
            ok = bot.place_block(block)
            return ok, f"placed {block}" if ok else why(f"could not place {block}")
        if action in ("use_item", "use"):
            seconds = params.get("seconds", 0.3)
            try:
                seconds = float(seconds)
            except (TypeError, ValueError):
                seconds = 0.3
            ok = bot.use_item(seconds)
            return ok, "used the item" if ok else why("could not use the item")
        if action == "sleep":
            ok = bot.sleep_in_bed()
            return ok, "went to bed" if ok else why("could not sleep")
        if action in ("wake", "wake_up"):
            ok = bot.wake_up()
            return ok, "got up" if ok else why("could not get up")
        if action in ("collect", "pickup", "pick_up"):
            ok = bot.collect_items()
            return ok, "picked up items" if ok else why("nothing to pick up")
        if action in ("fight", "attack_mob", "kill"):
            mob = item_of("mob", "target", "type") or None
            if mob and short(mob) in AVOID_MOBS and not params.get("force"):
                return False, f"not picking a fight with a {short(mob)}"
            ok = bot.fight(mob)
            return ok, f"fought {mob or 'a mob'}" if ok else why("nothing to fight")
        if action in ("flee", "run"):
            ok = bot.flee()
            return ok, "ran away" if ok else why("nothing to run from")
        if action == "emote":
            kind = str(params.get("kind") or params.get("type") or "wave")
            ok = bot.emote(kind)
            return ok, f"did a {kind}" if ok else "could not emote"
        if action == "look_around":
            ok = bot.look_around()
            return ok, "looked around" if ok else "could not look around"
        return None

    def _execute_skill(self, skill: skills.Skill, params: dict[str, Any], agent: Any, blocked: Any) -> tuple[bool, str]:
        """One of the bigger jobs (chop_tree, craft, harvest, go_home...), gated by the same switches."""
        flag = {"fighting": "fighting", "movement": "movement", "mining": "mining", "building": "building",
                "inventory": "inventory", "chat": "chat"}.get(skill.permission)
        if flag is not None and blocked(flag, skills.allowed(skill, agent)):
            return False, f"{skill.name} needs {flag} to be allowed in settings"
        target = short(str(params.get("mob") or params.get("animal") or params.get("target") or ""))
        if skill.name in ("shoot", "hunt") and target in AVOID_MOBS and not params.get("force"):
            return False, f"not picking a fight with a {target}"
        return self.bot.skill(skill.name, params)

    def _addressed_recently(self, window: float = 60.0) -> bool:
        """Did someone talk to this bot in the last ``window`` seconds (replies skip the cooldown)?"""
        from pymc_bot.mind import mentions

        name = self.bot.username
        now = time.time()
        for entry in reversed(self.bot.chat_history):
            if now - float(entry.get("time", 0)) > window:
                break
            who = str(entry.get("username", ""))
            if who.lower() != name.lower() and mentions(str(entry.get("message", "")), name):
                return True
        return False

    def run_once(self) -> tuple[bool, str]:
        """Decide and execute a single action (used by ``POST /api/step``)."""
        settings = self._settings()
        state = self.bot.snapshot()
        try:
            decision = self._decide(settings, state)
        except Exception as exc:
            # Same bookkeeping as the loop, so the panel can show why the brain failed.
            self._errors += 1
            self._last_error = str(exc)
            self.log.add(f"AI decision failed ({exc}); using built-in behaviour.", "warn", "agent")
            decision = self._heuristic.decide(state, self.bot.recent_chat(), self.allowed(settings))
        return self._act(decision)

    def turns_path(self, settings: AppSettings | None = None) -> Path:
        settings = settings or self._settings()
        path = Path(settings.agent.turns_file).expanduser()
        return path if path.is_absolute() else self.store.path.parent / path

    def _record_turn(self, decision: Decision, ok: bool, detail: str) -> None:
        """Append an LLM turn to ``agent.turns_file`` (training data for a Minecraft model)."""
        exchange = getattr(self, "_last_exchange", None)
        self._last_exchange = None
        settings = self._settings()
        if not settings.agent.record_turns or exchange is None or decision.source in ("heuristic", "trained"):
            return
        agent = settings.agent
        row = {
            "ts": round(time.time(), 1), "source": decision.source, "name": self.bot.username, "persona": agent.persona,
            "notes": agent.persona_notes, "allowed": self.allowed(settings), "max_chat_length": min(agent.max_chat_length, 160),
            "prompt": exchange[0], "reply": exchange[1], "decision": decision.to_dict(), "ok": bool(ok), "detail": str(detail)[:200],
        }
        try:
            path = self.turns_path(settings)
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._record_lock, path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            self.log.add(f"could not record the turn: {exc}", "warn", "agent")

    def _act(self, decision: Decision) -> tuple[bool, str]:
        """Bookkeeping + memory around :meth:`execute` for one decision of the loop."""
        self._decisions += 1
        self._last_decision = {**decision.to_dict(), "ts": time.time()}
        self.mind.resize(self._settings().agent.memory_turns)
        self.mind.note_decision(decision)
        if decision.thought:
            self.log.add(f"💭 {decision.thought}" + (f"  [goal: {decision.goal}]" if decision.goal else ""), "ai", "agent")
        self.log.add(f"AI chose: {decision.action} {decision.params or ''}".rstrip(), "ai", "agent")
        ok, detail = self.execute(decision)
        self.mind.note_result(decision, ok, detail)
        self._record_turn(decision, ok, detail)
        self.log.add(f"{'✓' if ok else '✗'} {detail}", "info" if ok else "warn", "agent")
        return ok, detail
