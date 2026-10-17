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
from typing import Any

from pymc_bot.bot import MinecraftBot
from pymc_bot.config import AppSettings, ConfigStore
from pymc_bot.cortex import CortexClient, client_from_settings
from pymc_bot.events import EventLog
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
}

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


def parse_decision(text: str, source: str = "ollama") -> Decision:
    """Extract one action from a (possibly chatty) LLM response."""
    cleaned = text.strip()
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
    """Cheap, predictable behaviour used when Ollama is off/unreachable."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._step = 0

    def decide(self, state: dict[str, Any], chat: list[str]) -> Decision:
        self._step += 1
        health = float(state.get("health") or 20)
        food = float(state.get("food") or 20)
        players = state.get("players") or []
        bot_name = str(state.get("server", {}).get("username") or "PyMC_Bot")

        if health <= 8 or food <= 6:
            return Decision("eat", {}, "heuristic")

        mentioned = [line for line in chat if bot_name.lower() in line.lower()]
        if mentioned:
            return Decision(
                "say",
                {"message": self._rng.choice(["on it!", "sure, coming", "hey", "yeah?", "one sec"])},
                "heuristic",
            )

        if players:
            nearest = players[0]
            if nearest.get("distance", 99) < 12 and self._step % 3 == 0:
                return Decision("follow", {"player": nearest["name"]}, "heuristic")

        if self._step % 5 == 0:
            return Decision("mine", {"block": self._rng.choice(MINING_TARGETS)}, "heuristic")
        if self._step % 7 == 0:
            return Decision("say", {"message": self._rng.choice(["just exploring", "nice server", "anyone need wood?"])}, "heuristic")
        if state.get("time_of_day") == "night" and self._step % 4 == 0:
            return Decision("mine", {"block": "coal_ore"}, "heuristic")
        return Decision("wander", {}, "heuristic")


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
                decision = self._heuristic.decide(state, self.bot.recent_chat())
            if decision is None:
                decision = self._heuristic.decide(state, self.bot.recent_chat())
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

    def _ask_ollama(self, settings: AppSettings, state: dict[str, Any]) -> Decision:
        client = self._ollama_client(settings)
        prompt = self._build_prompt(settings, state)
        system = self.system_prompt(settings, settings.ollama.system_prompt)
        try:
            reply = client.chat([{"role": "user", "content": prompt}], system=system)
        except OllamaError:
            # Some models/servers only implement /api/generate.
            reply = client.generate(prompt, system=system)
        return parse_decision(reply, source=f"ollama:{settings.ollama.model}")

    def _ask_cortex(self, settings: AppSettings, state: dict[str, Any]) -> Decision:
        """One decision from the model hosted by Cortex LLMHoster."""
        client = self._cortex_client(settings)
        prompt = self._build_prompt(settings, state)
        reply = client.decide(prompt, system=self.system_prompt(settings, settings.cortex.system_prompt))
        model = client.last_model or settings.cortex.model or "default"
        return parse_decision(reply, source=f"cortex:{model}")

    def allowed(self, settings: AppSettings) -> dict[str, bool]:
        agent = settings.agent
        return {
            "movement": agent.allow_movement,
            "chat": agent.allow_chat,
            "mining": agent.allow_mining,
            "attacking": agent.allow_attacking,
        }

    def system_prompt(self, settings: AppSettings, custom: str = "") -> str:
        """The system prompt the LLM gets: the owner's own text, or the detailed built-in one."""
        if custom and custom.strip():
            return custom
        agent = settings.agent
        return build_system_prompt(
            name=self.bot.username,
            persona=agent.persona,
            notes=agent.persona_notes,
            allowed=self.allowed(settings),
            max_chat_length=min(agent.max_chat_length, 160),
        )

    def _build_prompt(self, settings: AppSettings, state: dict[str, Any]) -> str:
        """This turn's situation as a person would see it (plus the exact JSON at the end)."""
        return build_turn_prompt(state, self.bot.username, self.mind, self.bot.chat_history, self.allowed(settings))

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

        if action in ("mine", "dig", "gather"):
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

        if action in ("wait", "noop", "none", "nothing"):
            return True, "waited"

        return False, f"unsupported action: {action}"

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
            decision = self._heuristic.decide(state, self.bot.recent_chat())
        return self._act(decision)

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
        self.log.add(f"{'✓' if ok else '✗'} {detail}", "info" if ok else "warn", "agent")
        return ok, detail
