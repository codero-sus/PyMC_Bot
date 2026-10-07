"""The bot fleet: one or many Minecraft players under one controller.

The panel's main bot (``primary``) is the one described by ``minecraft.*`` in the
config.  On top of that you can add any number of extra players:

* **offline bots** for cracked servers, typically a whole batch generated from a
  name pattern ("PyMC_Bot_1" … "PyMC_Bot_25") to populate a server;
* **premium bots** using a Microsoft account each (``auth="microsoft"``, one
  email per bot).  The first join prints a device code, the token is cached per
  account, and every bot runs in its own Node process with its own cache folder.

Each member owns its own :class:`~pymc_bot.bot.MinecraftBot` and, optionally, an
:class:`~pymc_bot.agent.AgentLoop` — heuristics by default so twenty bots do not
all hammer the same Ollama endpoint.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from pymc_bot.agent import AgentLoop
from pymc_bot.antiafk import AntiAfkKeeper
from pymc_bot.backends import BackendError
from pymc_bot.bot import MinecraftBot
from pymc_bot.config import (
    OFFLINE_NAME_RE,
    AppSettings,
    ConfigStore,
    FleetBotSpec,
)
from pymc_bot.events import EventLog

AI_MODES = ("off", "heuristic", "ollama", "trained")
SPONSORS = ("primary", "populate", "premium", "custom")


class FleetError(RuntimeError):
    """Raised when a bot cannot be added (bad name, duplicate, fleet full...)."""


def expand_names(
    pattern: str,
    count: int,
    base_name: str = "PyMC_Bot",
    taken: Iterable[str] = (),
) -> list[str]:
    """Turn a name pattern into ``count`` unique names.

    ``{n}``/``{i}`` are replaced with a 1-based counter, ``{name}`` with the
    primary bot's name.  Names already in ``taken`` are skipped.
    """
    if count < 1:
        raise FleetError("count must be at least 1")
    reserved = {name.strip().lower() for name in taken if name}
    names: list[str] = []
    index = 1
    guard = 0
    while len(names) < count:
        guard += 1
        if guard > count + 5000:
            raise FleetError("could not generate unique names from that pattern")
        candidate = (
            pattern.replace("{n}", str(index))
            .replace("{i}", str(index))
            .replace("{name}", base_name)
        )
        index += 1
        if not candidate or candidate.lower() in reserved:
            continue
        reserved.add(candidate.lower())
        names.append(candidate)
    return names


def validate_username(username: str, auth: str) -> str:
    """Check a username for the given auth mode and return it stripped."""
    username = (username or "").strip()
    if not username:
        raise FleetError("the username must not be empty")
    if auth == "microsoft":
        if "@" not in username or "." not in username.split("@")[-1]:
            raise FleetError(
                "a premium bot needs the Microsoft account email (e.g. player@example.com); "
                "the in-game name comes from that account"
            )
        if len(username) > 254:
            raise FleetError("that email is too long")
        return username
    if not OFFLINE_NAME_RE.match(username):
        raise FleetError(
            f"'{username}' is not a valid offline username: 1-16 characters, letters, digits and '_' only "
            "(offline servers make the name up, premium accounts keep their own)"
        )
    return username


@dataclass
class FleetMember:
    """One extra player managed by the fleet."""

    username: str
    auth: str = "offline"
    ai: str = "heuristic"
    sponsor: str = "custom"
    bot: MinecraftBot | None = None
    agent: AgentLoop | None = None
    state: str = "queued"
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    next_chatter_at: float = 0.0

    @property
    def key(self) -> str:
        return self.username.strip().lower()


class BotFleet:
    """Adds, starts, stops and reports on every bot of this instance."""

    def __init__(
        self,
        store: ConfigStore,
        log: EventLog | None = None,
        auto_reconnect: bool = True,
    ) -> None:
        self.store = store
        self.log = log or EventLog()
        self.auto_reconnect = auto_reconnect
        self._lock = threading.RLock()
        self._members: dict[str, FleetMember] = {}
        self._primary: MinecraftBot | None = None
        self._primary_agent: AgentLoop | None = None
        self._selected: str | None = None
        self._stopping = threading.Event()
        self._chatter_thread: threading.Thread | None = None
        self._batch_thread: threading.Thread | None = None
        self.antiafk = AntiAfkKeeper(self.all_bots, self.store, self.log)
        self.antiafk.start()

    # ------------------------------------------------------------- primary bot
    @property
    def primary(self) -> MinecraftBot:
        """The bot described by the ``minecraft`` section of the config."""
        with self._lock:
            if self._primary is None:
                settings = self.store.settings
                self._primary = MinecraftBot(
                    self.store,
                    self.log,
                    auto_reconnect=self.auto_reconnect,
                    max_reconnects=None if settings.minecraft.auto_rejoin else 3,
                )
            return self._primary

    # -------------------------------------------------------------- selection
    @property
    def selected_username(self) -> str:
        with self._lock:
            return self._selected or self.primary.username

    def select(self, username: str | None) -> str:
        """Choose which bot the main panel (and ``/api/bot/*``) talks to."""
        if not username:
            with self._lock:
                self._selected = None
            return self.primary.username
        bot = self.find(username)
        if bot is None:
            raise FleetError(f"no bot named '{username}' in the fleet")
        with self._lock:
            self._selected = bot.username.strip().lower()
        return bot.username

    def selected_bot(self) -> MinecraftBot:
        with self._lock:
            key = self._selected
        if key is None:
            return self.primary
        member = self._members.get(key)
        return member.bot if member and member.bot else self.primary

    def primary_agent(self) -> AgentLoop:
        """The (cached) AI loop of the main bot."""
        with self._lock:
            if self._primary_agent is None:
                self._primary_agent = AgentLoop(self.primary, self.store, self.log)
            return self._primary_agent

    def agent_for(self, bot: MinecraftBot | None = None) -> AgentLoop:
        """The AI loop belonging to a bot (created once per bot)."""
        bot = bot or self.selected_bot()
        with self._lock:
            if bot is self._primary or self._primary is None and bot is not None:
                return self.primary_agent()
            # fleet bots follow their own ai mode through AgentLoop(brain=...)
            for member in self._members.values():
                if member.bot is bot:
                    if member.agent is None:
                        member.agent = AgentLoop(
                            bot, self.store, self.log, prefer_ollama=(member.ai == "ollama"), brain=member.ai
                        )
                    return member.agent
        return AgentLoop(bot, self.store, self.log)

    def get(self, username: str) -> FleetMember | None:
        key = (username or "").strip().lower()
        with self._lock:
            return self._members.get(key)

    def find(self, username: str) -> MinecraftBot | None:
        """Find a bot by configured name, in-game name or email."""
        key = (username or "").strip().lower()
        if not key:
            return None
        with self._lock:
            member = self._members.get(key)
            if member and member.bot is not None:
                return member.bot
            for candidate in self._members.values():
                if candidate.bot is not None and candidate.bot.username.strip().lower() == key:
                    return candidate.bot
            primary = self._primary
            if primary is not None and primary.username.strip().lower() == key:
                return primary
            if primary is None:
                return None
        return None

    def all_bots(self) -> list[MinecraftBot]:
        """Every player of this instance: the main bot plus the fleet."""
        bots: list[MinecraftBot] = []
        with self._lock:
            if self._primary is not None:
                bots.append(self._primary)
            bots.extend(member.bot for member in self._members.values() if member.bot is not None)
        return bots

    @property
    def members(self) -> list[FleetMember]:
        with self._lock:
            return list(self._members.values())

    @property
    def size(self) -> int:
        """Total players this instance controls: the primary bot plus the fleet."""
        with self._lock:
            return 1 + len(self._members)

    # ------------------------------------------------------------------ spawn
    def _spawn_settings(self, member: FleetMember) -> AppSettings:
        """Config for one member: its own username/auth, everything else shared."""
        settings = self.store.settings
        minecraft = settings.minecraft.model_copy(update={"username": member.username, "auth": member.auth})
        return settings.model_copy(update={"minecraft": minecraft})

    def spawn(
        self,
        username: str,
        auth: str = "offline",
        ai: str = "heuristic",
        sponsor: str = "custom",
        start: bool = True,
        _batch: bool = False,
    ) -> FleetMember:
        """Register a new bot and start connecting to it in the background."""
        username = validate_username(username, auth)
        if ai not in AI_MODES:
            raise FleetError(f"unknown ai mode '{ai}' (use one of {', '.join(AI_MODES)})")

        settings = self.store.settings
        if not settings.fleet.enabled:
            raise FleetError("the fleet is disabled in the settings (fleet.enabled = false)")

        alive = 0
        with self._lock:
            for member in self._members.values():
                if member.key == username.lower():
                    raise FleetError(f"'{username}' is already in the fleet")
            primary = self._primary
            if primary is not None and primary.username.strip().lower() == username.lower():
                raise FleetError(f"'{username}' is already used by the main bot")
            if primary is None and settings.minecraft.username.lower() == username.lower():
                raise FleetError(f"'{username}' is already used by the main bot")
            alive = 1 + len(self._members)
            if alive >= settings.fleet.max_bots:
                raise FleetError(
                    f"the fleet is full ({alive}/{settings.fleet.max_bots} players) - "
                    "raise fleet.max_bots or stop a bot first"
                )

            member = FleetMember(username=username, auth=auth, ai=ai, sponsor=sponsor)
            member.bot = MinecraftBot(
                self.store,
                self.log,
                auto_reconnect=self.auto_reconnect,
                max_reconnects=None if settings.minecraft.auto_rejoin else 3,
                rejoin_seconds=settings.minecraft.rejoin_seconds,
            )
            member.agent = AgentLoop(
                member.bot, self.store, self.log, prefer_ollama=(ai == "ollama"), brain=ai
            )
            self._members[member.key] = member

        self.log.add(
            f"Adding player '{username}' ({auth}" + (f", ai: {ai}" if ai != "off" else "") + ").",
            "info",
            "fleet",
        )
        if not _batch:
            self._persist_roster()
        if start:
            threading.Thread(
                target=self._start_member, args=(member,), name=f"spawn-{username}", daemon=True
            ).start()
        return member

    def _start_member(self, member: FleetMember) -> None:
        member.state = "connecting"
        member.error = None
        try:
            assert member.bot is not None
            member.bot.start(self._spawn_settings(member))
        except (BackendError, Exception) as exc:  # noqa: B014 - any failure is a spawn failure
            member.state = "error"
            member.error = str(exc)
            self.log.add(f"Player '{member.username}' could not join: {exc}", "error", "fleet")
            return
        member.state = "connected"
        if member.bot is not None:
            member.bot.touch_activity()
            self.antiafk.record(member.bot).schedule(self.store.settings.antiafk, self.antiafk._rng)
        self.log.add(f"Player '{member.bot.username}' is now in the game.", "success", "fleet")
        if member.agent is not None and member.ai != "off":
            member.agent.start()
        self._ensure_chatter()

    def spawn_many(
        self,
        count: int | None = None,
        pattern: str | None = None,
        auth: str | None = None,
        ai: str | None = None,
        stagger: float | None = None,
        sponsor: str = "populate",
    ) -> list[FleetMember]:
        """Populate the server: register ``count`` bots and join them one by one.

        Registration happens immediately (so the panel lists all of them right
        away); joining is staggered to avoid connection throttling and runs on a
        background thread, so this call returns at once.
        """
        settings = self.store.settings.fleet
        count = int(count if count is not None else settings.count)
        pattern = pattern or settings.name_pattern
        auth = auth or settings.auth
        ai = ai or settings.ai_mode
        stagger = settings.stagger_seconds if stagger is None else float(stagger)

        if count < 1:
            raise FleetError("count must be at least 1")
        taken = [member.username for member in self.members]
        primary_username = self.primary.username if self._primary is not None else self.store.settings.minecraft.username
        taken.append(primary_username)
        names = expand_names(pattern, count, base_name=primary_username, taken=taken)

        capacity = self.store.settings.fleet.max_bots - self.size
        if len(names) > capacity:
            raise FleetError(
                f"{len(names)} more players would exceed fleet.max_bots="
                f"{self.store.settings.fleet.max_bots} (room for {max(0, capacity)})"
            )
        for name in names:
            validate_username(name, auth)

        members: list[FleetMember] = []
        for name in names:
            members.append(self.spawn(name, auth=auth, ai=ai, sponsor=sponsor, start=False, _batch=True))
        self._persist_roster()

        def _join_all() -> None:
            for index, member in enumerate(members):
                if self._stopping.is_set():
                    member.state = "stopped"
                    continue
                if index:
                    self._sleep(stagger)
                if member.state == "stopped":  # removed while queued
                    continue
                self._start_member(member)
            self.log.add(
                f"Population finished: {sum(1 for m in members if m.state == 'connected')}/{len(members)} joined.",
                "success",
                "fleet",
            )

        with self._lock:
            if self._batch_thread and self._batch_thread.is_alive():
                # Chain the new batch after the running one instead of racing it.
                previous = self._batch_thread
            else:
                previous = None
            thread = threading.Thread(target=_join_all, name="fleet-join", daemon=True)
            if previous is not None:
                threading.Thread(
                    target=lambda: (previous.join(), _join_all()),
                    name="fleet-join-queued",
                    daemon=True,
                ).start()
            else:
                thread.start()
            self._batch_thread = thread

        self.log.add(
            f"Queued {len(members)} player(s) to populate the server with a {stagger:g}s gap "
            f"between joins: {', '.join(m.username for m in members[:8])}"
            + (" ..." if len(members) > 8 else ""),
            "info",
            "fleet",
        )
        return members

    def _sleep(self, seconds: float) -> None:
        deadline = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < deadline:
            if self._stopping.is_set():
                return
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    # ------------------------------------------------------------- lifecycle
    def start_bot(self, username: str) -> FleetMember | None:
        """(Re)connect a bot that is already in the fleet."""
        member = self.get(username)
        if member is None:
            return None
        if member.bot is not None and member.bot.connected:
            return member
        member.state = "connecting"
        threading.Thread(
            target=self._start_member, args=(member,), name=f"restart-{member.username}", daemon=True
        ).start()
        return member

    def stop_bot(self, username: str) -> bool:
        member = self.get(username)
        if member is None:
            return False
        self._stop_member(member)
        return True

    def _stop_member(self, member: FleetMember) -> None:
        if member.agent is not None:
            member.agent.stop(wait=False)
        if member.bot is not None:
            member.bot.stop()
        member.state = "stopped"
        self.log.add(f"Player '{member.username}' left the game.", "info", "fleet")

    def remove_bot(self, username: str) -> bool:
        """Stop and forget a bot (it will not come back after a restart)."""
        member = self.get(username)
        if member is None:
            return False
        self._stop_member(member)
        with self._lock:
            self._members.pop(member.key, None)
            if self._selected == member.key:
                self._selected = None
        self._persist_roster()
        self.log.add(f"Player '{member.username}' removed from the fleet.", "info", "fleet")
        return True

    def stop_all(self, remove: bool = False) -> int:
        """Stop every fleet bot (the primary bot is managed by /api/bot/stop)."""
        stopped = 0
        for member in self.members:
            if self._stopping.is_set():
                pass
            self._stop_member(member)
            stopped += 1
            if remove:
                with self._lock:
                    self._members.pop(member.key, None)
        if remove:
            self._persist_roster()
        if stopped:
            self.log.add(f"Stopped {stopped} fleet player(s).", "info", "fleet")
        return stopped

    def start_all(self) -> int:
        started = 0
        for member in self.members:
            if member.bot is None or not member.bot.connected:
                self.start_bot(member.username)
                started += 1
        return started

    def shutdown(self) -> None:
        """Stop everything (used when the panel shuts down)."""
        self._stopping.set()
        for member in self.members:
            self._stop_member(member)
        with self._lock:
            self._members.clear()
            self.antiafk.stop(wait=False)
        agent = self._primary_agent
        if agent is not None:
            agent.stop(wait=False)
        if self._primary is not None:
            self._primary.stop()

    # ------------------------------------------------------------------ roster
    def _persist_roster(self) -> None:
        specs = [
            FleetBotSpec(username=member.username, auth=member.auth, ai=member.ai).model_dump()
            for member in self.members
        ]
        try:
            self.store.update({"fleet": {"roster": specs}})
        except Exception as exc:  # pragma: no cover - a bad write must not kill the bot
            self.log.add(f"Could not save the fleet roster: {exc}", "warn", "fleet")

    def restore_roster(self) -> int:
        """Re-add every player remembered in the config (if enabled)."""
        settings = self.store.settings
        if not settings.fleet.enabled or not settings.fleet.restore_on_start:
            return 0
        restored = 0
        for spec in settings.fleet.roster:
            if self.size >= settings.fleet.max_bots:
                self.log.add("Fleet roster truncated by fleet.max_bots.", "warn", "fleet")
                break
            try:
                self.spawn(spec.username, auth=spec.auth, ai=spec.ai, sponsor="custom", _batch=True)
                restored += 1
            except FleetError as exc:
                self.log.add(f"Skipping remembered player '{spec.username}': {exc}", "warn", "fleet")
        if restored:
            self.log.add(f"Restored {restored} player(s) from the saved roster.", "info", "fleet")
            self._ensure_chatter()
        return restored

    def clear_roster(self) -> None:
        self.store.update({"fleet": {"roster": []}})

    # --------------------------------------------------------------- chatter
    def _ensure_chatter(self) -> None:
        """Start the shared chatter thread (one per fleet, not one per bot)."""
        if not self.store.settings.fleet.chatter:
            return
        with self._lock:
            if self._chatter_thread and self._chatter_thread.is_alive():
                return
            self._chatter_thread = threading.Thread(
                target=self._chatter_loop, name="fleet-chatter", daemon=True
            )
            self._chatter_thread.start()

    def _chatter_loop(self) -> None:
        while not self._stopping.is_set():
            fleet = self.store.settings.fleet
            if not fleet.chatter:
                return
            now = time.monotonic()
            for member in self.members:
                if self._stopping.is_set():
                    return
                if member.bot is None or not member.bot.connected:
                    continue
                if member.next_chatter_at == 0.0:
                    member.next_chatter_at = now + random.uniform(3.0, fleet.chatter_interval)
                if now < member.next_chatter_at:
                    continue
                member.next_chatter_at = now + fleet.chatter_interval * random.uniform(0.6, 1.6)
                if self.store.settings.agent.allow_chat and fleet.chatter_lines:
                    member.bot.say(random.choice(fleet.chatter_lines))
            time.sleep(2.0)

    # ----------------------------------------------------------------- status
    def _member_status(self, member: FleetMember) -> dict[str, Any]:
        bot = member.bot
        snapshot = bot.snapshot() if bot is not None else {}
        state = member.state
        if state == "queued":
            pass
        elif bot is not None and bot.connected:
            state = "connected"
        elif state == "connected" and bot is not None and not bot.connected:
            state = bot.state if bot.state != "connected" else "disconnected"
        return {
            "username": bot.username if bot is not None else member.username,
            "configured_username": member.username,
            "auth": member.auth,
            "ai": member.ai,
            "ai_running": bool(member.agent and member.agent.running),
            "sponsor": member.sponsor,
            "state": state,
            "error": member.error,
            "position": snapshot.get("position"),
            "health": snapshot.get("health"),
            "uptime": snapshot.get("uptime", 0.0),
            "blocks_mined": (bot.stats["blocks_mined"] if bot is not None else 0),
            "chats_sent": (bot.stats["chats_sent"] if bot is not None else 0),
            "reconnects": (bot.stats["reconnects"] if bot is not None else 0),
            "joined_at": member.created_at,
            "busy": bool(bot.busy) if bot is not None else False,
            "idle_seconds": round(bot.idle_seconds, 1) if bot is not None else None,
            "antiafk_pokes": self.antiafk.record(bot).pokes if bot is not None else 0,
        }

    def status(self) -> dict[str, Any]:
        settings = self.store.settings
        bots = [self._member_status(member) for member in self.members]
        connected = sum(1 for entry in bots if entry["state"] == "connected")
        primary = self.primary
        primary_status = {
            "username": primary.username,
            "configured_username": settings.minecraft.username,
            "auth": settings.minecraft.auth,
            "ai": "ollama" if settings.ollama.enabled else "heuristic",
            "ai_running": False,
            "sponsor": "primary",
            "state": primary.state,
            "error": None,
            "position": primary.snapshot().get("position"),
            "health": primary.snapshot().get("health"),
            "uptime": primary.stats["uptime"],
            "blocks_mined": primary.stats["blocks_mined"],
            "chats_sent": primary.stats["chats_sent"],
            "reconnects": primary.stats["reconnects"],
            "joined_at": None,
            "busy": primary.busy,
            "idle_seconds": round(primary.idle_seconds, 1),
            "antiafk_pokes": self.antiafk.record(primary).pokes,
        }
        selected = self.selected_bot().username
        for entry in [primary_status, *bots]:
            entry["selected"] = entry["username"] == selected
        return {
            "enabled": settings.fleet.enabled,
            "antiafk": {
                "enabled": self.store.settings.antiafk.enabled,
                "running": self.antiafk.running,
                "pokes": self.antiafk.status([self.primary, *(m.bot for m in self.members if m.bot)])["pokes"],
            },
            "max_bots": settings.fleet.max_bots,
            "size": 1 + len(bots),
            "extra_bots": len(bots),
            "connected": connected + (1 if primary.connected else 0),
            "roster_size": len(settings.fleet.roster),
            "selected": selected,
            "primary": primary_status,
            "bots": [primary_status, *bots],
        }

    def broadcast(self, message: str) -> int:
        """Say something as every connected bot."""
        sent = 0
        for bot in [self.primary, *(m.bot for m in self.members)]:
            if bot is not None and bot.connected and bot.say(message):
                sent += 1
        return sent

    def action_target(self, mode: str) -> list[MinecraftBot]:
        """Resolve an action target selector: ``selected``/``all``/``<username>``."""
        mode = (mode or "selected").strip()
        if mode == "all":
            return [b for b in [self.primary, *(m.bot for m in self.members)] if b is not None]
        if mode == "selected":
            return [self.selected_bot()]
        bot = self.find(mode)
        if bot is None:
            raise FleetError(f"no bot named '{mode}' in the fleet")
        return [bot]
