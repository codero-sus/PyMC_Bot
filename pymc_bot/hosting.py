"""Support for free / shared Minecraft hosts: Aternos, exaroton, Minehut, FalixNodes and tunnels.

What differs from a server on your own PC:

* **Addresses** - ``myserver.aternos.me`` has an SRV record that points at the real host and port, so
  the port stays at 25565 (the Node side follows SRV records like the real client). Tunnels and
  "dyn IP" addresses come with a port, which people paste as ``host:port`` -
  :func:`split_address` turns that into the two config fields.
* **Sleeping servers** - while a free server is offline or starting, its address still answers the
  server list ping with a placeholder ("● Offline", "This server is offline"). :func:`probe_server`
  reports that as ``asleep`` so the bot can say "start it from the dashboard" (and optionally wait
  for it) instead of failing with a cryptic kick.
* **Cracked mode, whitelists, join throttling** - per-host hints for the usual kick reasons.
* **House rules** - Aternos does not allow bots to keep an empty server online (servers are shared
  and free). :class:`AloneWatcher` makes the bots leave when no real player is online on such hosts,
  so the server can stop as intended.

Made with love by codero-sus - https://github.com/codero-sus
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pymc_bot.events import EventLog

if TYPE_CHECKING:  # pragma: no cover
    from pymc_bot.config import AppSettings, ConfigStore

PROBE_SCRIPT = Path(__file__).resolve().parent / "node" / "probe.js"
DEFAULT_PORT = 25565


@dataclass(frozen=True)
class HostProfile:
    """What PyMC_Bot knows about one hosting provider."""

    key: str
    name: str
    suffixes: tuple[str, ...]
    start_hint: str
    cracked_hint: str
    motd_words: tuple[str, ...] = ()
    whitelist_hint: str = ""
    # The host's rules forbid keeping an empty server online with bots.
    leave_when_alone: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "start_hint": self.start_hint,
            "cracked_hint": self.cracked_hint,
            "whitelist_hint": self.whitelist_hint,
            "leave_when_alone": self.leave_when_alone,
            "notes": list(self.notes),
        }


PROFILES: tuple[HostProfile, ...] = (
    HostProfile(
        key="aternos",
        name="Aternos",
        suffixes=("aternos.me", "aternos.host", "aternos.org"),
        motd_words=("aternos",),
        start_hint=(
            "Start it from the Aternos dashboard (aternos.org > Server > Start), wait until it says "
            "Online, then connect."
        ),
        cracked_hint="On Aternos open Options and turn on Cracked so offline-mode bots can join (or use a Microsoft account).",
        whitelist_hint="On Aternos open Players > Whitelist and add the bot's name.",
        leave_when_alone=True,
        notes=(
            "Keep the port at 25565 for name.aternos.me (it has an SRV record), or paste the Dyn IP "
            "with its port, e.g. abc.aternos.host:12345.",
            "Aternos does not allow bots to keep an empty server online, so the bots leave when no "
            "real player is on and the server can stop as usual.",
        ),
    ),
    HostProfile(
        key="exaroton",
        name="exaroton",
        suffixes=("exaroton.me",),
        motd_words=("exaroton",),
        start_hint="Start it from the exaroton dashboard or app, then connect.",
        cracked_hint="On exaroton open Options and switch off online mode (cracked) for offline-mode bots, or use a Microsoft account.",
        whitelist_hint="On exaroton add the bot's name under Players > Whitelist.",
        notes=("Keep the port at 25565 for name.exaroton.me (it has an SRV record).",),
    ),
    HostProfile(
        key="minehut",
        name="Minehut",
        suffixes=("minehut.gg", "minehut.com"),
        motd_words=("minehut",),
        start_hint="Start it from the Minehut dashboard (minehut.com), then connect.",
        cracked_hint="Minehut's network only lets premium accounts in - switch Auth to Microsoft.",
        notes=("Minehut servers sit behind Minehut's proxy, which needs a Microsoft (premium) account.",),
    ),
    HostProfile(
        key="falix",
        name="FalixNodes",
        suffixes=("falixsrv.me", "falix.gg", "falixnodes.net"),
        motd_words=("falix",),
        start_hint="Start it from the FalixNodes panel, then connect.",
        cracked_hint="On FalixNodes set online-mode=false in server.properties (Files) for offline-mode bots.",
        notes=("FalixNodes addresses often need their port - paste host:port into Host.",),
    ),
    HostProfile(
        key="playit",
        name="playit.gg tunnel",
        suffixes=("playit.gg", "joinmc.link", "ply.gg", "at.ply.gg"),
        start_hint="Make sure both the Minecraft server and the playit agent are running on the host PC.",
        cracked_hint="Set online-mode=false in the server's server.properties for offline-mode bots.",
    ),
    HostProfile(
        key="ngrok",
        name="ngrok tunnel",
        suffixes=("ngrok.io", "ngrok.app"),
        start_hint="Make sure the server and the ngrok tunnel are running; ngrok addresses change on restart.",
        cracked_hint="Set online-mode=false in the server's server.properties for offline-mode bots.",
        notes=("ngrok TCP addresses include a port (e.g. 0.tcp.eu.ngrok.io:12345) - paste it whole into Host.",),
    ),
    HostProfile(
        key="e4mc",
        name="e4mc (shared LAN world)",
        suffixes=("e4mc.link",),
        start_hint="The player sharing the world must keep their game open with e4mc running.",
        cracked_hint="A shared LAN world checks accounts like the host's game - use a Microsoft account if offline bots are refused.",
    ),
)

GENERIC_START_HINT = "Start the server (or wait until it has finished starting), then connect."
GENERIC_CRACKED_HINT = (
    "The server is in online mode (premium accounts only). Switch Auth to Microsoft, or set "
    "online-mode=false in server.properties so offline-mode bots can join."
)
GENERIC_WHITELIST_HINT = "Add the bot to the whitelist (/whitelist add <name>) or turn the whitelist off."


# --------------------------------------------------------------------------- addresses
_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)


def split_address(text: str) -> tuple[str, int | None]:
    """``"name.aternos.me:12345"`` -> ``("name.aternos.me", 12345)``; no port -> ``(host, None)``.

    Accepts ``[::1]:25565`` for IPv6, ignores a ``minecraft://`` style prefix and a trailing slash.
    A bare IPv6 address (several colons, no brackets) is returned unchanged.
    """
    value = _SCHEME.sub("", str(text or "").strip()).rstrip("/").strip()
    if value.startswith("["):
        host, _, rest = value[1:].partition("]")
        if rest.startswith(":") and rest[1:].isdigit():
            return host, int(rest[1:])
        return host, None
    if value.count(":") == 1:
        host, _, port = value.partition(":")
        if port.isdigit() and host:
            return host.strip(), int(port)
    return value, None


def detect_host(host: str, motd: str = "") -> HostProfile | None:
    """The hosting provider an address (or a ping MOTD) belongs to, if PyMC_Bot knows it."""
    name = split_address(host)[0].lower().rstrip(".")
    for profile in PROFILES:
        if any(name == suffix or name.endswith("." + suffix) for suffix in profile.suffixes):
            return profile
    text = (motd or "").lower()
    if text:
        for profile in PROFILES:
            if any(word in text for word in profile.motd_words):
                return profile
    return None


def host_tips(host: str, auth: str = "offline") -> list[str]:
    """Short tips for the panel / doctor when the address belongs to a known host."""
    profile = detect_host(host)
    if profile is None:
        return []
    tips = [f"{profile.name} detected."]
    if auth == "offline":
        tips.append(profile.cracked_hint)
    tips.extend(profile.notes)
    return tips


def leave_rule_active(settings: AppSettings) -> tuple[bool, HostProfile | None]:
    """Whether bots should leave when no real player is online (always on for Aternos-like hosts)."""
    profile = detect_host(settings.minecraft.host)
    if settings.minecraft.leave_when_alone == "always":
        return True, profile
    return bool(profile and profile.leave_when_alone), profile


# --------------------------------------------------------------------------- probe
def probe_server(host: str, port: int = DEFAULT_PORT, timeout: float = 8.0) -> dict[str, Any]:
    """Server list ping through ``node/probe.js`` (follows SRV records). Never raises.

    Returns ``{"ok": True, "online": bool, "asleep": bool, "state", "version", "protocol",
    "players": {"online", "max", "sample"}, "motd", "latency", "resolved"}`` or
    ``{"ok": False, "error": "..."}``, plus ``host``/``port``/``hosting`` (a profile dict or None).
    """
    host, inline_port = split_address(host)
    port = int(inline_port or port or DEFAULT_PORT)
    result: dict[str, Any]
    node = shutil.which("node")
    if not node:
        result = {"ok": False, "error": "Node.js was not found on PATH (needed to ping Minecraft servers)."}
    elif not PROBE_SCRIPT.exists():  # pragma: no cover - packaging problem
        result = {"ok": False, "error": f"probe script missing: {PROBE_SCRIPT}"}
    else:
        try:
            out = subprocess.run(
                [node, str(PROBE_SCRIPT), host, str(port), str(int(timeout * 1000))],
                capture_output=True,
                text=True,
                timeout=timeout + 5,
                cwd=str(Path(__file__).resolve().parent.parent),
            )
            lines = [line for line in out.stdout.splitlines() if line.strip().startswith("{")]
            result = json.loads(lines[-1]) if lines else {
                "ok": False,
                "error": (out.stderr.strip().splitlines() or ["the probe printed nothing"])[-1],
            }
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            result = {"ok": False, "error": f"could not run the probe: {exc}"}
    profile = detect_host(host, str(result.get("motd") or ""))
    result.update({"host": host, "port": port, "hosting": profile.as_dict() if profile else None})
    return result


def describe_probe(result: dict[str, Any]) -> str:
    """One line for logs, the CLI and the panel."""
    where = f"{result.get('host')}:{result.get('port')}"
    hosting = result.get("hosting") or {}
    host_name = f" ({hosting['name']})" if hosting.get("name") else ""
    if not result.get("ok"):
        return f"{where}{host_name} did not answer: {result.get('error') or 'unknown error'}"
    if result.get("asleep"):
        start = hosting.get("start_hint") or GENERIC_START_HINT
        motd = f' - "{result["motd"]}"' if result.get("motd") else ""
        return f"{where}{host_name} is {result.get('state', 'offline')}{motd}. {start}"
    players = result.get("players") or {}
    via = ""
    if result.get("resolved"):
        via = f" via {result['resolved']['host']}:{result['resolved']['port']}"
    return (
        f"{where}{host_name} is online{via}: {result.get('version') or '?'} "
        f"(protocol {result.get('protocol')}), {players.get('online', 0)}/{players.get('max', 0)} players, "
        f"{result.get('latency', '?')} ms"
    )


# --------------------------------------------------------------------------- etiquette
class AloneWatcher:
    """Makes the bots leave when no real player has been online for a while.

    On by default for hosts whose rules forbid bots that keep an empty server online (Aternos), and
    for any server when ``minecraft.leave_when_alone`` is ``"always"``. "Real" players are everyone
    on the tab list who is not one of this app's own bots. Bots that left are not re-joined
    automatically - press Connect when people are back.
    """

    def __init__(
        self,
        bots: Callable[[], Iterable[Any]],
        store: ConfigStore,
        log: EventLog | None = None,
        interval: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
        leave: Callable[[], Any] | None = None,
    ) -> None:
        self._bots = bots
        self._leave = leave
        self.store = store
        self.log = log or EventLog()
        self.interval = interval
        self._clock = clock
        self._alone_since: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="alone-watcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.check()
            except Exception as exc:  # pragma: no cover - the watcher must never die
                self.log.add(f"Leave-when-alone check failed: {exc}", "debug", "hosting")

    @property
    def alone_for(self) -> float:
        return 0.0 if self._alone_since is None else max(0.0, self._clock() - self._alone_since)

    def check(self) -> bool:
        """One pass; returns True when it made the bots leave."""
        settings = self.store.settings
        active, profile = leave_rule_active(settings)
        connected = [bot for bot in self._bots() if getattr(bot, "connected", False)]
        if not active or not connected:
            self._alone_since = None
            return False
        ours = {str(bot.username).lower() for bot in self._bots()}
        online: set[str] = set()
        for bot in connected:
            backend = getattr(bot, "backend", None)
            snapshot = backend.snapshot() if backend is not None else {}
            names = snapshot.get("online_players")
            if names is None:  # older bridges: fall back to the players in view
                names = [player.get("name", "") for player in snapshot.get("players") or []]
            online.update(str(name).lower() for name in names if name)
        humans = online - ours
        if humans:
            self._alone_since = None
            return False
        now = self._clock()
        if self._alone_since is None:
            self._alone_since = now
            return False
        limit = settings.minecraft.alone_minutes * 60.0
        if now - self._alone_since < limit:
            return False
        reason = (
            f"{profile.name} does not allow bots to keep an empty server online"
            if profile and profile.leave_when_alone
            else "leave_when_alone is set to always"
        )
        self.log.add(
            f"No real player has been online for {settings.minecraft.alone_minutes:g} min - the bots are "
            f"leaving ({reason}). Press Connect when people are back.",
            "warn",
            "hosting",
        )
        if self._leave is not None:
            self._leave()
        else:
            for bot in connected:
                try:
                    bot.stop()
                except Exception as exc:  # pragma: no cover
                    self.log.add(f"Could not stop {bot.username}: {exc}", "warn", "hosting")
        self._alone_since = None
        return True
