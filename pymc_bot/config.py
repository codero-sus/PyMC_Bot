"""Settings models and JSON persistence for PyMC_Bot."""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

DEFAULT_CONFIG_FILENAME = "pymc_bot_config.json"

DEFAULT_SYSTEM_PROMPT = """You are PyMC_Bot, an AI player inside a Minecraft server.
Every turn you receive the current world state as compact JSON and you must pick exactly
ONE action that moves you towards a sensible goal: exploring, gathering wood/stone,
following and helping other players, chatting when someone talks to you, and staying alive.

Reply with a single JSON object and nothing else. Available actions:

  {"action": "say",      "message": "<short chat message>"}
  {"action": "goto",     "x": <int>, "y": <int>, "z": <int>}      # walk to coordinates
  {"action": "follow",   "player": "<name>"}                       # follow a player
  {"action": "wander"}                                             # explore randomly nearby
  {"action": "mine",     "block": "<block name, e.g. oak_log, stone, coal_ore>"}
  {"action": "attack",   "player": "<name>"}                       # only if allowed
  {"action": "jump"}                                               # hop over something
  {"action": "eat"}                                                # eat if hungry
  {"action": "look_at_player", "player": "<name>"}
  {"action": "stop"}                                               # stop moving
  {"action": "wait"}                                               # do nothing this turn

Keep messages short and friendly. Never invent coordinates that are extremely far away.
If a player is talking to you, reply with "say" before doing anything else."""


def default_config_path() -> Path:
    """Where the JSON config lives (override with ``PYMC_BOT_CONFIG``)."""
    env = os.environ.get("PYMC_BOT_CONFIG")
    if env:
        return Path(env).expanduser()
    return Path.cwd() / DEFAULT_CONFIG_FILENAME


OFFLINE_NAME_RE = re.compile(r"^[A-Za-z0-9_]{1,16}$")


class MinecraftSettings(BaseModel):
    """Everything needed to join a server.

    ``auth="offline"`` (cracked servers): ``username`` is the in-game name,
    1-16 characters of ``[A-Za-z0-9_]``.

    ``auth="microsoft"`` (premium / online-mode): ``username`` is the **email of
    the Microsoft account** - the in-game name is whatever that account owns and
    is decided by Mojang. The first join prints a device code for
    https://www.microsoft.com/link; the token is cached in ``profiles_folder``
    afterwards, so later joins need no interaction.
    """

    host: str = "127.0.0.1"
    port: int = Field(default=25565, ge=1, le=65535)
    username: str = Field(default="PyMC_Bot", min_length=1, max_length=254)
    # "auto" lets mineflayer sniff the protocol version from the server ping.
    version: str = "auto"
    # Cracked servers need "offline"; premium accounts use "microsoft".
    auth: Literal["offline", "microsoft"] = "offline"
    connect_timeout: float = Field(default=20.0, gt=0, le=180)
    view_distance: Literal["far", "normal", "short", "tiny"] = "normal"
    # auto -> try the Node bridge, fall back to the simulated world if unavailable.
    backend: Literal["auto", "node", "simulated"] = "auto"
    # Directory for cached Microsoft refresh tokens; every account gets its own
    # sub-folder so several premium accounts can coexist.
    profiles_folder: str = ".pymc_profiles"
    # Keep re-joining after a disconnect (handy when populating a server).
    auto_rejoin: bool = True
    rejoin_seconds: float = Field(default=10.0, ge=1.0, le=600.0)

    @field_validator("host")
    @classmethod
    def _strip_host(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("host must not be empty")
        return value

    @field_validator("username")
    @classmethod
    def _strip_username(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _check_credentials(self) -> MinecraftSettings:
        if self.auth == "offline":
            if not OFFLINE_NAME_RE.match(self.username):
                raise ValueError(
                    "offline usernames must be 1-16 characters using only letters, digits and '_' "
                    f"(got {self.username!r})"
                )
        elif "@" not in self.username or "." not in self.username.split("@")[-1]:
            raise ValueError(
                "auth='microsoft' needs the Microsoft account email as the username "
                f"(got {self.username!r}); the in-game name comes from the account"
            )
        return self

    @property
    def account_label(self) -> str:
        """Short human description of the account used to join."""
        return "premium (microsoft)" if self.auth == "microsoft" else "offline (cracked)"


class OllamaSettings(BaseModel):
    """Local LLM settings. Disabled by default -- the bot still works without it."""

    enabled: bool = False
    base_url: str = "http://127.0.0.1:11434"
    model: str = "llama3.2"
    temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    decision_interval: float = Field(default=6.0, ge=0.5, le=300.0)
    request_timeout: float = Field(default=60.0, gt=0, le=600)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


class CortexSettings(BaseModel):
    """Cortex LLMHoster: a local, OpenAI-compatible model host (llama.cpp GGUF etc.).

    Disabled by default, like Ollama. ``model`` may stay blank to use the model Cortex
    marks as default. The bearer key is never stored here - only the name of the
    environment variable that holds it (Cortex's ``CORTEX_API_KEY`` by default).
    """

    enabled: bool = False
    base_url: str = "http://127.0.0.1:8624"
    api_path: str = "/v1"
    model: str = ""
    api_key_env: str = "CORTEX_API_KEY"
    temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=16, le=4096)
    json_mode: bool = True
    decision_interval: float = Field(default=6.0, ge=0.5, le=300.0)
    request_timeout: float = Field(default=120.0, gt=0, le=900)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @field_validator("base_url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not re.match(r"^https?://[^\s/]+", value):
            raise ValueError("base_url must look like http://127.0.0.1:8624")
        return value

    @field_validator("api_path")
    @classmethod
    def _check_path(cls, value: str) -> str:
        value = value.strip().strip("/")
        return f"/{value}" if value else ""

    @field_validator("model")
    @classmethod
    def _strip_model(cls, value: str) -> str:
        return value.strip()

    @field_validator("api_key_env")
    @classmethod
    def _check_env_name(cls, value: str) -> str:
        value = value.strip()
        if value and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError("api_key_env is the NAME of an environment variable, e.g. CORTEX_API_KEY")
        return value

    def key_present(self) -> bool:
        """Whether the key variable is set (the value itself is never exposed)."""
        return bool(self.api_key_env and os.environ.get(self.api_key_env, "").strip())


class AgentSettings(BaseModel):
    """What the AI player is allowed to do.

    ``mode`` chooses the brain:

    * ``auto``        - Ollama when it is enabled, else Cortex when it is enabled,
      else the heuristic.
    * ``heuristic``   - always the built-in deterministic behaviour.
    * ``ollama``      - always the local LLM (falls back to the heuristic on errors).
    * ``cortex``      - always the model hosted by Cortex LLMHoster (OpenAI-compatible,
      falls back to the heuristic on errors).
    * ``trained``     - the player model trained on recorded playtime
      (:mod:`pymc_bot.local_model`), either the run named in ``training.active_run``
      or the newest one in ``training.models_dir``. No Ollama needed.
    """

    mode: Literal["auto", "heuristic", "ollama", "cortex", "trained"] = "auto"
    allow_movement: bool = True
    allow_chat: bool = True
    allow_mining: bool = True
    allow_attacking: bool = False
    greet_players: bool = True
    max_chat_length: int = Field(default=200, ge=1, le=500)
    wander_radius: int = Field(default=16, ge=2, le=200)
    action_timeout: float = Field(default=30.0, ge=1.0, le=300.0)


class FleetBotSpec(BaseModel):
    """One extra player in the fleet (persisted so a restart can restore it)."""

    username: str = Field(min_length=1, max_length=254)
    auth: Literal["offline", "microsoft"] = "offline"
    ai: Literal["off", "heuristic", "ollama", "cortex", "trained"] = "heuristic"

    @field_validator("username")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()


class FleetSettings(BaseModel):
    """How to fill the server with players.

    ``name_pattern`` may contain ``{n}`` (1-based index) and ``{name}`` (the
    primary bot name). Names already taken are skipped automatically.
    """

    enabled: bool = True
    max_bots: int = Field(default=25, ge=1, le=200)
    count: int = Field(default=5, ge=1, le=200)
    name_pattern: str = "PyMC_Bot_{n}"
    auth: Literal["offline", "microsoft"] = "offline"
    ai_mode: Literal["off", "heuristic", "ollama", "cortex", "trained"] = "heuristic"
    stagger_seconds: float = Field(default=1.5, ge=0.0, le=60.0)
    chatter: bool = False
    chatter_interval: float = Field(default=45.0, ge=1.0, le=3600.0)
    chatter_lines: list[str] = Field(
        default_factory=lambda: [
            "hey everyone!",
            "anyone want to build something?",
            "nice server :)",
            "found any diamonds yet?",
            "brb mining",
            "this place looks great",
        ]
    )
    restore_on_start: bool = False
    roster: list[FleetBotSpec] = Field(default_factory=list)

    @field_validator("name_pattern")
    @classmethod
    def _valid_pattern(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name_pattern must not be empty")
        if "{n}" not in value and "{i}" not in value:
            raise ValueError("name_pattern must contain {n} so every bot gets a unique name")
        expanded = value.replace("{n}", "1").replace("{i}", "1").replace("{name}", "Bot")
        if len(expanded) > 254:
            raise ValueError("name_pattern expands to a name that is too long")
        return value

    @field_validator("roster")
    @classmethod
    def _cap_roster(cls, value: list[FleetBotSpec]) -> list[FleetBotSpec]:
        return value[:200]


class AntiAfkSettings(BaseModel):
    """Keep every bot looking alive so servers do not kick it for being AFK.

    Some servers kick for no movement, others for no head rotation, others for
    no arm swings - so the anti-AFK keeper mixes all of them, with randomised
    timing and smooth head turns so it does not look like a metronome.
    """

    enabled: bool = True
    # Idle time (no AI action, no manual command) before a "burst" happens.
    interval_min: float = Field(default=8.0, ge=1.0, le=600.0)
    interval_max: float = Field(default=25.0, ge=1.0, le=900.0)
    # Which habits the bots may perform.
    look: bool = True
    look_at_players: bool = True
    walk: bool = True
    max_walk_distance: float = Field(default=3.0, ge=0.5, le=12.0)
    jump: bool = True
    sneak: bool = True
    swing: bool = True
    # Chance (0-1) that a burst performs 1/2/3 habits in a row instead of one.
    combo_probability: float = Field(default=0.35, ge=0.0, le=1.0)
    # Log every burst (debug) - keeps the panel readable for big fleets.
    verbose: bool = False

    @model_validator(mode="after")
    def _check_intervals(self) -> AntiAfkSettings:
        if self.interval_max < self.interval_min:
            raise ValueError("antiafk.interval_max must be >= antiafk.interval_min")
        return self


class TrainingSettings(BaseModel):
    """Training a Minecraft player model on recorded playtime.

    The Fabric mod in ``./mod`` records the player into
    ``<gameDir>/pymc-playtime/``; ``dataset`` is the file the trainer reads and
    ``models_dir`` is where checkpoints (and their ``model.json`` cards) land.
    """

    models_dir: str = "models"
    dataset: str = "pymc-playtime/dataset.jsonl"
    # Empty means "the newest trained run in models_dir".
    active_run: str = ""
    engine: Literal["mlp", "transformer"] = "mlp"
    steps: int = Field(default=1000, ge=1, le=2_000_000)
    batch_size: int = Field(default=64, ge=1, le=4096)
    lr: float = Field(default=3e-3, gt=0.0, le=1.0)
    checkpoint_every: int = Field(default=200, ge=1, le=100_000)
    hidden: list[int] = Field(default_factory=lambda: [128, 64])
    include_blocks: bool = True
    #: Advanced training: learn entity and item vocabularies from the recorded playtime and
    #: feed them to the model (needs the mod's `/pymc advanced on` recording).
    advanced: bool = False
    entity_slots: int = Field(default=24, ge=0, le=256)
    item_slots: int = Field(default=32, ge=0, le=512)
    val_split: float = Field(default=0.1, ge=0.0, le=0.5)
    # How the trained policy is used when it drives the bot.
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    step_seconds: float = Field(default=0.4, ge=0.1, le=3.0)
    # Retrain in the background whenever the dataset grows past this many new samples.
    auto_retrain_samples: int = Field(default=0, ge=0, le=10_000_000)

    @field_validator("hidden")
    @classmethod
    def _check_hidden(cls, value: list[int]) -> list[int]:
        cleaned = [int(size) for size in value if int(size) > 0]
        if not cleaned:
            raise ValueError("training.hidden must contain at least one positive layer size")
        return cleaned[:4]


class ServerSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)


class AppSettings(BaseModel):
    """Root config object persisted to ``pymc_bot_config.json``."""

    minecraft: MinecraftSettings = Field(default_factory=MinecraftSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    cortex: CortexSettings = Field(default_factory=CortexSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    antiafk: AntiAfkSettings = Field(default_factory=AntiAfkSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    fleet: FleetSettings = Field(default_factory=FleetSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    def public_dict(self) -> dict[str, Any]:
        """Config as a plain dict (safe to send to the web UI)."""
        return self.model_dump(mode="json")


class ConfigStore:
    """Load/save :class:`AppSettings` with atomic writes.

    The store is thread-safe because the FastAPI handlers and background
    threads (bot + agent loop) all read the current config.
    """

    def __init__(self, path: Path | str | None = None, autoload: bool = True) -> None:
        self.path = Path(path) if path is not None else default_config_path()
        self._lock = threading.RLock()
        self._settings = AppSettings()
        self._load_error: str | None = None
        if autoload and self.path.exists():
            self.load()

    # ------------------------------------------------------------------ load
    def load(self) -> AppSettings:
        with self._lock:
            if not self.path.exists():
                self._settings = AppSettings()
                self._load_error = None
                return self._settings
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                self._settings = AppSettings.model_validate(raw)
                self._load_error = None
            except (OSError, json.JSONDecodeError, ValidationError) as exc:
                # Never crash on a bad config file: keep the defaults.
                self._settings = AppSettings()
                self._load_error = f"Could not read {self.path.name}: {exc}"
            return self._settings

    # ------------------------------------------------------------------ save
    def save(self, settings: AppSettings | None = None) -> AppSettings:
        with self._lock:
            if settings is not None:
                self._settings = settings
            payload = json.dumps(self._settings.public_dict(), indent=2) + "\n"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(self.path)
            return self._settings

    def update(self, patch: dict[str, Any]) -> AppSettings:
        """Deep-merge a partial dict into the current settings and persist."""
        with self._lock:
            current = self._settings.public_dict()
            merged = _deep_merge(current, patch)
            settings = AppSettings.model_validate(merged)
            return self.save(settings)

    # --------------------------------------------------------------- helpers
    @property
    def settings(self) -> AppSettings:
        with self._lock:
            return self._settings

    @property
    def load_error(self) -> str | None:
        with self._lock:
            return self._load_error


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
