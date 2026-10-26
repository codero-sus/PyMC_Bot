"""FastAPI + Uvicorn config page and REST/WebSocket API for PyMC_Bot."""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.requests import Request

from pymc_bot import (
    __author__,
    __author_url__,
    __license__,
    __version__,
    __websites__,
    about,
    autopilot,
    hosting,
    node_setup,
    skills,
)
from pymc_bot import auth as panel_auth
from pymc_bot.agent import ACTIONS, AgentLoop, Decision, parse_decision
from pymc_bot.backends import BackendError, available_backends, node_available
from pymc_bot.bot import MinecraftBot
from pymc_bot.config import AppSettings, ConfigStore, default_config_path
from pymc_bot.cortex import CortexError, client_from_settings
from pymc_bot.events import EventLog
from pymc_bot.fleet import AI_MODES, BotFleet, FleetError, primary_brain
from pymc_bot.knowledge import shared_knowledge
from pymc_bot.local_model import PolicyError, TrainedPolicy, export_ollama_model
from pymc_bot.mind import behaviours
from pymc_bot.mobs import CATEGORIES, DIMENSIONS, SPAWN_TIMES, is_version, load_mobs
from pymc_bot.ollama import OllamaClient, OllamaError
from pymc_bot.playtime import (
    dataset_summary,
    load_examples,
    resolve_dataset_path,
    synthesize_playtime,
)
from pymc_bot.train import TrainConfig, TrainingService, default_run_name, list_models, read_card

WEB_DIR = Path(__file__).resolve().parent / "web"
OLLAMA_HEALTH_TTL = 15.0

BRIDGE_COMMANDS = [
    "connect", "say", "command", "control", "look", "stop_motion", "pathfind_to",
    "stop_path", "dig", "eat", "attack", "jump", "swing_arm", "disconnect", "ping", "state",
    # inventory & life (pymc_bot.inventory numbers slots 0-8 hotbar, 9-35 backpack)
    "select_slot", "move_slot", "equip", "unequip", "swap_hands", "toss", "place", "use_item",
    "sleep", "wake", "collect", "attack_mob",
    # skills (pymc_bot.skills): chop_tree, craft, smelt, harvest, deposit, build_shelter...
    *skills.SKILL_COMMANDS,
]

AuthMode = Literal["offline", "microsoft"]
AiMode = Literal["off", "heuristic", "ollama", "cortex", "trained"]


# ---------------------------------------------------------------------------
# request bodies
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    # which bot speaks: a username, "selected" (default) or "all"
    bot: str | None = None


class CommandRequest(BaseModel):
    command: str = Field(min_length=1, max_length=500)
    bot: str | None = None


class ActionRequest(BaseModel):
    # The panel sends the same flat JSON the AI emits ({"action": "mine", "block": "oak_log"}):
    # fields next to "action" are parameters too.
    model_config = ConfigDict(extra="allow")

    action: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    raw: str | None = None
    bot: str | None = None


class SkillRequest(BaseModel):
    """Run one skill (``/api/skills`` lists them) on the selected bot, one by name, or all."""

    name: str = Field(min_length=1, max_length=40)
    params: dict[str, Any] = Field(default_factory=dict)
    bot: str | None = None


class AutopilotRunRequest(BaseModel):
    bot: str | None = None
    rule: str | None = None  # run this rule's plan only (default: the most urgent one)
    paused: bool | None = None  # pause / resume the autopilot for this bot instead


class BridgeSetupRequest(BaseModel):
    method: Literal["auto", "python", "bundle", "npm"] = "auto"
    node: Literal["auto", "download", "skip"] = "auto"
    full: bool = False
    force: bool = False


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=panel_auth.MAX_PASSWORD)
    remember: bool = False


class SetupRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=panel_auth.MAX_PASSWORD)
    code: str = Field(min_length=1, max_length=40)


class PasswordRequest(BaseModel):
    current: str = Field(min_length=1, max_length=panel_auth.MAX_PASSWORD)
    new: str = Field(min_length=1, max_length=panel_auth.MAX_PASSWORD)


# Reachable without logging in: the login page and what it needs.
PUBLIC_PATHS = {"/login", "/healthz", "/favicon.ico", "/api/auth/status", "/api/auth/login",
                "/api/auth/setup", "/api/auth/logout"}


def _is_public(path: str) -> bool:
    return path in PUBLIC_PATHS or path.startswith("/static/")


class KnowledgeRequest(BaseModel):
    query: str = Field(min_length=1, max_length=200)


class TrainingRequest(BaseModel):
    """Body of ``POST /api/training/start`` (everything is optional)."""

    dataset: str | None = None
    run_name: str | None = None
    engine: Literal["mlp", "transformer"] | None = None
    steps: int | None = Field(default=None, ge=1, le=2_000_000)
    epochs: int = Field(default=0, ge=0, le=10_000)
    batch_size: int | None = Field(default=None, ge=1, le=4096)
    lr: float | None = Field(default=None, gt=0.0, le=1.0)
    checkpoint_every: int | None = Field(default=None, ge=1, le=100_000)
    max_seconds: float = Field(default=0.0, ge=0.0, le=86400.0)
    hidden: list[int] | None = None
    d_model: int = Field(default=64, ge=8, le=512)
    heads: int = Field(default=4, ge=1, le=16)
    layers: int = Field(default=2, ge=1, le=8)
    reg_weight: float = Field(default=0.5, ge=0.0, le=10.0)
    include_blocks: bool | None = None
    #: Advanced training: learn entity and item vocabularies from the playtime.
    advanced: bool | None = None
    entity_slots: int | None = Field(default=None, ge=0, le=256)
    item_slots: int | None = Field(default=None, ge=0, le=512)
    resume: bool = False
    #: Generate this many minutes of synthetic playtime first (demos / tests).
    simulate_minutes: float = Field(default=0.0, ge=0.0, le=600.0)


class ActivateModelRequest(BaseModel):
    """Body of ``POST /api/models/activate``."""

    run: str = Field(min_length=1)


class ExportOllamaRequest(BaseModel):
    """Body of ``POST /api/models/export-ollama``."""

    run: str = Field(min_length=1)
    name: str | None = None
    base_model: str | None = None


class ModelRequest(BaseModel):
    model: str = Field(min_length=1, max_length=200)


class SpawnRequest(BaseModel):
    """Add one player. ``auth="microsoft"`` makes it a premium player."""

    username: str = Field(min_length=1, max_length=254)
    auth: AuthMode = "offline"
    ai: AiMode = "heuristic"
    start: bool = True


class FleetTargetRequest(BaseModel):
    """Which fleet bot an operation applies to ("all" or a username)."""

    bot: str | None = None
    remove: bool = False


class PopulateRequest(BaseModel):
    """Add many players at once (offline names, or premium emails)."""

    count: int | None = Field(default=None, ge=1, le=200)
    pattern: str | None = None
    auth: AuthMode | None = None
    ai: AiMode | None = None
    stagger: float | None = Field(default=None, ge=0.0, le=60.0)
    usernames: list[str] | None = None


# ---------------------------------------------------------------------------
# app factory
# ---------------------------------------------------------------------------
def create_app(
    config_path: Path | str | None = None,
    autostart: bool = False,
    auto_agent: bool = False,
    log: EventLog | None = None,
    bot: MinecraftBot | None = None,
    fleet: BotFleet | None = None,
    auth: panel_auth.AuthStore | None = None,
) -> FastAPI:
    store = ConfigStore(config_path)
    store.load()
    # the login (pymc_bot.auth): on unless PYMC_BOT_NO_AUTH / serve --no-auth
    auth_on = not panel_auth.auth_disabled()
    accounts = auth or panel_auth.AuthStore(panel_auth.default_auth_path(store.path))
    event_log = log or EventLog()
    bot_fleet = fleet or BotFleet(store, event_log)
    if bot is not None:  # injected bot (tests) becomes the primary player
        bot_fleet._primary = bot  # noqa: SLF001 - deliberate injection point

    def selected_bot() -> MinecraftBot:
        return bot_fleet.selected_bot()

    def selected_agent() -> AgentLoop:
        return bot_fleet.agent_for(selected_bot())

    def local_agent() -> AgentLoop:
        """The AI loop of the main bot (cached inside the fleet)."""
        return bot_fleet.primary_agent()

    def bots_for(selector: str | None) -> list[MinecraftBot]:
        try:
            return bot_fleet.action_target(selector or "selected")
        except FleetError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    training_service = TrainingService(event_log)
    # "Install mineflayer" button: runs pymc_bot.node_setup in the background (no npm needed)
    bridge_setup: dict[str, Any] = {"running": False, "ok": None, "ready": None, "message": "", "status": None}
    dataset_cache: dict[str, Any] = {"path": None, "mtime": 0.0, "summary": None}

    def models_dir() -> str:
        return store.settings.training.models_dir

    def dataset_overview(path: str | None = None) -> dict[str, Any]:
        """Cached statistics for the training dataset (kept fresh, cheap to poll)."""
        target = resolve_dataset_path(path or store.settings.training.dataset)
        mtime = 0.0
        if target.is_file():
            mtime = target.stat().st_mtime
        if (
            dataset_cache["summary"] is not None
            and dataset_cache["path"] == str(target)
            and dataset_cache["mtime"] == mtime
        ):
            return dataset_cache["summary"]
        summary: dict[str, Any] = {"dataset": str(target), "exists": target.is_file()}
        if target.is_file():
            try:
                summary.update(dataset_summary(target))
            except Exception as exc:  # pragma: no cover - unreadable dataset
                summary["error"] = str(exc)
        dataset_cache.update({"path": str(target), "mtime": mtime, "summary": summary})
        return summary

    def training_payload() -> dict[str, Any]:
        settings = store.settings.training
        status = training_service.status()
        return {
            "status": status,
            "settings": settings.model_dump(mode="json"),
            "models_dir": str(Path(settings.models_dir).expanduser()),
            "active_run": settings.active_run,
            "dataset": dataset_overview(),
            "brain": store.settings.agent.mode,
            "advanced": advanced_overview(),
        }

    def advanced_overview() -> dict[str, Any]:
        """What the newest trained model learned about entities and items."""
        settings = store.settings.training
        runs = [card for card in list_models(settings.models_dir) if card.get("checkpoint_exists")]
        wanted = (settings.active_run or "").strip()
        card = next((run for run in runs if run.get("run") == wanted), None) if wanted else None
        card = card or (runs[0] if runs else None)
        if not card:
            return {
                "available": dataset_overview().get("entities_and_items", False),
                "trained": False,
                "entities": [],
                "items": [],
            }
        entities = [str(name) for name in card.get("entity_vocabulary") or []]
        items = [str(name) for name in card.get("item_vocabulary") or []]
        return {
            "available": dataset_overview().get("entities_and_items", False),
            "trained": bool(card.get("advanced")),
            "run": card.get("run"),
            "entities": entities,
            "items": items,
            "entity_slots": len(entities),
            "item_slots": len(items),
            "coverage": {
                "entity": (card.get("dataset_summary") or {}).get("entity_coverage"),
                "item": (card.get("dataset_summary") or {}).get("item_coverage"),
            },
        }

    health_cache: dict[str, Any] = {"ts": 0.0, "ok": None, "detail": ""}

    def ollama_health(force: bool = False) -> dict[str, Any]:
        now = time.time()
        if not force and health_cache["ok"] is not None and now - health_cache["ts"] < OLLAMA_HEALTH_TTL:
            return {"ok": health_cache["ok"], "detail": health_cache["detail"], "checked_at": health_cache["ts"]}
        settings = store.settings.ollama
        try:
            client = OllamaClient(base_url=settings.base_url, model=settings.model, timeout=5.0)
            ok, detail = client.ping()
            client.close()
        except Exception as exc:  # pragma: no cover - defensive
            ok, detail = False, str(exc)
        health_cache.update({"ts": now, "ok": ok, "detail": detail})
        return {"ok": ok, "detail": detail, "checked_at": now}

    cortex_cache: dict[str, Any] = {"ts": 0.0, "ok": None, "detail": "", "key": None}

    def cortex_health(force: bool = False) -> dict[str, Any]:
        """Cached Cortex reachability (same TTL as Ollama); never exposes the key."""
        settings = store.settings.cortex
        now = time.time()
        # Re-check immediately when the endpoint/model/key settings change.
        key = (settings.base_url, settings.api_path, settings.model, settings.api_key_env, settings.key_present())
        fresh = now - cortex_cache["ts"] < OLLAMA_HEALTH_TTL and cortex_cache["key"] == key
        if force or cortex_cache["ok"] is None or not fresh:
            try:
                client = client_from_settings(settings, timeout=5.0)
                ok, detail = client.ping()
                client.close()
            except Exception as exc:  # pragma: no cover - defensive
                ok, detail = False, str(exc)
            cortex_cache.update({"ts": now, "ok": ok, "detail": detail, "key": key})
        return {
            "ok": cortex_cache["ok"],
            "detail": cortex_cache["detail"],
            "checked_at": cortex_cache["ts"],
            "enabled": settings.enabled,
            # Cortex is optional: only complain when something actually uses it.
            "in_use": settings.enabled or primary_brain(store.settings) == "cortex",
            "key_env": settings.api_key_env,
            "key_set": settings.key_present(),
        }

    def status_payload() -> dict[str, Any]:
        controller = selected_bot()
        snapshot = controller.snapshot()
        agent_status = bot_fleet.agent_for(controller).status
        return {
            "version": __version__,
            "time": time.time(),
            "config_path": str(store.path),
            "config_load_error": store.load_error,
            "bot": {
                "state": controller.state,
                **snapshot,
                # last: the snapshot's own (possibly empty) username must not win
                "username": controller.username,
            },
            "agent": agent_status,
            "stats": controller.stats,
            "fleet": bot_fleet.status(),
            "antiafk": {
                "enabled": store.settings.antiafk.enabled,
                "running": bot_fleet.antiafk.running,
                "pokes": bot_fleet.antiafk.status([])["pokes"],
            },
            "autopilot": {
                "enabled": store.settings.autopilot.enabled,
                "running": bot_fleet.autopilot.running,
                "actions": bot_fleet.autopilot.actions,
                "bot": bot_fleet.autopilot.memory(controller).to_dict(),
            },
            "bridge_setup": {key: bridge_setup[key] for key in ("running", "ok", "ready", "message")},
            "training": {
                "running": training_service.running,
                "step": training_service.status().get("step"),
                "run": training_service.status().get("run"),
                "active_run": store.settings.training.active_run,
                "brain": store.settings.agent.mode,
                "models": len(list_models(store.settings.training.models_dir)),
            },
            "backends": available_backends(),
            "ollama": {**ollama_health(), "settings": store.settings.ollama.model_dump()},
            "cortex": {**cortex_health(), "settings": store.settings.cortex.model_dump()},
            "hosting": hosting_status(),
        }

    def hosting_status() -> dict[str, Any]:
        minecraft = store.settings.minecraft
        profile = hosting.detect_host(minecraft.host)
        leave, _ = hosting.leave_rule_active(store.settings)
        return {
            "provider": profile.as_dict() if profile else None,
            "tips": hosting.host_tips(minecraft.host, minecraft.auth),
            "leave_when_alone": leave,
            "alone_minutes": minecraft.alone_minutes,
            "alone_for": round(bot_fleet.alone_watcher.alone_for, 1),
        }

    # ------------------------------------------------------------------ lifecycle
    def _autostart() -> None:
        if autostart:
            try:
                bot_fleet.primary.start()
            except Exception as exc:
                event_log.add(f"Autostart failed: {exc}", "error", "app")
        restored = bot_fleet.restore_roster()
        del restored
        if auto_agent and bot_fleet.primary.connected:
            local_agent().start()

    def setup_code() -> str:
        """No account yet: the one-time code for the panel's "create account" form (console only)."""
        if accounts.setup_code:
            return accounts.setup_code
        code = accounts.new_setup_code()
        print(f"\n  PyMC_Bot: no login yet. Open the panel and create the account with the setup code:  {code}\n"
              "  (or run: python -m pymc_bot auth set-password)\n", flush=True)
        return code

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        event_log.add(f"PyMC_Bot {__version__} control panel ready (config: {store.path}).", "info", "app")
        if not auth_on:
            event_log.add("Login is OFF (--no-auth): anyone who can reach this port controls the bots.", "warn", "app")
        elif not accounts.configured:
            setup_code()
        if store.load_error:
            event_log.add(store.load_error, "warn", "app")
        if autostart or auto_agent:
            threading.Thread(target=_autostart, name="autostart", daemon=True).start()
        try:
            yield
        finally:
            try:
                local_agent().stop(wait=False)
            except Exception:  # pragma: no cover
                pass
            bot_fleet.shutdown()

    app = FastAPI(
        title="PyMC_Bot",
        version=__version__,
        description=(
            "Control panel for a Python Minecraft player bot fleet (cracked/offline mode friendly).\n\n"
            f"Made with \u2764\ufe0f by [{__author__}]({__author_url__}) \u00b7 " + " \u00b7 ".join(f"[{site}]({site})" for site in __websites__)
        ),
        contact={"name": __author__, "url": __author_url__},
        license_info={"name": __license__, "url": "https://polyformproject.org/licenses/noncommercial/1.0.0"},
        lifespan=_lifespan,
    )
    app.state.store = store
    app.state.log = event_log
    app.state.fleet = bot_fleet
    app.state.bot = bot_fleet.primary
    app.state.agent = local_agent()
    app.state.auth = accounts

    # ------------------------------------------------------------------- login
    def session_for(cookies: Any, headers: Any) -> panel_auth.Session | None:
        if not auth_on:
            return panel_auth.Session("", float("inf"), via="disabled")
        return (accounts.verify_token(panel_auth.bearer_token(headers.get("authorization")))
                or accounts.verify_session(cookies.get(panel_auth.COOKIE_NAME)))

    @app.middleware("http")
    async def require_login(request: Request, call_next):
        session = session_for(request.cookies, request.headers)
        request.state.session = session
        path = request.url.path
        if session is not None or _is_public(path):
            return await call_next(request)
        if request.method == "GET" and not path.startswith("/api/"):
            target = path + (f"?{request.url.query}" if request.url.query else "")
            return RedirectResponse(f"/login?next={quote(target, safe='')}", status_code=303)
        return JSONResponse({"detail": "login required", "login": "/login"}, status_code=401,
                            headers={"WWW-Authenticate": 'Bearer realm="PyMC_Bot"'})

    def client_address(request: Request) -> str:
        return request.client.host if request.client else "?"

    def set_session_cookie(request: Request, response: JSONResponse, remember: bool) -> None:
        value, lifetime = accounts.issue_session(remember)
        secure = request.url.scheme == "https" or request.headers.get("x-forwarded-proto", "").startswith("https")
        response.set_cookie(panel_auth.COOKIE_NAME, value, max_age=lifetime if remember else None, httponly=True,
                            samesite="lax", secure=secure, path="/")

    def auth_status(request: Request) -> dict[str, Any]:
        session = getattr(request.state, "session", None)
        if not auth_on:
            return {"enabled": False, "configured": False, "authenticated": True, "username": "", "via": "disabled",
                    "token": False}
        configured = accounts.configured
        if not configured:
            setup_code()
        return {"enabled": True, "configured": configured, "authenticated": session is not None,
                "username": accounts.username if session else "", "via": session.via if session else None,
                "token": accounts.has_token if session else False}

    @app.get("/api/auth/status")
    def api_auth_status(request: Request) -> dict[str, Any]:
        """Is the login on, is there an account, is this browser logged in?"""
        return auth_status(request)

    @app.post("/api/auth/login")
    def api_auth_login(body: LoginRequest, request: Request) -> JSONResponse:
        if not auth_on:
            return JSONResponse({"ok": True, "detail": "login is off"})
        if not accounts.configured:
            raise HTTPException(status_code=409, detail="no account yet - create one with the setup code")
        address = client_address(request)
        try:
            accounts.login(body.username, body.password, address)
        except panel_auth.LockedOut as exc:
            raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": str(exc.seconds)}) from exc
        except panel_auth.AuthError as exc:
            event_log.add(f"Failed panel login from {address}.", "warn", "auth")
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        event_log.add(f"{accounts.username} logged in to the panel from {address}.", "info", "auth")
        response = JSONResponse({"ok": True, "username": accounts.username})
        set_session_cookie(request, response, body.remember)
        return response

    @app.post("/api/auth/setup")
    def api_auth_setup(body: SetupRequest, request: Request) -> JSONResponse:
        """First run: create the panel's account (needs the setup code from the console)."""
        if not auth_on:
            raise HTTPException(status_code=409, detail="login is off (--no-auth)")
        if accounts.configured:
            raise HTTPException(status_code=409, detail="an account already exists - log in instead")
        address = client_address(request)
        try:
            accounts.check_lockout(address)
            accounts.setup(body.username, body.password, body.code)
        except panel_auth.LockedOut as exc:
            raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": str(exc.seconds)}) from exc
        except panel_auth.AuthError as exc:
            if "setup code" in str(exc):
                accounts._record_failure(address)  # noqa: SLF001 - same throttle as logins
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        event_log.add(f"Panel account '{accounts.username}' created from {address}.", "info", "auth")
        response = JSONResponse({"ok": True, "username": accounts.username})
        set_session_cookie(request, response, False)
        return response

    @app.post("/api/auth/logout")
    def api_auth_logout() -> JSONResponse:
        response = JSONResponse({"ok": True})
        response.delete_cookie(panel_auth.COOKIE_NAME, path="/")
        return response

    def require_browser_session(request: Request) -> None:
        if not auth_on:
            raise HTTPException(status_code=409, detail="login is off (--no-auth)")
        session = getattr(request.state, "session", None)
        if session is None or session.via != "cookie":
            raise HTTPException(status_code=403, detail="log in with the password for this (an API token cannot)")

    @app.post("/api/auth/password")
    def api_auth_password(body: PasswordRequest, request: Request) -> JSONResponse:
        """Change the password; every other browser is logged out."""
        require_browser_session(request)
        try:
            accounts.change_password(body.current, body.new)
        except panel_auth.AuthError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        event_log.add("Panel password changed - other sessions were logged out.", "warn", "auth")
        response = JSONResponse({"ok": True})
        set_session_cookie(request, response, False)
        return response

    @app.post("/api/auth/token")
    def api_auth_new_token(request: Request) -> dict[str, Any]:
        """A new API token for scripts and the mod (shown once; replaces the old one)."""
        require_browser_session(request)
        token = accounts.new_token()
        event_log.add("A new panel API token was created (the old one stopped working).", "info", "auth")
        return {"ok": True, "token": token}

    @app.delete("/api/auth/token")
    def api_auth_revoke_token(request: Request) -> dict[str, Any]:
        require_browser_session(request)
        accounts.revoke_token()
        event_log.add("The panel API token was revoked.", "info", "auth")
        return {"ok": True}

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
    templates = Jinja2Templates(directory=str(WEB_DIR)) if WEB_DIR.exists() else None

    # -------------------------------------------------------------------- pages
    @app.get("/", include_in_schema=False)
    def index(request: Request):
        if templates is None:  # pragma: no cover - package without web assets
            return JSONResponse({"detail": "web assets missing", "api": "/api/status"}, status_code=500)
        return templates.TemplateResponse(request, "index.html", {"version": __version__, "about": about()})

    @app.get("/login", include_in_schema=False)
    def login_page(request: Request):
        target = request.query_params.get("next") or "/"
        if not target.startswith("/") or target.startswith("//"):
            target = "/"  # never redirect off-site
        if not auth_on or getattr(request.state, "session", None) is not None:
            return RedirectResponse(target, status_code=303)
        if templates is None:  # pragma: no cover - package without web assets
            return JSONResponse({"detail": "web assets missing"}, status_code=500)
        configured = accounts.configured
        if not configured:
            setup_code()
        return templates.TemplateResponse(request, "login.html", {"version": __version__, "setup": not configured})

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Any:
        icon = WEB_DIR / "favicon.svg"
        if icon.exists():
            return FileResponse(icon, media_type="image/svg+xml")
        return JSONResponse({}, status_code=204)

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, Any]:
        return {"ok": True, "version": __version__}

    @app.get("/api/about")
    def api_about() -> dict[str, Any]:
        """Author, links and credits (the panel's About dialog reads this)."""
        return about()

    # ------------------------------------------------------------------ status
    @app.get("/api/status")
    def api_status() -> dict[str, Any]:
        return status_payload()

    @app.get("/api/contract")
    def api_contract() -> dict[str, Any]:
        return {
            "version": __version__,
            "actions": ACTIONS,
            "bridge_commands": BRIDGE_COMMANDS,
            "auth_modes": ["offline", "microsoft"],
            "ai_modes": list(AI_MODES),
            "antiafk_habits": ["look", "look_at_player", "stroll", "strafe", "hop", "crouch", "swing", "hotbar"],
            "brains": ["auto", "heuristic", "ollama", "cortex", "trained"],
            "training_engines": ["mlp", "transformer"],
            "training_modes": ["basic", "advanced"],
            "advanced_signals": ["entities", "items", "held_item", "armor", "ground_items"],
            "player_actions": [
                "forward", "back", "left", "right", "jump", "sneak",
                "look", "attack", "use", "hold", "move", "none",
            ],
            "websocket": {"path": "/ws", "frames": ["hello", "event", "status"]},
        }

    # ------------------------------------------------------------------ config
    @app.get("/api/config")
    def get_config() -> dict[str, Any]:
        return {"config": store.settings.public_dict(), "path": str(store.path), "load_error": store.load_error}

    @app.put("/api/config")
    def put_config(patch: dict[str, Any] = Body(...)) -> dict[str, Any]:  # noqa: B008 - FastAPI idiom
        try:
            settings = store.update(patch)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors(include_url=False, include_context=False)) from exc
        event_log.add("Configuration updated from the web panel.", "info", "app")
        return {"config": settings.public_dict(), "path": str(store.path)}

    @app.post("/api/config/reset")
    def reset_config() -> dict[str, Any]:
        settings = store.save(AppSettings())
        event_log.add("Configuration reset to defaults.", "warn", "app")
        return {"config": settings.public_dict()}

    # -------------------------------------------------------------------- logs
    @app.get("/api/logs")
    def get_logs(limit: int = 200) -> dict[str, Any]:
        return {"events": event_log.tail(limit)}

    @app.post("/api/logs/clear")
    def clear_logs() -> dict[str, Any]:
        event_log.clear()
        return {"ok": True}

    # --------------------------------------------------------------------- bot
    @app.post("/api/bot/start")
    def bot_start(bot: str | None = None) -> dict[str, Any]:
        controller = selected_bot()
        if controller is bot_fleet.primary:
            if bot_fleet.primary.state in ("connected", "connecting", "waiting"):
                return {"ok": True, "state": bot_fleet.primary.state, "detail": "already running"}
            try:
                bot_fleet.primary.start()
            except BackendError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            return {"ok": True, "state": bot_fleet.primary.state, "username": bot_fleet.primary.username}
        member = bot_fleet.start_bot(controller.username)
        if member is None:
            raise HTTPException(status_code=404, detail=f"no bot named '{controller.username}' in the fleet")
        return {"ok": True, "state": member.state, "username": member.bot.username if member.bot else member.username}

    @app.post("/api/bot/stop")
    def bot_stop(bot: str | None = None) -> dict[str, Any]:
        controller = selected_bot()
        if controller is bot_fleet.primary:
            local_agent().stop(wait=False)
            bot_fleet.primary.stop()
            return {"ok": True, "state": bot_fleet.primary.state, "username": bot_fleet.primary.username}
        stopped = bot_fleet.stop_bot(controller.username)
        if not stopped:
            raise HTTPException(status_code=404, detail=f"no bot named '{controller.username}' in the fleet")
        return {"ok": True, "state": "stopped", "username": controller.username}

    @app.post("/api/bot/reconnect")
    def bot_reconnect() -> dict[str, Any]:
        controller = selected_bot()
        try:
            if controller is bot_fleet.primary:
                bot_fleet.primary.stop()
                bot_fleet.primary.start()
            else:
                bot_fleet.stop_bot(controller.username)
                bot_fleet.start_bot(controller.username)
        except BackendError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {"ok": True, "state": controller.state, "username": controller.username}

    @app.get("/api/bot/snapshot")
    def bot_snapshot() -> dict[str, Any]:
        controller = selected_bot()
        return {"state": controller.state, "username": controller.username, **controller.snapshot()}

    @app.get("/api/bot/chat")
    def bot_chat_history() -> dict[str, Any]:
        return {"messages": selected_bot().chat_history}

    @app.post("/api/bot/chat")
    def bot_chat(request: ChatRequest) -> dict[str, Any]:
        if (request.bot or "selected") == "all":
            sent = bot_fleet.broadcast(request.message)
            if not sent:
                raise HTTPException(status_code=409, detail="no connected bots to talk through")
            return {"ok": True, "sent": sent}
        bots = bots_for(request.bot)
        sent = sum(1 for bot in bots if bot.say(request.message))
        if not sent:
            raise HTTPException(status_code=409, detail="bot is not connected")
        return {"ok": True, "sent": sent}

    @app.post("/api/bot/command")
    def bot_command(request: CommandRequest) -> dict[str, Any]:
        bots = bots_for(request.bot)
        sent = sum(1 for bot in bots if bot.command(request.command))
        if not sent:
            raise HTTPException(status_code=409, detail="bot is not connected")
        return {"ok": True, "sent": sent}

    @app.post("/api/bot/action")
    def bot_action(request: ActionRequest) -> dict[str, Any]:
        decision = _decision_from_request(request)
        bots = bots_for(request.bot)
        results = []
        for bot in bots:
            if not bot.connected:
                results.append({"username": bot.username, "ok": False, "detail": "not connected"})
                continue
            ok, detail = bot_fleet.agent_for(bot).execute(decision, enforce_permissions=False)
            results.append({"username": bot.username, "ok": ok, "detail": detail})
        if not any(entry["ok"] for entry in results):
            detail = results[0]["detail"] if results else "no bots matched"
            if detail.startswith("unsupported action"):
                raise HTTPException(status_code=422, detail=detail)
        return {
            "ok": any(entry["ok"] for entry in results),
            "detail": results[0]["detail"] if len(results) == 1 else f"{sum(1 for r in results if r['ok'])}/{len(results)} bots ran it",
            "decision": decision.to_dict(),
            "results": results,
        }

    # --------------------------------------------------------------- knowledge
    def knowledge_base():
        return shared_knowledge(store.settings.search, store.path.parent)

    @app.get("/api/knowledge")
    def knowledge_status() -> dict[str, Any]:
        """Web search settings in use, cache size, requests this hour and the latest lookups."""
        return knowledge_base().status()

    @app.post("/api/knowledge/search")
    def knowledge_search(request: KnowledgeRequest) -> dict[str, Any]:
        """Look something up exactly like the AI would (cached answers are free)."""
        return knowledge_base().lookup(request.query).to_dict()

    @app.post("/api/knowledge/clear")
    def knowledge_clear() -> dict[str, Any]:
        return {"cleared": knowledge_base().clear_cache()}

    # ------------------------------------------------------------------- mob database
    @app.get("/api/mobs")
    def mobs_list(category: str | None = None, hostile: bool | None = None, dimension: str | None = None,
                  spawn_time: str | None = None, added_since: str | None = None, exists_in: str | None = None,
                  q: str | None = None, full: bool = False) -> dict[str, Any]:
        """The mob database (data/mobs.json), filterable. ``full=true`` returns whole entries."""
        if category is not None and category not in CATEGORIES:
            raise HTTPException(422, f"category must be one of {', '.join(CATEGORIES)}")
        if dimension is not None and dimension not in DIMENSIONS:
            raise HTTPException(422, f"dimension must be one of {', '.join(DIMENSIONS)}")
        if spawn_time is not None and spawn_time not in SPAWN_TIMES:
            raise HTTPException(422, f"spawn_time must be one of {', '.join(SPAWN_TIMES)}")
        for name, value in (("added_since", added_since), ("exists_in", exists_in)):
            if value is not None and not is_version(value):
                raise HTTPException(422, f"{name} must be a version like 1.21.4 or 26.2")
        db = load_mobs()
        found = db.search(category=category, hostile=hostile, dimension=dimension, spawn_time=spawn_time,
                          added_since=added_since, exists_in=exists_in, query=q)
        if full:
            mobs = [db.with_derived(m) for m in found]
        else:
            mobs = [{"id": m["id"], "name": m["name"], "category": m["category"], "hostile": m["hostile"],
                     "health": m["health"], "max_damage_normal": db.max_damage(m), "spawn_time": m["spawn"]["when"],
                     "added_in": m["added_in"], "mineflayer_supported": db.supported_by_mineflayer(m)}
                    for m in found]
        return {"meta": db.meta, "count": len(mobs), "mobs": mobs}

    @app.get("/api/mobs/{name}")
    def mob_entry(name: str) -> dict[str, Any]:
        """One mob by id, name, alias or plural ("sulphur cube", "zombie pigman"), plus a text summary."""
        db = load_mobs()
        mob = db.get(name)
        if mob is None:
            raise HTTPException(404, f"no mob called {name!r} in the database")
        return {**db.with_derived(mob), "summary": db.describe(mob)}

    # ------------------------------------------------------------------- fleet
    @app.get("/api/fleet/status")
    def fleet_status() -> dict[str, Any]:
        return bot_fleet.status()

    @app.post("/api/fleet/spawn")
    def fleet_spawn(request: SpawnRequest) -> dict[str, Any]:
        """Add one player: offline (any name) or premium (Microsoft email)."""
        try:
            member = bot_fleet.spawn(
                request.username, auth=request.auth, ai=request.ai, sponsor="premium" if request.auth == "microsoft" else "custom",
                start=request.start,
            )
        except FleetError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        entry = next(
            (bot for bot in bot_fleet.status()["bots"] if bot["configured_username"] == member.username),
            None,
        )
        return {"ok": True, "member": entry}

    @app.post("/api/fleet/populate")
    def fleet_populate(request: PopulateRequest) -> dict[str, Any]:
        """Add many players at once to fill the server."""
        try:
            if request.usernames:
                if len(request.usernames) > 200:
                    raise FleetError("at most 200 players per request")
                members = [
                    bot_fleet.spawn(
                        name,
                        auth=request.auth or store.settings.fleet.auth,
                        ai=request.ai or store.settings.fleet.ai_mode,
                        sponsor="populate",
                    )
                    for name in request.usernames
                ]
            else:
                members = bot_fleet.spawn_many(
                    count=request.count,
                    pattern=request.pattern,
                    auth=request.auth,
                    ai=request.ai,
                    stagger=request.stagger,
                )
        except FleetError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "ok": True,
            "queued": [member.username for member in members],
            "count": len(members),
            "fleet": bot_fleet.status(),
            "antiafk": {
                "enabled": store.settings.antiafk.enabled,
                "running": bot_fleet.antiafk.running,
                "pokes": bot_fleet.antiafk.status([])["pokes"],
            },
        }

    @app.post("/api/fleet/start")
    def fleet_start(request: FleetTargetRequest = FleetTargetRequest()) -> dict[str, Any]:  # noqa: B008
        if request.bot and request.bot not in ("all", "selected"):
            member = bot_fleet.start_bot(request.bot)
            if member is None:
                raise HTTPException(status_code=404, detail=f"no bot named '{request.bot}' in the fleet")
            return {"ok": True, "started": 1, "fleet": bot_fleet.status()}
        started = bot_fleet.start_all()
        return {"ok": True, "started": started, "fleet": bot_fleet.status()}

    @app.post("/api/fleet/stop")
    def fleet_stop(request: FleetTargetRequest = FleetTargetRequest()) -> dict[str, Any]:  # noqa: B008
        if request.bot and request.bot not in ("all", "selected"):
            done = (
                bot_fleet.remove_bot(request.bot)
                if request.remove
                else bot_fleet.stop_bot(request.bot)
            )
            if not done:
                raise HTTPException(status_code=404, detail=f"no bot named '{request.bot}' in the fleet")
            return {"ok": True, "fleet": bot_fleet.status()}
        stopped = bot_fleet.stop_all(remove=request.remove)
        return {"ok": True, "stopped": stopped, "fleet": bot_fleet.status()}

    @app.post("/api/fleet/remove")
    def fleet_remove(request: FleetTargetRequest) -> dict[str, Any]:
        if not request.bot or not bot_fleet.remove_bot(request.bot):
            raise HTTPException(status_code=404, detail=f"no bot named '{request.bot}' in the fleet")
        return {"ok": True, "fleet": bot_fleet.status()}

    @app.post("/api/fleet/select")
    def fleet_select(request: FleetTargetRequest) -> dict[str, Any]:
        try:
            username = bot_fleet.select(request.bot)
        except FleetError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"ok": True, "selected": username}

    @app.post("/api/fleet/broadcast")
    def fleet_broadcast(request: ChatRequest) -> dict[str, Any]:
        sent = bot_fleet.broadcast(request.message)
        if not sent:
            raise HTTPException(status_code=409, detail="no connected bots to talk through")
        return {"ok": True, "sent": sent}

    @app.get("/api/fleet/roster")
    def fleet_roster() -> dict[str, Any]:
        return {"roster": [spec.model_dump() for spec in store.settings.fleet.roster]}

    @app.delete("/api/fleet/roster")
    def fleet_roster_clear() -> dict[str, Any]:
        bot_fleet.clear_roster()
        return {"ok": True, "roster": []}

    # ---------------------------------------------------------------------- ai
    @app.get("/api/ai/status")
    def ai_status() -> dict[str, Any]:
        return selected_agent().status

    @app.get("/api/ai/behaviours")
    def ai_behaviours() -> dict[str, Any]:
        """How the LLM bot can behave: names and short descriptions only (the prompts stay hidden)."""
        return {"selected": store.settings.agent.persona, "behaviours": behaviours()}

    @app.post("/api/ai/start")
    def ai_start() -> dict[str, Any]:
        controller = selected_bot()
        if not controller.connected:
            raise HTTPException(status_code=409, detail="Connect the bot to a server first.")
        agent = bot_fleet.agent_for(controller)
        started = agent.start()
        return {"ok": True, "started": started, "username": controller.username, **agent.status}

    @app.post("/api/ai/stop")
    def ai_stop() -> dict[str, Any]:
        controller = selected_bot()
        agent = bot_fleet.agent_for(controller)
        stopped = agent.stop()
        return {"ok": True, "stopped": stopped, "username": controller.username, **agent.status}

    @app.post("/api/ai/step")
    def ai_step() -> dict[str, Any]:
        controller = selected_bot()
        if not controller.connected:
            raise HTTPException(status_code=409, detail="Connect the bot to a server first.")
        agent = bot_fleet.agent_for(controller)
        ok, detail = agent.run_once()
        return {"ok": ok, "detail": detail, "decision": agent.status["last_decision"]}

    # ---------------------------------------------------------------- antiafk
    @app.get("/api/antiafk/status")
    def antiafk_status() -> dict[str, Any]:
        return bot_fleet.antiafk.status(bot_fleet.all_bots())

    @app.post("/api/antiafk/poke")
    def antiafk_poke(request: FleetTargetRequest = FleetTargetRequest()) -> dict[str, Any]:  # noqa: B008
        """Force an anti-AFK burst right now (selected bot, one by name, or all)."""
        try:
            bots = bot_fleet.action_target(request.bot or "all")
        except FleetError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if len(bots) == 1:
            bot = bots[0]
            habits = bot_fleet.antiafk.poke_bot(bot)
            bot_fleet.log.add(
                f"Anti-AFK poke: {bot.username} did {', '.join(habits) or 'nothing'}.",
                "info",
                "antiafk",
            )
            return {"ok": bool(habits), "results": [{"username": bot.username, "habits": habits, "poked": bool(habits)}]}

        # A crowd is poked in the background so the request returns immediately:
        # each burst can last a few seconds, and 200 of them would time out.
        def _poke_all() -> None:
            poked = 0
            for bot in bots:
                if bot.connected and bot_fleet.antiafk.poke_bot(bot):
                    poked += 1
            bot_fleet.log.add(f"Anti-AFK poke: {poked}/{len(bots)} bots moved.", "info", "antiafk")

        threading.Thread(target=_poke_all, name="antiafk-poke-all", daemon=True).start()
        return {"ok": True, "queued": len(bots), "results": []}

    # ------------------------------------------------------------------ cortex
    @app.get("/api/server/probe")
    def server_probe(host: str | None = None, port: int | None = None) -> dict[str, Any]:
        """Ping a Minecraft server without joining: online/asleep, version, players, host tips."""
        minecraft = store.settings.minecraft
        result = hosting.probe_server(host or minecraft.host, port or (minecraft.port if not host else 25565))
        result["summary"] = hosting.describe_probe(result)
        result["tips"] = hosting.host_tips(result["host"], minecraft.auth)
        return result

    @app.get("/api/cortex/health")
    def cortex_health_endpoint(force: bool = False) -> dict[str, Any]:
        return cortex_health(force=force)

    @app.get("/api/cortex/models")
    def cortex_models() -> dict[str, Any]:
        """Models configured in Cortex LLMHoster, with their declared capabilities."""
        client = client_from_settings(store.settings.cortex, timeout=10.0)
        try:
            models = client.list_models()
        except CortexError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        finally:
            client.close()
        return {
            "models": [str(entry.get("id")) for entry in models],
            "details": [
                {
                    "id": entry.get("id"),
                    "runtime": entry.get("runtime"),
                    "capabilities": entry.get("capabilities") or [],
                    "chat": not entry.get("capabilities") or "text_generation" in (entry.get("capabilities") or []),
                }
                for entry in models
            ],
        }

    # ------------------------------------------------------------------ ollama
    @app.get("/api/ollama/health")
    def ollama_health_endpoint(force: bool = False) -> dict[str, Any]:
        return ollama_health(force=force)

    @app.get("/api/ollama/models")
    def ollama_models() -> dict[str, Any]:
        settings = store.settings.ollama
        client = OllamaClient(base_url=settings.base_url, model=settings.model, timeout=10.0)
        try:
            models = client.list_models()
        except OllamaError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        finally:
            client.close()
        return {"models": [m.get("name") for m in models]}

    @app.post("/api/ollama/pull")
    def ollama_pull(request: ModelRequest) -> dict[str, Any]:
        settings = store.settings.ollama
        client = OllamaClient(base_url=settings.base_url, model=request.model, timeout=max(600.0, settings.request_timeout))
        event_log.add(f"Pulling Ollama model '{request.model}' (this can take a while)...", "info", "ollama")
        try:
            client.pull(request.model)
        except OllamaError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        finally:
            client.close()
        event_log.add(f"Ollama model '{request.model}' is ready.", "success", "ollama")
        return {"ok": True, "model": request.model}

    # --------------------------------------------------------------- training
    @app.get("/api/training/status")
    def training_status() -> dict[str, Any]:
        return training_payload()

    @app.get("/api/training/dataset")
    def training_dataset(dataset: str | None = None) -> dict[str, Any]:
        return dataset_overview(dataset)

    @app.post("/api/training/start")
    def training_start(request: TrainingRequest) -> dict[str, Any]:
        if training_service.running:
            raise HTTPException(status_code=409, detail="training is already running")
        settings = store.settings.training
        dataset = (request.dataset or settings.dataset).strip()
        if not request.simulate_minutes:
            dataset = str(resolve_dataset_path(dataset))
        advanced = settings.advanced if request.advanced is None else bool(request.advanced)

        if request.simulate_minutes and request.simulate_minutes > 0:
            info = synthesize_playtime(dataset, minutes=request.simulate_minutes, advanced=advanced)
            event_log.add(
                f"Generated {info['samples']} synthetic playtime samples for the demo "
                f"({info['minutes']} min of pretend play).",
                "info",
                "train",
            )
        if not Path(dataset).expanduser().is_file():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"no playtime dataset at {dataset} - record some with the Fabric mod in ./mod "
                    "(/pymcbot record start, /pymcbot export), or ask for a synthetic one with "
                    '{"simulate_minutes": 5}'
                ),
            )

        config = TrainConfig(
            dataset=dataset,
            run_name=(request.run_name or default_run_name(dataset)),
            models_dir=settings.models_dir,
            engine=request.engine or settings.engine,
            hidden=tuple(request.hidden or settings.hidden),
            d_model=request.d_model,
            layers=request.layers,
            heads=request.heads,
            batch_size=request.batch_size or settings.batch_size,
            lr=request.lr or settings.lr,
            steps=request.steps or settings.steps,
            epochs=request.epochs,
            max_seconds=request.max_seconds,
            checkpoint_every=request.checkpoint_every or settings.checkpoint_every,
            val_split=settings.val_split,
            reg_weight=request.reg_weight,
            include_blocks=settings.include_blocks if request.include_blocks is None else request.include_blocks,
            advanced=advanced,
            entity_slots=request.entity_slots if request.entity_slots is not None else settings.entity_slots,
            item_slots=request.item_slots if request.item_slots is not None else settings.item_slots,
            resume=request.resume,
        )
        if not training_service.start(config):
            raise HTTPException(status_code=409, detail="training is already running")
        # Remember what the panel asked for, so a restart keeps the same setup.
        store.update(
            {
                "training": {
                    "dataset": dataset,
                    "engine": config.engine,
                    "steps": config.steps,
                    "batch_size": config.batch_size,
                    "checkpoint_every": config.checkpoint_every,
                    "active_run": config.run_name,
                    "advanced": config.advanced,
                    "entity_slots": config.entity_slots,
                    "item_slots": config.item_slots,
                }
            }
        )
        return {"ok": True, "run": config.run_name, "config": config.to_dict(), **training_payload()}

    @app.post("/api/training/stop")
    def training_stop() -> dict[str, Any]:
        stopped = training_service.stop(wait=False)
        if not stopped:
            raise HTTPException(status_code=409, detail="training is not running")
        return {"ok": True, "stopping": True}

    # ----------------------------------------------------------------- models
    @app.get("/api/models")
    def models_list() -> dict[str, Any]:
        settings = store.settings.training
        return {
            "models_dir": str(Path(settings.models_dir).expanduser()),
            "active_run": settings.active_run,
            "brain": store.settings.agent.mode,
            "models": list_models(settings.models_dir),
        }

    @app.get("/api/models/{run}")
    def model_detail(run: str) -> dict[str, Any]:
        card = read_card(models_dir(), run)
        if card is None:
            raise HTTPException(status_code=404, detail=f"no trained model '{run}'")
        return card

    @app.post("/api/models/activate")
    def model_activate(request: ActivateModelRequest) -> dict[str, Any]:
        card = read_card(models_dir(), request.run)
        if card is None:
            raise HTTPException(status_code=404, detail=f"no trained model '{request.run}'")
        store.update({"training": {"active_run": request.run}, "agent": {"mode": "trained"}})
        # Restart the AI loop so it picks the model up immediately.
        agent = selected_agent()
        was_running = agent.running
        if was_running:
            agent.stop(wait=True)
            selected_agent().start()
        policy = None
        try:
            policy = selected_agent().policy(store.settings)
        except Exception as exc:  # pragma: no cover - defensive
            event_log.add(f"Could not preload the trained model: {exc}", "warn", "train")
        event_log.add(
            f"Activated player model '{request.run}' "
            f"({card.get('engine')}, step {card.get('step')}, {card.get('params')} params) - "
            f"brain: trained{'' if was_running else ' (start the AI loop to play)'}.",
            "success",
            "train",
        )
        return {
            "ok": True,
            "active_run": request.run,
            "brain": store.settings.agent.mode,
            "agent_running": selected_agent().running,
            "policy": policy.describe() if policy is not None else None,
        }

    @app.post("/api/models/deactivate")
    def model_deactivate() -> dict[str, Any]:
        store.update({"agent": {"mode": "auto"}})
        event_log.add("Brain back to auto (Ollama when enabled, else heuristics).", "info", "train")
        return {"ok": True, "brain": store.settings.agent.mode}

    @app.get("/api/policy/preview")
    def policy_preview(run: str | None = None) -> dict[str, Any]:
        """What would the trained model do right now, given the live bot snapshot?"""
        settings = store.settings.training
        try:
            policy = TrainedPolicy.load(
                run or settings.active_run or "",
                settings.models_dir,
                temperature=0.0,
                step_seconds=settings.step_seconds,
            )
            snapshot = selected_bot().snapshot()
            prediction = policy.predict(snapshot)
        except PolicyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive (torch missing, bad checkpoint...)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "ok": True,
            "run": policy.run.run,
            "prediction": prediction.to_dict(),
            "observation": {
                "position": snapshot.get("position"),
                "yaw": snapshot.get("yaw"),
                "pitch": snapshot.get("pitch"),
                "health": snapshot.get("health"),
                "food": snapshot.get("food"),
                "players": len(snapshot.get("players") or []),
            },
            "policy": policy.describe(),
        }

    @app.post("/api/models/export-ollama")
    def model_export_ollama(request: ExportOllamaRequest) -> dict[str, Any]:
        settings = store.settings
        try:
            policy = TrainedPolicy.load(request.run, settings.training.models_dir)
        except PolicyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        name = (request.name or f"pymc-{request.run}").strip()
        client = OllamaClient(
            base_url=settings.ollama.base_url,
            model=settings.ollama.model,
            timeout=max(120.0, settings.ollama.request_timeout),
        )
        try:
            result = export_ollama_model(
                policy,
                name=name,
                client=client,
                base_model=request.base_model or settings.ollama.model,
                observations=[
                    observation
                    for observation, _ in load_examples(settings.training.dataset, limit=200)
                ],
            )
        except OllamaError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        finally:
            client.close()
        event_log.add(
            f"Exported player model '{request.run}' to Ollama as '{name}' ({result['modelfile']}).",
            "success",
            "train",
        )
        return {"ok": True, "model": name, "modelfile": result["modelfile"], "priors": result["priors"]["priorities"]}

    # ---------------------------------------------------------------- backends
    def _bridge_status() -> dict[str, Any]:
        try:
            status = node_setup.bridge_status()
        except Exception as exc:  # pragma: no cover - a status check never breaks the panel
            status = {"ready": False, "error": str(exc)}
        bridge_setup["status"] = status
        bridge_setup["ready"] = bool(status.get("ready"))
        return status

    @app.get("/api/backends")
    def backends() -> dict[str, Any]:
        ok, reason = node_available()
        return {"backends": available_backends(), "node_ready": ok, "reason": reason, "setup": _bridge_status(),
                "installing": bridge_setup["running"]}

    @app.get("/api/setup/bridge")
    def setup_bridge_status() -> dict[str, Any]:
        return {**{k: v for k, v in bridge_setup.items() if k != "status"}, "status": _bridge_status()}

    @app.post("/api/setup/bridge")
    def setup_bridge(request: BridgeSetupRequest = BridgeSetupRequest()) -> dict[str, Any]:  # noqa: B008
        """Install Node.js (portable, if missing) and mineflayer - without npm - in the background."""
        if bridge_setup["running"]:
            raise HTTPException(status_code=409, detail="The installer is already running.")
        bridge_setup.update(running=True, ok=None, message="installing...")

        def say(message: str) -> None:
            event_log.add(str(message), "info", "setup")

        def work() -> None:
            try:
                report = node_setup.setup(request.method, node=request.node, slim=not request.full,
                                          force=request.force, log=say)
                bridge_setup.update(ok=True, message=f"mineflayer installed ({report.get('method') or request.method})")
                event_log.add("mineflayer is ready - pick the real-server backend and connect.", "info", "setup")
            except Exception as exc:  # SetupError, network trouble...
                bridge_setup.update(ok=False, message=str(exc))
                event_log.add(f"Installing mineflayer failed: {exc}", "error", "setup")
            finally:
                bridge_setup["running"] = False
                _bridge_status()

        threading.Thread(target=work, name="bridge-setup", daemon=True).start()
        event_log.add(f"Installing mineflayer (method {request.method}, no npm needed)...", "info", "setup")
        return {"ok": True, "running": True}

    # ------------------------------------------------------------------ skills
    @app.get("/api/skills")
    def skill_list() -> dict[str, Any]:
        return {"skills": skills.catalogue(), "groups": list(skills.GROUPS), "count": len(skills.SKILLS)}

    @app.post("/api/bot/skill")
    def bot_skill(request: SkillRequest) -> dict[str, Any]:
        skill = skills.get(request.name)
        if skill is None:
            raise HTTPException(status_code=404, detail=f"unknown skill {request.name!r}")
        bots = bots_for(request.bot)
        results = []
        for bot in bots:
            if not bot.connected:
                results.append({"username": bot.username, "ok": False, "detail": "not connected"})
                continue
            ok, detail = bot.skill(skill.name, request.params)
            results.append({"username": bot.username, "ok": ok, "detail": detail})
        ok = any(entry["ok"] for entry in results)
        detail = results[0]["detail"] if len(results) == 1 else f"{sum(1 for r in results if r['ok'])}/{len(results)} bots did it"
        return {"ok": ok, "skill": skill.name, "detail": detail if results else "no bots matched", "results": results}

    # --------------------------------------------------------------- autopilot
    @app.get("/api/autopilot/status")
    def autopilot_status() -> dict[str, Any]:
        status = bot_fleet.autopilot.status(bot_fleet.all_bots())
        controller = selected_bot()
        plan = bot_fleet.autopilot.plan_for(controller) if controller.connected else None
        status["next"] = plan.to_dict() if plan else None
        status["settings"] = store.settings.autopilot.model_dump(mode="json")
        status["rule_order"] = list(autopilot.RULES)
        return status

    @app.post("/api/autopilot/run")
    def autopilot_run(request: AutopilotRunRequest = AutopilotRunRequest()) -> dict[str, Any]:  # noqa: B008
        """Do the most urgent reflex now (or pause/resume the autopilot for a bot)."""
        bots = bots_for(request.bot)
        results = []
        for bot in bots:
            if request.paused is not None:
                bot_fleet.autopilot.pause(bot, request.paused)
                results.append({"username": bot.username, "ok": True, "paused": request.paused})
                continue
            if not bot.connected:
                results.append({"username": bot.username, "ok": False, "detail": "not connected"})
                continue
            situation = bot_fleet.autopilot.situation(bot)
            situation.cooldowns = {}
            situation.busy = False
            settings = store.settings.autopilot.model_copy(update={"enabled": True})
            if request.rule:
                if request.rule not in autopilot.RULES:
                    raise HTTPException(status_code=422, detail=f"unknown rule {request.rule!r}")
                plan = autopilot.RULE_FUNCS[request.rule](settings, situation)
            else:
                plan = autopilot.evaluate(settings, situation)
            if plan is None:
                results.append({"username": bot.username, "ok": False, "detail": "nothing to do right now"})
                continue
            ok, detail = bot_fleet.autopilot.run(bot, plan)
            results.append({"username": bot.username, "ok": ok, "detail": detail, "plan": plan.to_dict()})
        return {"ok": any(r["ok"] for r in results), "results": results}

    # --------------------------------------------------------------- websocket
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        if session_for(websocket.cookies, websocket.headers) is None:
            await websocket.close(code=4401)  # not logged in (before accept: the handshake gets HTTP 403)
            return
        await websocket.accept()
        loop = asyncio.get_running_loop()
        sub_id, queue = event_log.subscribe(loop)
        try:
            await websocket.send_json({"type": "hello", "data": {"version": __version__, "config": str(store.path)}})
            await websocket.send_json({"type": "status", "data": status_payload()})
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.5)
                except asyncio.TimeoutError:
                    await websocket.send_json({"type": "status", "data": status_payload()})
                    continue
                await websocket.send_json({"type": "event", "data": event.to_dict()})
        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            pass
        except Exception as exc:  # pragma: no cover - unexpected socket issues
            event_log.add(f"WebSocket closed: {exc}", "debug", "app")
        finally:
            event_log.unsubscribe(sub_id)

    return app


def _decision_from_request(request: ActionRequest) -> Decision:
    if request.raw:
        try:
            return parse_decision(request.raw, source="manual")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not request.action:
        raise HTTPException(status_code=422, detail="Provide either 'action' (+ params) or 'raw'.")
    action = request.action.strip().lower()
    if action not in ACTIONS and skills.get(action) is None:
        raise HTTPException(
            status_code=422,
            detail=f"unknown action '{action}'. Valid actions: {', '.join(sorted(ACTIONS))}",
        )
    params = {**(request.model_extra or {}), **(request.params or {})}
    return Decision(action=action, params=params, source="manual")


def config_path_for_cli(cli_path: str | None) -> Path:
    return Path(cli_path).expanduser() if cli_path else default_config_path()
