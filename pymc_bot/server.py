"""FastAPI + Uvicorn config page and REST/WebSocket API for PyMC_Bot."""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ValidationError
from starlette.requests import Request

from pymc_bot import __version__
from pymc_bot.agent import ACTIONS, AgentLoop, Decision, parse_decision
from pymc_bot.backends import BackendError, available_backends, node_available
from pymc_bot.bot import MinecraftBot
from pymc_bot.config import AppSettings, ConfigStore, default_config_path
from pymc_bot.events import EventLog
from pymc_bot.fleet import AI_MODES, BotFleet, FleetError
from pymc_bot.local_model import PolicyError, TrainedPolicy, export_ollama_model
from pymc_bot.ollama import OllamaClient, OllamaError
from pymc_bot.playtime import dataset_summary, load_examples, synthesize_playtime
from pymc_bot.train import TrainConfig, TrainingService, default_run_name, list_models, read_card

WEB_DIR = Path(__file__).resolve().parent / "web"
OLLAMA_HEALTH_TTL = 15.0

BRIDGE_COMMANDS = [
    "connect", "say", "command", "control", "look", "stop_motion", "pathfind_to",
    "stop_path", "dig", "eat", "attack", "jump", "swing_arm", "disconnect", "ping", "state",
]

AuthMode = Literal["offline", "microsoft"]
AiMode = Literal["off", "heuristic", "ollama"]


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
    action: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    raw: str | None = None
    bot: str | None = None


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
) -> FastAPI:
    store = ConfigStore(config_path)
    store.load()
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
    dataset_cache: dict[str, Any] = {"path": None, "mtime": 0.0, "summary": None}

    def models_dir() -> str:
        return store.settings.training.models_dir

    def dataset_overview(path: str | None = None) -> dict[str, Any]:
        """Cached statistics for the training dataset (kept fresh, cheap to poll)."""
        target = Path(path or store.settings.training.dataset).expanduser()
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

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        event_log.add(f"PyMC_Bot {__version__} control panel ready (config: {store.path}).", "info", "app")
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
        description="Control panel for a Python Minecraft player bot fleet (cracked/offline mode friendly).",
        lifespan=_lifespan,
    )
    app.state.store = store
    app.state.log = event_log
    app.state.fleet = bot_fleet
    app.state.bot = bot_fleet.primary
    app.state.agent = local_agent()

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
    templates = Jinja2Templates(directory=str(WEB_DIR)) if WEB_DIR.exists() else None

    # -------------------------------------------------------------------- pages
    @app.get("/", include_in_schema=False)
    def index(request: Request):
        if templates is None:  # pragma: no cover - package without web assets
            return JSONResponse({"detail": "web assets missing", "api": "/api/status"}, status_code=500)
        return templates.TemplateResponse(request, "index.html", {"version": __version__})

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Any:
        icon = WEB_DIR / "favicon.svg"
        if icon.exists():
            return FileResponse(icon, media_type="image/svg+xml")
        return JSONResponse({}, status_code=204)

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, Any]:
        return {"ok": True, "version": __version__}

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
            "antiafk_habits": ["look", "look_at_player", "stroll", "strafe", "hop", "crouch", "swing"],
            "brains": ["auto", "heuristic", "ollama", "trained"],
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
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
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
            if bot_fleet.primary.state in ("connected", "connecting"):
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
                    "(/pymc record start, /pymc export), or ask for a synthetic one with "
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
    @app.get("/api/backends")
    def backends() -> dict[str, Any]:
        ok, reason = node_available()
        return {"backends": available_backends(), "node_ready": ok, "reason": reason}

    # --------------------------------------------------------------- websocket
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
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
    if action not in ACTIONS:
        raise HTTPException(
            status_code=422,
            detail=f"unknown action '{action}'. Valid actions: {', '.join(sorted(ACTIONS))}",
        )
    return Decision(action=action, params=dict(request.params or {}), source="manual")


def config_path_for_cli(cli_path: str | None) -> Path:
    return Path(cli_path).expanduser() if cli_path else default_config_path()
