"""FastAPI + Uvicorn config page and REST/WebSocket API for PyMC_Bot."""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

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
from pymc_bot.ollama import OllamaClient, OllamaError

WEB_DIR = Path(__file__).resolve().parent / "web"
OLLAMA_HEALTH_TTL = 15.0

BRIDGE_COMMANDS = [
    "connect", "say", "command", "control", "look", "stop_motion", "pathfind_to",
    "stop_path", "dig", "eat", "attack", "jump", "disconnect", "ping", "state",
]

# ---------------------------------------------------------------------------
# request bodies
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)


class CommandRequest(BaseModel):
    command: str = Field(min_length=1, max_length=500)


class ActionRequest(BaseModel):
    action: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    raw: str | None = None


class ModelRequest(BaseModel):
    model: str = Field(min_length=1, max_length=200)


# ---------------------------------------------------------------------------
# app factory
# ---------------------------------------------------------------------------
def create_app(
    config_path: Path | str | None = None,
    autostart: bool = False,
    auto_agent: bool = False,
    log: EventLog | None = None,
    bot: MinecraftBot | None = None,
) -> FastAPI:
    store = ConfigStore(config_path)
    store.load()
    event_log = log or EventLog()
    controller = bot or MinecraftBot(store, event_log)
    agent = AgentLoop(controller, store, event_log)

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
        snapshot = controller.snapshot()
        return {
            "version": __version__,
            "time": time.time(),
            "config_path": str(store.path),
            "config_load_error": store.load_error,
            "bot": {"state": controller.state, **snapshot},
            "agent": agent.status,
            "stats": controller.stats,
            "backends": available_backends(),
            "ollama": {**ollama_health(), "settings": store.settings.ollama.model_dump()},
        }

    # ------------------------------------------------------------------ lifecycle
    def _autostart() -> None:
        if autostart:
            try:
                controller.start()
            except Exception as exc:
                event_log.add(f"Autostart failed: {exc}", "error", "app")
        if auto_agent and controller.state == "connected":
            agent.start()

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
            agent.stop(wait=False)
            controller.stop()

    app = FastAPI(
        title="PyMC_Bot",
        version=__version__,
        description="Control panel for a Python Minecraft player bot (cracked/offline mode friendly).",
        lifespan=_lifespan,
    )
    app.state.store = store
    app.state.log = event_log
    app.state.bot = controller
    app.state.agent = agent

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
    def bot_start() -> dict[str, Any]:
        if controller.state in ("connected", "connecting"):
            return {"ok": True, "state": controller.state, "detail": "already running"}
        try:
            controller.start()
        except BackendError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"ok": True, "state": controller.state}

    @app.post("/api/bot/stop")
    def bot_stop() -> dict[str, Any]:
        agent.stop(wait=False)
        controller.stop()
        return {"ok": True, "state": controller.state}

    @app.post("/api/bot/reconnect")
    def bot_reconnect() -> dict[str, Any]:
        try:
            controller.stop()
            controller.start()
        except BackendError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {"ok": True, "state": controller.state}

    @app.get("/api/bot/snapshot")
    def bot_snapshot() -> dict[str, Any]:
        return {"state": controller.state, **controller.snapshot()}

    @app.get("/api/bot/chat")
    def bot_chat_history() -> dict[str, Any]:
        return {"messages": controller.chat_history}

    @app.post("/api/bot/chat")
    def bot_chat(request: ChatRequest) -> dict[str, Any]:
        ok = controller.say(request.message)
        if not ok:
            raise HTTPException(status_code=409, detail="bot is not connected")
        return {"ok": True}

    @app.post("/api/bot/command")
    def bot_command(request: CommandRequest) -> dict[str, Any]:
        ok = controller.command(request.command)
        if not ok:
            raise HTTPException(status_code=409, detail="bot is not connected")
        return {"ok": True}

    @app.post("/api/bot/action")
    def bot_action(request: ActionRequest) -> dict[str, Any]:
        decision = _decision_from_request(request)
        ok, detail = agent.execute(decision, enforce_permissions=False)
        if not ok and detail.startswith("unsupported action"):
            raise HTTPException(status_code=422, detail=detail)
        return {"ok": ok, "detail": detail, "decision": decision.to_dict()}

    # ---------------------------------------------------------------------- ai
    @app.get("/api/ai/status")
    def ai_status() -> dict[str, Any]:
        return agent.status

    @app.post("/api/ai/start")
    def ai_start() -> dict[str, Any]:
        if not controller.connected:
            raise HTTPException(status_code=409, detail="Connect the bot to a server first.")
        started = agent.start()
        return {"ok": True, "started": started, **agent.status}

    @app.post("/api/ai/stop")
    def ai_stop() -> dict[str, Any]:
        stopped = agent.stop()
        return {"ok": True, "stopped": stopped, **agent.status}

    @app.post("/api/ai/step")
    def ai_step() -> dict[str, Any]:
        if not controller.connected:
            raise HTTPException(status_code=409, detail="Connect the bot to a server first.")
        ok, detail = agent.run_once()
        return {"ok": ok, "detail": detail, "decision": agent.status["last_decision"]}

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
