"""Command line entry point: ``python -m pymc_bot [serve|run|action|doctor|stub]``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from pymc_bot import __version__
from pymc_bot.backends import BackendError, available_backends, node_available
from pymc_bot.config import ConfigStore, default_config_path
from pymc_bot.events import EventLog
from pymc_bot.fleet import BotFleet, FleetError
from pymc_bot.ollama import OllamaClient


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=None,
        help=f"path to the JSON config file (default: {default_config_path()})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pymc-bot",
        description="PyMC_Bot - a Python-controlled Minecraft player bot with an Ollama brain and a web config page.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"PyMC_Bot {__version__}")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="run the web config panel (Uvicorn)")
    _add_config_arg(serve)
    serve.add_argument("--host", default=None, help="bind address (default: from config, usually 0.0.0.0)")
    serve.add_argument("--port", type=int, default=None, help="port (default: from config, usually 8000)")
    serve.add_argument("--autostart", action="store_true", help="connect the bot as soon as the panel starts")
    serve.add_argument("--auto-agent", action="store_true", help="also start the AI loop on startup")
    serve.add_argument("--log-level", default="info", choices=["critical", "error", "warning", "info", "debug", "trace"])
    serve.add_argument("--reload", action="store_true", help="auto-reload on code changes (development)")

    run = sub.add_parser("run", help="headless: connect the bot and play (optionally with the AI loop)")
    _add_config_arg(run)
    run.add_argument("--no-ai", action="store_true", help="connect only, do not start the AI loop")
    run.add_argument("--backend", choices=["auto", "node", "simulated"], default=None, help="override the backend")
    run.add_argument("--server", default=None, help="override host:port, e.g. play.example.com:25565")
    run.add_argument("--seconds", type=float, default=0.0, help="stop after N seconds (0 = run until Ctrl+C)")
    run.add_argument("--populate", type=int, default=0, metavar="N",
                     help="additionally spawn N offline players to populate the server")
    run.add_argument("--premium", action="append", default=[], metavar="EMAIL",
                     help="add a premium (Microsoft) player; the device code is printed here (repeatable)")
    run.add_argument("--quiet", action="store_true", help="only print warnings and errors")

    action = sub.add_parser("action", help="send one manual action to a running panel")
    action.add_argument("payload", help='action JSON, e.g. \'{"action":"wander"}\' or {"action":"goto","x":10,"z":4}')
    action.add_argument("--url", default="http://127.0.0.1:8000", help="panel base URL")

    doctor = sub.add_parser("doctor", help="check the environment (node, mineflayer, ollama, config)")
    _add_config_arg(doctor)

    stub = sub.add_parser("stub", help="run the Ollama-compatible demo stub (rule-based, NOT a real LLM)")
    stub.add_argument("--host", default="127.0.0.1")
    stub.add_argument("--port", type=int, default=11434)
    stub.add_argument("--quiet", action="store_true", help="do not log every request")

    return parser


# ---------------------------------------------------------------------- serve
def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from pymc_bot.server import create_app

    store = ConfigStore(args.config)
    store.load()
    settings = store.settings
    host = args.host or settings.server.host
    port = args.port or settings.server.port

    app = create_app(
        config_path=args.config,
        autostart=args.autostart,
        auto_agent=args.auto_agent,
    )
    print(f"PyMC_Bot {__version__} control panel: http://{host}:{port}  (config: {store.path})")
    uvicorn.run(app, host=host, port=port, log_level=args.log_level, reload=args.reload)
    return 0


# ------------------------------------------------------------------------ run
def cmd_run(args: argparse.Namespace) -> int:
    store = ConfigStore(args.config)
    store.load()
    patch: dict[str, Any] = {}
    if args.backend:
        patch.setdefault("minecraft", {})["backend"] = args.backend
    if args.server:
        if ":" in args.server:
            host, _, port = args.server.rpartition(":")
            patch.setdefault("minecraft", {})["host"] = host
            patch.setdefault("minecraft", {})["port"] = int(port)
        else:
            patch.setdefault("minecraft", {})["host"] = args.server
    if patch:
        store.update(patch)

    log = EventLog()
    fleet = BotFleet(store, log)
    bot = fleet.primary
    agent = fleet.primary_agent()

    level_rank = {"debug": 10, "info": 20, "success": 20, "chat": 20, "bot": 20, "ai": 20, "warn": 30, "error": 40}
    minimum = 30 if args.quiet else 10
    log.add_listener(
        lambda event: print(f"[{event.level:>7}] {event.source}: {event.message}")
        if level_rank.get(event.level, 20) >= minimum
        else None
    )

    try:
        bot.start()
    except BackendError as exc:
        print(f"Could not connect: {exc}", file=sys.stderr)
        return 2

    if not args.no_ai:
        agent.start()

    for email in args.premium:
        try:
            fleet.spawn(email, auth="microsoft", ai="off" if args.no_ai else "heuristic")
            print(f"Adding premium player {email} - watch for the device code below.")
        except FleetError as exc:
            print(f"Could not add premium player {email}: {exc}", file=sys.stderr)
    if args.populate:
        try:
            members = fleet.spawn_many(count=args.populate)
            print(f"Populating the server with {len(members)} players: "
                  f"{', '.join(m.username for m in members[:6])}{' ...' if len(members) > 6 else ''}")
        except FleetError as exc:
            print(f"Could not populate the server: {exc}", file=sys.stderr)

    deadline = time.monotonic() + args.seconds if args.seconds else None
    try:
        while deadline is None or time.monotonic() < deadline:
            time.sleep(0.2)
            if bot.state not in ("connected", "connecting"):
                print("Bot is no longer connected.", file=sys.stderr)
                break
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        fleet.shutdown()
    return 0


# --------------------------------------------------------------------- action
def cmd_action(args: argparse.Namespace) -> int:
    import httpx

    url = args.url.rstrip("/") + "/api/bot/action"
    try:
        payload = json.loads(args.payload)
    except json.JSONDecodeError:
        payload = {"raw": args.payload}
    try:
        response = httpx.post(url, json=payload, timeout=60.0)
    except httpx.HTTPError as exc:
        print(f"Could not reach {url}: {exc}", file=sys.stderr)
        return 2
    print(response.status_code, response.text)
    return 0 if response.status_code < 400 else 1


# --------------------------------------------------------------------- doctor
def cmd_doctor(args: argparse.Namespace) -> int:
    store = ConfigStore(args.config)
    store.load()
    settings = store.settings

    print(f"PyMC_Bot {__version__}")
    print(f"python      : {sys.version.split()[0]}")
    backends = available_backends()
    node = backends["node"]
    print(f"node        : {backends.get('node_version') or 'NOT FOUND'}")
    print(f"mineflayer  : {'ready' if node['available'] else node['reason']}")
    print(f"config      : {store.path} {'(exists)' if store.path.exists() else '(will be created)'}")
    if store.load_error:
        print(f"config note : {store.load_error}")
    print(f"server      : {settings.minecraft.host}:{settings.minecraft.port} "
          f"as {settings.minecraft.username} (auth={settings.minecraft.auth}, backend={settings.minecraft.backend})")
    print(f"web panel   : http://{settings.server.host}:{settings.server.port}")
    print(f"ai brain    : {'ollama ' + settings.ollama.model if settings.ollama.enabled else 'built-in heuristics (ollama disabled)'}")

    if settings.ollama.enabled or True:
        client = OllamaClient(
            base_url=settings.ollama.base_url,
            model=settings.ollama.model,
            timeout=5.0,
        )
        ok, detail = client.ping()
        client.close()
        print(f"ollama      : {settings.ollama.base_url} -> {'OK' if ok else 'unreachable'} ({detail})")

    if not node_available()[0]:
        print("\nTip: install the Minecraft bridge with  npm install  in the project root.")
    return 0


# ----------------------------------------------------------------------- stub
def cmd_stub(args: argparse.Namespace) -> int:
    from pymc_bot.ollama_stub import main as stub_main

    argv = ["--host", args.host, "--port", str(args.port)]
    if args.quiet:
        argv.append("--quiet")
    return stub_main(argv)


DISPATCH = {
    "serve": cmd_serve,
    "run": cmd_run,
    "action": cmd_action,
    "doctor": cmd_doctor,
    "stub": cmd_stub,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        # `python -m pymc_bot` with no arguments -> serve the control panel.
        args = parser.parse_args(["serve", *(argv or [])])
    handler = DISPATCH[args.command]
    return handler(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
