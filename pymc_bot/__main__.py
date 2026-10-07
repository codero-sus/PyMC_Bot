"""Command line entry point.

``python -m pymc_bot [serve|run|train|models|action|doctor|stub]``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
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
    run.add_argument("--think", choices=["auto", "heuristic", "ollama", "trained"], default=None,
                     help="which brain drives this run ('trained' = the model trained on playtime)")
    run.add_argument("--run-name", default=None, help="activate this trained model (with --think trained)")
    run.add_argument("--models-dir", default=None, help="where trained models live")

    train = sub.add_parser(
        "train",
        help="train a player model on recorded playtime (self-checkpointing, resumable)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_config_arg(train)
    train.add_argument("--dataset", default=None,
                       help="playtime dataset (dataset.jsonl, an episode dir, or <gameDir>/pymc-playtime)")
    train.add_argument("--simulate", type=float, default=0.0, metavar="MINUTES",
                       help="first generate MINUTES of synthetic playtime (no Minecraft needed) and train on it")
    train.add_argument("--simulate-out", default=None, help="where to write the synthetic dataset")
    train.add_argument("--inspect", action="store_true", help="only print dataset statistics, do not train")
    train.add_argument("--run-name", default=None, help="checkpoint folder name under models_dir")
    train.add_argument("--models-dir", default=None, help="where checkpoints live")
    train.add_argument("--engine", choices=["mlp", "transformer"], default=None,
                       help="mlp (NumPy, default) or transformer (needs requirements-train.txt)")
    train.add_argument("--steps", type=int, default=None, help="training steps (batches)")
    train.add_argument("--epochs", type=int, default=0, help="train for N passes over the data instead of --steps")
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--lr", type=float, default=None)
    train.add_argument("--hidden", default=None, help="comma separated MLP hidden sizes, e.g. 128,64")
    train.add_argument("--checkpoint-every", type=int, default=None, help="write a checkpoint every N steps")
    train.add_argument("--max-seconds", type=float, default=0.0, help="stop after N seconds (long runs)")
    train.add_argument("--no-blocks", action="store_true", help="ignore the 3x3x3 block neighbourhood")
    train.add_argument("--advanced", action="store_true",
                       help="advanced training: learn entity and item vocabularies from the playtime")
    train.add_argument("--entity-slots", type=int, default=0, metavar="N",
                       help="entity words to learn in advanced mode (0 = config/default 24)")
    train.add_argument("--item-slots", type=int, default=0, metavar="N",
                       help="item words to learn in advanced mode (0 = config/default 32)")
    train.add_argument("--resume", action="store_true", help="continue the run from its last checkpoint")
    train.add_argument("--from-checkpoint", default="", help="resume from a specific checkpoint file")
    train.add_argument("--ollama-model", default="", metavar="NAME",
                       help="also export the trained policy to Ollama as NAME (Modelfile + /api/create)")
    train.add_argument("--quiet", action="store_true", help="only print the final summary")

    models = sub.add_parser("models", help="list trained player models")
    _add_config_arg(models)
    models.add_argument("--models-dir", default=None)
    models.add_argument("--json", action="store_true", help="print raw JSON")

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
    if args.think:
        patch.setdefault("agent", {})["mode"] = args.think
    if getattr(args, "run_name", None):
        patch.setdefault("training", {})["active_run"] = args.run_name
    if getattr(args, "models_dir", None):
        patch.setdefault("training", {})["models_dir"] = args.models_dir
    if patch:
        store.update(patch)
    settings = store.settings
    if args.think == "trained":
        from pymc_bot.local_model import PolicyError, TrainedPolicy

        run = settings.training.active_run or "(newest)"
        try:
            policy = TrainedPolicy.load(
                settings.training.active_run or _newest_run(settings.training.models_dir),
                settings.training.models_dir,
                temperature=settings.training.temperature,
                step_seconds=settings.training.step_seconds,
            )
            print(
                f"Playing with the model trained on playtime: {policy.run.run} "
                f"({policy.run.engine}, step {policy.run.card.get('step')}, "
                f"{policy.run.card.get('params')} params) - {policy.run.checkpoint}"
            )
        except PolicyError as exc:
            print(f"Could not load the trained model ({run}): {exc}", file=sys.stderr)
            return 2

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


def _newest_run(models_dir: str) -> str:
    from pymc_bot.train import list_models

    runs = [card for card in list_models(models_dir) if card.get("checkpoint_exists")]
    if not runs:
        raise SystemExit(
            f"no trained model in {models_dir} - create one with "
            "'python -m pymc_bot train --simulate 5' (or record playtime with the mod)"
        )
    return str(runs[0]["run"])


# ---------------------------------------------------------------------- train
def cmd_train(args: argparse.Namespace) -> int:
    from pymc_bot.playtime import dataset_summary, synthesize_playtime
    from pymc_bot.train import TrainConfig, default_run_name, train

    store = ConfigStore(args.config)
    store.load()
    training = store.settings.training
    models_dir = args.models_dir or training.models_dir
    dataset = args.dataset or training.dataset

    advanced = bool(args.advanced or training.advanced)
    if args.simulate > 0:
        target = Path(args.simulate_out) if args.simulate_out else Path(dataset)
        info = synthesize_playtime(target, minutes=args.simulate, advanced=advanced)
        print(
            f"Generated {info['samples']} synthetic playtime samples "
            f"({info['minutes']} min) -> {info['dataset']}"
        )
        dataset = str(target)

    if args.inspect:
        print(json.dumps(dataset_summary(dataset), indent=2))
        return 0

    config = TrainConfig(
        dataset=dataset,
        run_name=args.run_name or default_run_name(dataset),
        models_dir=models_dir,
        engine=args.engine or training.engine,
        batch_size=args.batch_size or training.batch_size,
        lr=args.lr or training.lr,
        steps=args.steps or training.steps,
        epochs=args.epochs,
        max_seconds=args.max_seconds,
        checkpoint_every=args.checkpoint_every or training.checkpoint_every,
        include_blocks=not args.no_blocks and training.include_blocks,
        advanced=advanced,
        entity_slots=args.entity_slots or training.entity_slots,
        item_slots=args.item_slots or training.item_slots,
        hidden=tuple(int(part) for part in args.hidden.split(",")) if args.hidden else tuple(training.hidden),
        resume=args.resume,
        from_checkpoint=args.from_checkpoint,
        verbose=not args.quiet,
    )
    mode = "advanced (entities + items)" if config.advanced else "basic"
    print(
        f"Training '{config.run_name}' on {config.dataset} "
        f"({config.engine}, {config.steps} steps, checkpoint every {config.checkpoint_every}, {mode}) -> "
        f"{Path(config.models_dir).expanduser() / config.run_name}"
    )
    if not args.quiet:
        print("  (each checkpoint also updates models/<run>/model.json and metrics.jsonl)")
    summary = train(config)
    print(
        f"Done: step {summary['steps']} ({summary['epochs']} epochs), "
        f"train_loss={summary['loss']}, val_loss={summary['val_loss']}, val_acc={summary['val_acc']}"
    )
    print(f"Checkpoint: {summary['checkpoint']}")
    print(
        f"Run it with:  python -m pymc_bot run --think trained "
        f"--run-name {summary['run']} --models-dir {config.models_dir} --backend simulated"
    )

    if args.ollama_model:
        from pymc_bot.local_model import TrainedPolicy, export_ollama_model
        from pymc_bot.playtime import load_examples

        client = OllamaClient(
            base_url=store.settings.ollama.base_url,
            model=store.settings.ollama.model,
            timeout=store.settings.ollama.request_timeout,
        )
        policy = TrainedPolicy.load(summary["run"], config.models_dir)
        observations = [observation for observation, _ in load_examples(dataset, limit=200)]
        try:
            result = export_ollama_model(
                policy,
                name=args.ollama_model,
                client=client,
                base_model=store.settings.ollama.model,
                observations=observations,
            )
            print(f"Exported the trained policy to Ollama as '{args.ollama_model}' ({result['modelfile']})")
        except Exception as exc:
            print(f"Could not export to Ollama: {exc}", file=sys.stderr)
        finally:
            client.close()
    return 0


# --------------------------------------------------------------------- models
def cmd_models(args: argparse.Namespace) -> int:
    from pymc_bot.train import list_models

    store = ConfigStore(args.config)
    store.load()
    models_dir = args.models_dir or store.settings.training.models_dir
    runs = list_models(models_dir)
    if args.json:
        print(json.dumps(runs, indent=2))
        return 0
    if not runs:
        print(f"No trained models in {models_dir}.")
        print("Train one with:  python -m pymc_bot train --simulate 5   (or use the Fabric mod in ./mod)")
        return 0
    active = store.settings.training.active_run
    print(f"{'run':32} {'engine':12} {'mode':26} {'step':>7} {'params':>9} {'val_loss':>9} {'val_acc':>8}  active")
    for card in runs:
        metrics = card.get("metrics") or {}
        marker = "<=" if card.get("run") == active else ""
        mode = (
            f"advanced ({len(card.get('entity_vocabulary') or [])}e/"
            f"{len(card.get('item_vocabulary') or [])}i)"
            if card.get("advanced")
            else "basic"
        )
        print(
            f"{str(card.get('run'))[:32]:32} {str(card.get('engine'))[:12]:12} {mode[:26]:26} "
            f"{card.get('step') or 0:>7} {card.get('params') or 0:>9} "
            f"{(metrics.get('val_loss') if metrics.get('val_loss') is not None else float('nan')):>9.4f} "
            f"{(metrics.get('val_accuracy') or 0.0):>8.3f}  {marker}"
        )
    print("\nActivate one with:  python -m pymc_bot models --models-dir " + models_dir
          + "   then set agent.mode=trained in the panel (or use 'run --think trained')")
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
    "train": cmd_train,
    "models": cmd_models,
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
