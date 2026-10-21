"""Command line entry point.

``python -m pymc_bot [serve|run|train|models|action|doctor|probe|search|mob|llm|stub|about]``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from pymc_bot import __version__, about, credit_line
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


class _TopFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Defaults in the help, and the credit epilog kept on one line."""


class _VersionAction(argparse.Action):
    """``--version`` printed verbatim (argparse's own action re-wraps long lines)."""

    def __call__(self, parser, namespace, values, option_string=None):  # noqa: ANN001
        print(credit_line())
        parser.exit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pymc-bot",
        description="PyMC_Bot - a Python-controlled Minecraft player bot with an Ollama brain and a web config page.",
        formatter_class=_TopFormatter,
    )
    parser.epilog = "Made with \u2764\ufe0f by codero-sus \u00b7 https://github.com/codero-sus \u00b7 https://the-codero.web.app/"
    parser.add_argument("--version", action=_VersionAction, nargs=0, help="show the version and author, then exit")
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
    run.add_argument("--think", choices=["auto", "heuristic", "ollama", "cortex", "trained"], default=None,
                     help="which brain drives this run ('trained' = the model trained on playtime)")
    run.add_argument("--cortex-url", default=None, metavar="URL",
                     help="Cortex LLMHoster address for --think cortex (default http://127.0.0.1:8624)")
    run.add_argument("--cortex-model", default=None, metavar="ID",
                     help="Cortex model id (blank = the model Cortex marks as default)")
    run.add_argument("--run-name", default=None, help="activate this trained model (with --think trained)")
    run.add_argument("--models-dir", default=None, help="where trained models live")

    train = sub.add_parser(
        "train",
        help="train a player model on recorded playtime (self-checkpointing, resumable)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_config_arg(train)
    train.add_argument("--dataset", default=None,
                       help="playtime dataset (dataset.jsonl, an episode dir, or <gameDir>/pymcbot)")
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
    train.add_argument("--min-episode", type=float, default=0.0, metavar="SECONDS",
                       help="leave out episodes shorter than this (the mod's Power tab export filter)")
    train.add_argument("--idle-keep", type=float, default=100.0, metavar="PERCENT",
                       help="keep only this share of standing-still samples (the mod's Power tab export filter)")
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

    doctor = sub.add_parser("doctor", help="check the environment (node, mineflayer, ollama, cortex, config)")
    _add_config_arg(doctor)

    probe = sub.add_parser("probe", help="ping a server without joining (Aternos & co: shows if it is asleep)")
    probe.add_argument("address", nargs="?", help="host or host:port (default: the configured server)")
    probe.add_argument("--port", type=int, default=None, help="port when not part of the address")
    probe.add_argument("--json", action="store_true", help="print the raw result as JSON")
    _add_config_arg(probe)

    search = sub.add_parser("search", help="look something up like the AI does (Minecraft Wiki by default)")
    search.add_argument("query", nargs="+", help='a short question, e.g. "how to tame a wolf"')
    search.add_argument("--provider", choices=["minecraft_wiki", "duckduckgo", "searxng", "brave"], default=None)
    search.add_argument("--no-cache", action="store_true", help="ask the web even when the answer is cached")
    search.add_argument("--no-fallback", action="store_true", help="do not fall back to DuckDuckGo")
    search.add_argument("--no-mob-db", action="store_true", help="skip the built-in mob database, always ask the web")
    search.add_argument("--json", action="store_true", help="print the raw result as JSON")
    _add_config_arg(search)

    mob = sub.add_parser("mob", help="show a mob from the built-in mob database (offline)")
    mob.add_argument("name", nargs="*", help='a mob name, alias or plural, e.g. "sulphur cube"')
    mob.add_argument("--list", action="store_true", help="list mobs (combine with the filters)")
    mob.add_argument("--category", choices=["hostile", "neutral", "passive", "boss"], default=None)
    mob.add_argument("--dimension", choices=["overworld", "nether", "end"], default=None)
    mob.add_argument("--since", metavar="VERSION", default=None, help="only mobs added in VERSION or later")
    mob.add_argument("--version", metavar="VERSION", default=None,
                     help="say whether the mob exists in VERSION / list only mobs that exist in it")
    mob.add_argument("--json", action="store_true", help="print JSON")

    llm = sub.add_parser("llm", help="train your own Minecraft LLM: build data, presets, import, eval (docs/MINECRAFT_LLM.md)")
    llm_sub = llm.add_subparsers(dest="llm_command")
    build = llm_sub.add_parser("build", help="write train.jsonl / eval.jsonl + the Kaggle training script")
    build.add_argument("--out", default="llm_data", help="output folder (upload it to Kaggle as a dataset)")
    build.add_argument("--turns", type=int, default=6000, help="game situations answered by the built-in rules")
    build.add_argument("--wiki-pages", type=int, default=600, help="Minecraft Wiki pages to use (cached)")
    build.add_argument("--no-wiki", action="store_true", help="offline: skip the Minecraft Wiki")
    build.add_argument("--teacher", choices=["ollama", "cortex"], default=None,
                       help="let a bigger model you run answer extra situations (uses its settings from the config)")
    build.add_argument("--teacher-model", default=None, help="model name for the teacher (default: the configured one)")
    build.add_argument("--teacher-turns", type=int, default=500)
    build.add_argument("--recorded", action="append", default=None, metavar="FILE",
                       help="turns recorded by agent.record_turns (default: the configured file if it exists)")
    build.add_argument("--playtime", default=None, metavar="PATH", help="your recorded playtime (mod dataset / folder)")
    build.add_argument("--screenshots", nargs="?", const="", default=None, metavar="PATH",
                       help="add picture rows for the Minecraftism VL editions from playtime recorded with "
                            "'/pymcbot screenshots on' (mod 0.10+); no PATH = the --playtime folder")
    build.add_argument("--kaggle-user", default="your-kaggle-name", help="for dataset-metadata.json")
    build.add_argument("--seed", type=int, default=7)
    _add_config_arg(build)
    llm_sub.add_parser("presets", help="the model sizes you can train and why you would pick each")
    imp = llm_sub.add_parser("import", help="add a model trained on Kaggle to Ollama and switch the bot to it")
    imp.add_argument("path", nargs="?", help="the downloaded model folder (from Kaggle's Output tab) or a .gguf file")
    imp.add_argument("--hf", metavar="USER/REPO", help="pull from Hugging Face instead, e.g. you/Minecraftism-4B-GGUF")
    imp.add_argument("--quant", default="Q4_K_M", help="GGUF quantisation to pull with --hf (default: Q4_K_M)")
    imp.add_argument("--name", default=None,
                     help="Ollama model name (default: the Minecraftism edition, e.g. minecraftism-vl, else pymc-minecraft)")
    imp.add_argument("--no-activate", action="store_true", help="only create the Ollama model, keep the bot's settings")
    _add_config_arg(imp)
    ev = llm_sub.add_parser("eval", help="score an Ollama model on eval.jsonl (valid JSON, right action, right answers)")
    ev.add_argument("--model", default=None, help="Ollama model (default: the configured one)")
    ev.add_argument("--data", default="llm_data/eval.jsonl")
    ev.add_argument("--limit", type=int, default=100)
    ev.add_argument("--json", action="store_true")
    _add_config_arg(ev)

    about_cmd = sub.add_parser("about", help="author, links and credits")
    about_cmd.add_argument("--json", action="store_true", help="print the credits as JSON")

    stub = sub.add_parser("stub", help="run the Ollama/OpenAI-compatible demo stub (rule-based, NOT a real LLM)")
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
    print("Made with \u2764\ufe0f by codero-sus \u00b7 https://github.com/codero-sus \u00b7 https://the-codero.web.app/")
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
    if getattr(args, "cortex_url", None):
        patch.setdefault("cortex", {})["base_url"] = args.cortex_url
    if getattr(args, "cortex_model", None) is not None:
        patch.setdefault("cortex", {})["model"] = args.cortex_model
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

    if args.think == "cortex":
        from pymc_bot.cortex import client_from_settings

        client = client_from_settings(settings.cortex, timeout=5.0)
        ok, detail = client.ping()
        client.close()
        if ok:
            print(f"Thinking with Cortex LLMHoster at {settings.cortex.base_url}: {detail}")
        else:
            # Not fatal: every failed decision falls back to the built-in behaviour.
            print(f"Cortex LLMHoster is not usable yet ({detail}); the bot plays on its "
                  "built-in behaviour until it is.", file=sys.stderr)

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
    if not args.simulate:
        from pymc_bot.playtime import resolve_dataset_path

        dataset = str(resolve_dataset_path(dataset))

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
        min_episode_seconds=max(0.0, args.min_episode),
        idle_keep_percent=max(0.0, min(100.0, args.idle_keep)),
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

    print(credit_line())
    print(f"python      : {sys.version.split()[0]}")
    backends = available_backends()
    node = backends["node"]
    print(f"node        : {backends.get('node_version') or 'NOT FOUND'}")
    print(f"mineflayer  : {'ready' if node['available'] else node['reason']}")
    mc_range = backends.get("minecraft_versions")
    if mc_range:
        print(f"minecraft   : {mc_range['oldest']} - {mc_range['latest']} (newer servers need ViaVersion + ViaBackwards "
              "until mineflayer adds them; then run the updater or `npm update`)")
    print(f"config      : {store.path} {'(exists)' if store.path.exists() else '(will be created)'}")
    if store.load_error:
        print(f"config note : {store.load_error}")
    print(f"server      : {settings.minecraft.host}:{settings.minecraft.port} "
          f"as {settings.minecraft.username} (auth={settings.minecraft.auth}, backend={settings.minecraft.backend})")
    print(f"web panel   : http://{settings.server.host}:{settings.server.port}")
    from pymc_bot.fleet import primary_brain

    brain = primary_brain(settings)
    brain_detail = {
        "ollama": f"ollama ({settings.ollama.model})",
        "cortex": f"cortex llmhoster ({settings.cortex.model or 'its default model'})",
        "heuristic": "built-in heuristics (no LLM enabled)",
        "trained": f"trained playtime model ({settings.training.active_run or 'newest run'})",
    }.get(brain, brain)
    print(f"ai brain    : {brain_detail}  [agent.mode={settings.agent.mode}]")

    if settings.ollama.enabled or True:
        client = OllamaClient(
            base_url=settings.ollama.base_url,
            model=settings.ollama.model,
            timeout=5.0,
        )
        ok, detail = client.ping()
        client.close()
        print(f"ollama      : {settings.ollama.base_url} -> {'OK' if ok else 'unreachable'} ({detail})")

    from pymc_bot.cortex import client_from_settings

    cortex_cfg = settings.cortex
    client = client_from_settings(cortex_cfg, timeout=5.0)
    ok, detail = client.ping()
    client.close()
    key_note = (
        f"key from ${cortex_cfg.api_key_env}" if cortex_cfg.key_present()
        else f"${cortex_cfg.api_key_env or '-'} not set (fine if Cortex runs without a key)"
    )
    if ok or cortex_cfg.enabled or brain == "cortex":
        print(f"cortex      : {cortex_cfg.base_url}{cortex_cfg.api_path} -> {'OK' if ok else 'unavailable'} ({detail}; {key_note})")
    else:
        print("cortex      : not running (optional: only needed for the Cortex brain)")

    from pymc_bot.hosting import describe_probe, host_tips, probe_server

    if node_available()[0]:
        result = probe_server(settings.minecraft.host, settings.minecraft.port, timeout=6.0)
        print(f"server ping : {describe_probe(result)}")
    for tip in host_tips(settings.minecraft.host, settings.minecraft.auth):
        print(f"hosting     : {tip}")

    if not node_available()[0]:
        print("\nTip: install the Minecraft bridge with  npm install  in the project root.")
    return 0


# ---------------------------------------------------------------------- probe
def cmd_probe(args: argparse.Namespace) -> int:
    """Server list ping: online/asleep, version, players, and tips for free hosts."""
    from pymc_bot.hosting import describe_probe, host_tips, probe_server

    store = ConfigStore(args.config)
    store.load()
    minecraft = store.settings.minecraft
    address = args.address or minecraft.host
    port = args.port or (minecraft.port if not args.address else 25565)
    result = probe_server(address, port)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(describe_probe(result))
        for tip in host_tips(result["host"], minecraft.auth)[1:]:
            print(f"  - {tip}")
    if not result.get("ok"):
        return 1
    return 3 if result.get("asleep") else 0


# --------------------------------------------------------------------- search
def cmd_search(args: argparse.Namespace) -> int:
    """One lookup through pymc_bot.knowledge - handy to check what the AI would read."""
    from pymc_bot.knowledge import KnowledgeBase, shared_knowledge

    store = ConfigStore(args.config)
    store.load()
    settings = store.settings.search.model_copy(update={"enabled": True})
    if args.provider:
        settings = settings.model_copy(update={"provider": args.provider})
    if args.no_fallback:
        settings = settings.model_copy(update={"fallback": False})
    if args.no_mob_db:
        settings = settings.model_copy(update={"use_mob_database": False})
    query = " ".join(args.query)
    kb = KnowledgeBase(settings, None) if args.no_cache else shared_knowledge(settings, store.path.parent)
    result = kb.lookup(query)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    elif result.ok:
        print(f"{result.title or query}  [{result.source}{', cached' if result.cached else ''}]")
        if result.url:
            print(result.url)
        print()
        print(result.text)
    else:
        print(f"No answer for {query!r}: {result.error}")
    return 0 if result.ok else 1


def cmd_mob(args: argparse.Namespace) -> int:
    """Look a mob up in pymc_bot/data/mobs.json, or list mobs with filters."""
    from pymc_bot.mobs import is_version, load_mobs

    db = load_mobs()
    for value in (args.since, args.version):
        if value and not is_version(value):
            print(f"Not a version: {value!r} (use e.g. 1.21.4 or 26.2)", file=sys.stderr)
            return 2
    name = " ".join(args.name).strip()
    if args.list or not name:
        found = db.search(category=args.category, dimension=args.dimension, added_since=args.since,
                          exists_in=args.version, query=name or None)
        if args.json:
            print(json.dumps([db.with_derived(m) for m in found], indent=2, ensure_ascii=False))
            return 0
        for m in found:
            flag = "" if db.supported_by_mineflayer(m) else "  (newer than mineflayer)"
            print(f"{m['name']:<20} {m['category']:<8} {str(m['health']):>6} HP  up to {db.max_damage(m):>5g} dmg  "
                  f"added {m['added_in']}{flag}")
        print(f"{len(found)} of {len(db)} mobs")
        return 0
    mob = db.get(name)
    if mob is None:
        print(f"No mob called {name!r} in the database. Try: python -m pymc_bot mob --list", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(db.with_derived(mob), indent=2, ensure_ascii=False))
    else:
        print(db.describe(mob, args.version).replace(". ", ".\n"))
    return 0


# ----------------------------------------------------------------------- stub
# ---------------------------------------------------------------------- about
def cmd_about(args: argparse.Namespace) -> int:
    info = about()
    if getattr(args, "json", False):
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0
    print(f"{info['name']} {info['version']}  \u2014  {info['made_with']}")
    print(f"author      : {info['author']}  ({info['author_github']})")
    print(f"website     : {info['website']}")
    print(f"repository  : {info['repository']}")
    print(f"license     : {info['license']}")
    print("credits     :")
    for credit in info["credits"]:
        print(f"  - {credit['name']} by {credit['by']} \u2014 {credit['role']} ({credit['url']})")
    print("built with  : " + ", ".join(item["name"] for item in info["built_with"]))
    return 0


def cmd_llm(args: argparse.Namespace) -> int:
    """Build training data, list presets, import a trained GGUF, evaluate a model."""
    from pymc_bot import llm_train

    command = getattr(args, "llm_command", None)
    if command == "presets" or command is None:
        if command is None:
            print("usage: python -m pymc_bot llm {build,presets,import,eval} - see docs/MINECRAFT_LLM.md\n")
        return llm_train.main(["--list-presets"])
    store = ConfigStore(args.config)
    settings = store.settings
    if command == "build":
        return _llm_build(args, store)
    if command == "import":
        from pymc_bot.llm_tools import import_model, import_model_hf

        if bool(args.path) == bool(args.hf):
            print("give either a folder/.gguf path or --hf USER/REPO", file=sys.stderr)
            return 2
        try:
            if args.hf:
                result = import_model_hf(args.hf, store, name=args.name, quant=args.quant, activate=not args.no_activate)
            else:
                result = import_model(Path(args.path), store, name=args.name, activate=not args.no_activate)
        except (FileNotFoundError, ValueError) as exc:
            print(exc, file=sys.stderr)
            return 2
        print(result.message)
        if not args.hf:
            print(f"Modelfile: {result.modelfile}")
        if result.info.get("vision"):
            print("Vision edition: the bot will attach its view every few turns (agent.vision; needs the optional "
                  "bridge packages - see docs/MINECRAFT_LLM.md).")
        if result.activated:
            print(f"Bot switched to Ollama model '{result.name}' with the short Minecraft prompt ({store.path}).")
        return 0 if result.created or result.activated else 1
    if command == "eval":
        from pymc_bot.llm_dataset import read_jsonl
        from pymc_bot.llm_tools import evaluate

        data = Path(args.data)
        if not data.is_file():
            print(f"{data} not found - run: python -m pymc_bot llm build", file=sys.stderr)
            return 2
        client = OllamaClient(settings.ollama.base_url, args.model or settings.ollama.model, 0.3,
                              max(60.0, settings.ollama.request_timeout))
        summary = evaluate(read_jsonl(data), lambda messages, system: client.chat(messages, system=system), args.limit)
        summary["model"] = client.model
        print(json.dumps(summary, indent=2) if args.json else
              f"{client.model}: valid JSON {summary['json_valid_pct']}% | right action {summary['action_match_pct']}% | "
              f"right answers {summary['answer_keywords_pct']}% ({summary['rows']} rows, {summary['errors']} errors)")
        return 0
    return 2


def _llm_build(args: argparse.Namespace, store: ConfigStore) -> int:
    import shutil

    from pymc_bot import llm_dataset, llm_train

    settings = store.settings
    out = Path(args.out)
    pages = None
    if not args.no_wiki and args.wiki_pages > 0:
        try:
            pages = llm_dataset.fetch_wiki(limit=args.wiki_pages, cache_dir=out / "wiki_cache")
        except Exception as exc:  # offline: build without the wiki
            print(f"Minecraft Wiki skipped ({type(exc).__name__}: {exc})")
    teacher = None
    if args.teacher == "ollama":
        client = OllamaClient(settings.ollama.base_url, args.teacher_model or settings.ollama.model,
                              settings.ollama.temperature, max(90.0, settings.ollama.request_timeout))
        teacher = lambda system, prompt: client.chat([{"role": "user", "content": prompt}], system=system)  # noqa: E731
    elif args.teacher == "cortex":
        from pymc_bot.cortex import client_from_settings

        cortex = client_from_settings(settings, timeout=90.0)
        if args.teacher_model:
            cortex.model = args.teacher_model
        teacher = lambda system, prompt: cortex.decide(prompt, system=system)  # noqa: E731
    screenshots = None
    if args.screenshots is not None:
        if not (args.screenshots or args.playtime):
            print("--screenshots needs a PATH or --playtime (the folder the mod exported, with its shots/ folder)",
                  file=sys.stderr)
            return 2
        screenshots = Path(args.screenshots or args.playtime)
    recorded = [Path(p) for p in args.recorded] if args.recorded else []
    if args.recorded is None:
        default = Path(settings.agent.turns_file).expanduser()
        default = default if default.is_absolute() else store.path.parent / default
        if default.is_file():
            recorded = [default]
    card = llm_dataset.build_dataset(out, seed=args.seed, rule_turns=args.turns, wiki_pages=pages, teacher=teacher,
                                     teacher_turns=args.teacher_turns if teacher else 0, recorded=recorded,
                                     playtime=Path(args.playtime) if args.playtime else None, screenshots=screenshots)
    shutil.copyfile(llm_train.__file__, out / "train_minecraft_llm.py")
    slug = "pymc-minecraft-data"
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "PyMC Bot Minecraft LLM data", "id": f"{args.kaggle_user}/{slug}",
        "licenses": [{"name": "CC-BY-NC-SA-4.0"}],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n{card['train']} training rows ready in {out}/. Next:\n"
          f"  1. kaggle.com > Datasets > New Dataset > upload the files in {out}/ (skip wiki_cache), or\n"
          f"     kaggle datasets create -p {out}   (edit the id in dataset-metadata.json first)\n"
          "  2. open training/llm/PyMC_Minecraft_LLM_Kaggle.ipynb on Kaggle, Add Input = your dataset,\n"
          "     Accelerator = GPU T4 x2, pick the Minecraftism EDITIONS, Run All (llm presets lists them)\n"
          "  3. python -m pymc_bot llm import --hf <you>/Minecraftism-4B-Uncensored-GGUF   (HF_TOKEN secret set;\n"
          "     UNCENSORED = False in the notebook: <you>/Minecraftism-4B-GGUF)\n"
          "     or download /kaggle/working/<edition> and run: python -m pymc_bot llm import <folder>")
    if card.get("images"):
        print(f"  {card['images']} pictures in {out}/images/ - upload that folder too (the VL editions need it)")
    return 0


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
    "probe": cmd_probe,
    "search": cmd_search,
    "mob": cmd_mob,
    "llm": cmd_llm,
    "stub": cmd_stub,
    "about": cmd_about,
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
