#!/usr/bin/env bash
# Small helper around the common PyMC_Bot tasks.
#
#   ./scripts/dev.sh setup     # create .venv, install python + node deps
#   ./scripts/dev.sh serve     # run the web config panel (autostart + AI loop)
#   ./scripts/dev.sh demo      # panel + Ollama stub, simulated world, AI on
#   ./scripts/dev.sh run       # headless bot
#   ./scripts/dev.sh test      # pytest
#   ./scripts/dev.sh lint      # ruff + node syntax checks
#   ./scripts/dev.sh doctor    # environment check
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-python3}
VENV=${VENV:-.venv}
PYBIN="$VENV/bin/python"

if [[ -x "$PYBIN" ]]; then
  PY="$PYBIN"
fi

cmd=${1:-help}
shift || true

case "$cmd" in
  setup)
    "$PY" -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install -r requirements-dev.txt
    if command -v npm >/dev/null 2>&1; then npm install; else echo "npm not found - real Minecraft servers need Node >= 18"; fi
    echo "Done. Try: ./scripts/dev.sh serve"
    ;;

  serve)
    exec "$PY" -m pymc_bot serve --autostart --auto-agent "$@"
    ;;

  demo)
    # Ollama stub on 11434 + panel with the simulated world already running.
    "$PY" -m pymc_bot stub --port 11434 --quiet &
    STUB_PID=$!
    trap 'kill $STUB_PID 2>/dev/null || true' EXIT
    sleep 0.7
    "$PY" - <<'PYCODE'
import json, os, pathlib
from pymc_bot.config import ConfigStore

store = ConfigStore()
store.load()
store.update({
    "minecraft": {"backend": "simulated"},
    "ollama": {"enabled": True, "base_url": "http://127.0.0.1:11434", "model": "pymc-stub:latest",
               "decision_interval": 3.0},
})
print(f"demo config written to {store.path} (backend=simulated, brain=ollama stub)")
PYCODE
    exec "$PY" -m pymc_bot serve --autostart --auto-agent "$@"
    ;;

  run)
    exec "$PY" -m pymc_bot run "$@"
    ;;

  test)
    exec "$PY" -m pytest -q "$@"
    ;;

  lint)
    "$PY" -m ruff check .
    if command -v node >/dev/null 2>&1; then
      node --check pymc_bot/node/minecraft_bridge.js
      node --check pymc_bot/web/app.js
      node --check tests/fixtures/fake_bridge.js
      node scripts/check_ui.mjs
    fi
    echo "lint ok"
    ;;

  doctor)
    exec "$PY" -m pymc_bot doctor
    ;;

  *)
    sed -n '2,12p' "$0"
    ;;
esac
