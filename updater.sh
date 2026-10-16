#!/bin/sh
# PyMC_Bot updater for macOS / Linux - runs updater.py with the best Python available.
#
#   ./updater.sh              update PyMC_Bot (+ dependencies when they changed)
#   ./updater.sh --check      only show what would change
#   ./updater.sh --mod        also update the Fabric mod in ~/.minecraft/mods
#   ./updater.sh --help       all options
#
# The update may replace this very file, so the script hands over to Python with `exec`
# and nothing below that line is ever read again.
set -e
cd "$(dirname "$0")"

if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
elif command -v python >/dev/null 2>&1; then
  PY="python"
else
  echo "[update] ERROR: Python 3.10+ was not found - install it from https://www.python.org/downloads/" >&2
  exit 1
fi

exec "$PY" updater.py "$@"
