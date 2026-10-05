#!/usr/bin/env bash
# Abre el visor interactivo de MuJoCo con el Pentabot.
# Uso:  bash ver.sh            (robot quieto, controla los servos con los sliders del panel "Control")
#       bash ver.sh demo       (marcha de prueba en lazo abierto)
set -e
cd "$(dirname "$0")"
. .venv/bin/activate
if [ "$1" = "demo" ]; then
  python scripts/demo_marcha.py
else
  python scripts/ver.py
fi
