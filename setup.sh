#!/usr/bin/env bash
# Instala todo lo necesario en un entorno virtual local (.venv). Ubuntu 20.04+.
# Uso:  bash setup.sh
set -e
cd "$(dirname "$0")"

echo "==> Paquetes del sistema (pide contraseña de sudo)"
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip libgl1 libglfw3 libegl1

echo "==> Entorno virtual .venv"
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Prueba rápida del modelo"
python scripts/test_modelo.py

echo
echo "Listo. Para abrir el simulador:  bash ver.sh"
