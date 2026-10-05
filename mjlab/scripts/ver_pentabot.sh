#!/usr/bin/env bash
# Ver la politica entrenada del Pentabot.
# Uso:  pentabot/ver_pentabot.sh <ruta_al_checkpoint.pt | ultimo>  [native|viser]
#   "ultimo" = el checkpoint mas alto de la corrida mas reciente.
# Si el visor nativo falla, reintenta solo con --viewer viser.
set -uo pipefail

REPO="$HOME/unitree_rl_mjlab"
CKPT="${1:-}"
VISOR="${2:-auto}"

LOGS="$REPO/logs/rsl_rl/pentabot_velocity"

if [ -z "$CKPT" ]; then
  echo "Uso: $0 <ruta_al_checkpoint.pt | ultimo> [native|viser]"
  echo "  ultimo = el checkpoint mas alto de la corrida mas reciente"
  echo
  echo "Checkpoints disponibles:"
  find "$LOGS" -name "model_*.pt" 2>/dev/null | sort || echo "  (ninguno todavia)"
  exit 1
fi

# "ultimo": corrida mas reciente + checkpoint de numero mas alto.
if [ "$CKPT" = "ultimo" ]; then
  DIR="$(ls -dt "$LOGS"/*/ 2>/dev/null | head -1)"
  if [ -z "$DIR" ]; then
    echo "ERROR: no hay ninguna corrida en $LOGS" >&2
    exit 1
  fi
  CKPT="$(ls -1v "$DIR"model_*.pt 2>/dev/null | tail -1)"
  if [ -z "$CKPT" ]; then
    echo "ERROR: la corrida mas reciente ($DIR) todavia no tiene checkpoints." >&2
    echo "       Los checkpoints salen cada 100 iteraciones." >&2
    exit 1
  fi
  echo "ultimo -> $(basename "$CKPT")  (corrida $(basename "${DIR%/}"))"
fi

if [ ! -f "$CKPT" ]; then
  echo "ERROR: no existe el checkpoint: $CKPT" >&2
  exit 1
fi
CKPT="$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")"

# PYTHONPATH de ROS rompe el env de Python 3.11: se limpia.
unset PYTHONPATH
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate unitree_rl_mjlab
cd "$REPO"

echo "Checkpoint: $CKPT"

if [ "$VISOR" != "auto" ]; then
  exec python scripts/play.py Pentabot-Flat --checkpoint_file="$CKPT" --viewer "$VISOR"
fi

echo "Visor: nativo (si falla, reintento con viser en el navegador)"
python scripts/play.py Pentabot-Flat --checkpoint_file="$CKPT"
if [ $? -ne 0 ]; then
  echo
  echo "El visor nativo fallo. Reintentando con viser (se abre en el navegador)..."
  exec python scripts/play.py Pentabot-Flat --checkpoint_file="$CKPT" --viewer viser
fi
