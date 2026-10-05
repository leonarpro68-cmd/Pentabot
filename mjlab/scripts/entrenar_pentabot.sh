#!/usr/bin/env bash
# Entrenamiento completo del Pentabot (tarea Pentabot-Flat) + tensorboard.
# Uso:  pentabot/entrenar_pentabot.sh  [num_envs]      (por defecto 4096)
set -uo pipefail

REPO="$HOME/unitree_rl_mjlab"
NUM_ENVS="${1:-4096}"
LOG="$REPO/logs/entrenamiento_pentabot.log"
TB_LOG="$REPO/logs/tensorboard.log"
PUERTO=6006

# PYTHONPATH de ROS (python3.10/3.12) rompe el env de Python 3.11: se limpia.
unset PYTHONPATH
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate unitree_rl_mjlab
cd "$REPO"
mkdir -p "$REPO/logs"

echo "=== Entrenamiento del Pentabot ==="
echo "Entornos  : $NUM_ENVS"
echo "Log       : $LOG"

nohup python scripts/train.py Pentabot-Flat --env.scene.num-envs="$NUM_ENVS" \
  > "$LOG" 2>&1 &
PID_TRAIN=$!
echo "Entrenamiento lanzado (PID $PID_TRAIN)"

# Tensorboard (si el puerto ya esta ocupado, no se relanza)
if ss -ltn 2>/dev/null | grep -q ":$PUERTO "; then
  echo "Tensorboard ya estaba escuchando en el puerto $PUERTO"
else
  nohup tensorboard --logdir "$REPO/logs/rsl_rl/pentabot_velocity" --port "$PUERTO" \
    > "$TB_LOG" 2>&1 &
  echo "Tensorboard lanzado (PID $!)"
fi

echo
echo "Tensorboard : http://localhost:$PUERTO"
echo "Ver el log  : tail -f $LOG"
echo "Detener     : kill $PID_TRAIN"
