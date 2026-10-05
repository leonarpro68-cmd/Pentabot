"""Pilota el bipedo (o el Pentabot) en MuJoCo con el mando de PS5 (DualSense).

Uso:  python jugar_mando.py <checkpoint.pt | ultimo>  [--viewer native|viser]
                            [--robot bipedo|pentabot|pentabot_sin_encoder]   (por defecto bipedo)

Mapeo:
  stick izquierdo arriba/abajo -> avanzar / retroceder   (lin_vel_x)
  stick izquierdo izq/der      -> desplazarse de lado   (lin_vel_y)
  stick derecho izq/der        -> girar                 (ang_vel_z)
  circulo                      -> parada inmediata

El signo de "avance" ya viene corregido para el DualSense de esta maquina
(su ABS_Y crece hacia arriba). Si conectas otro mando y algun eje va al
reves:  --invertir-avance | --invertir-lateral | --invertir-giro

Por defecto el robot nace mirando siempre en la misma direccion (+x de la
escena) para que "adelante" en el stick sea "adelante" en la pantalla. Al
entrenar, el yaw inicial es aleatorio en todo el circulo (-3.14..3.14), asi
que sin esto el robot aparece mirando a cualquier lado y parece que camina
al reves. Con --yaw-libre se deja el comportamiento aleatorio original.

Los rangos se leen del propio env_cfg, asi que siempre coinciden con los
que se usaron al entrenar: pedirle mas de eso la saca de su distribucion
y la politica se cae.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import mjlab.tasks  # noqa: E402,F401  (registra las tareas de mjlab)
import scripts.play as play_mod  # noqa: E402
import src.tasks  # noqa: E402,F401  (registra Bipedo-Flat y Pentabot-Flat)
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer  # noqa: E402

from mando_ps5 import MandoPS5, escalar  # noqa: E402

# robot -> (tarea, carpeta de logs, signo del frente del modelo en x)
ROBOTS = {
  "bipedo": ("Bipedo-Flat", "bipedo_velocity", -1.0),
  "pentabot": ("Pentabot-Flat", "pentabot_velocity", 1.0),
  "pentabot_sin_encoder": ("Pentabot-Flat-SinEncoder", "pentabot_sin_encoder", 1.0),
}
ROBOT = sys.argv[sys.argv.index("--robot") + 1] if "--robot" in sys.argv else "bipedo"
if ROBOT not in ROBOTS:
  raise SystemExit(f"ERROR: --robot debe ser uno de {list(ROBOTS)}")
TAREA, _CARPETA, FRENTE = ROBOTS[ROBOT]
LOGS = REPO / "logs" / "rsl_rl" / _CARPETA


def resolver_checkpoint(arg: str) -> Path:
  if arg != "ultimo":
    p = Path(arg)
    if not p.is_file():
      raise SystemExit(f"ERROR: no existe el checkpoint: {p}")
    return p.resolve()
  corridas = sorted((d for d in LOGS.glob("*/") if d.is_dir()), key=lambda d: d.stat().st_mtime)
  if not corridas:
    raise SystemExit(f"ERROR: no hay ninguna corrida en {LOGS}")
  ultima = corridas[-1]
  modelos = sorted(ultima.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
  if not modelos:
    raise SystemExit(f"ERROR: la corrida {ultima.name} todavia no tiene checkpoints.")
  print(f"ultimo -> {modelos[-1].name}  (corrida {ultima.name})")
  return modelos[-1].resolve()


def enganchar_mando(env, mando: MandoPS5, inv_avance: bool = False,
                    inv_lateral: bool = False, inv_giro: bool = False) -> None:
  """Hace que el mando sobreescriba el comando de velocidad en cada paso.

  Se envuelve `compute` del termino en vez de escribir desde fuera: asi el
  valor del mando se aplica DESPUES del remuestreo aleatorio interno, que
  si no lo pisaria cada pocos segundos.
  """
  base = env.unwrapped
  term = base.command_manager.get_term("twist")
  rangos = term.cfg.ranges
  rx = tuple(rangos.lin_vel_x)
  ry = tuple(rangos.lin_vel_y)
  rz = tuple(rangos.ang_vel_z)

  print("\n=== mando enganchado ===")
  print(f"  avance  (stick izq. vertical)   -> lin_vel_x en [{rx[0]:+.2f}, {rx[1]:+.2f}] m/s")
  print(f"  lateral (stick izq. horizontal) -> lin_vel_y en [{ry[0]:+.2f}, {ry[1]:+.2f}] m/s")
  print(f"  giro    (stick der. horizontal) -> ang_vel_z en [{rz[0]:+.2f}, {rz[1]:+.2f}] rad/s")
  print("  circulo -> parada inmediata")
  print("\nLectura en vivo abajo: 'pide' es lo que manda el stick, 'real' lo que")
  print("consigue el robot. Compara adelante y atras con el mismo empuje de stick.\n")

  # Bipedo: el frente del modelo es -x (ver ESTADO.md, "camino 1"): mirando hacia -x,
  # la izquierda del robot es -y. Por eso avance y lateral van negados; el giro no
  # cambia. Pentabot: frente = +x, sin correccion (FRENTE viene de ROBOTS).
  sa = FRENTE * (-1.0 if inv_avance else 1.0)
  sl = FRENTE * (-1.0 if inv_lateral else 1.0)
  sg = -1.0 if inv_giro else 1.0
  for nombre, invertido in (("avance", inv_avance), ("lateral", inv_lateral), ("giro", inv_giro)):
    if invertido:
      print(f"  OJO: eje '{nombre}' INVERTIDO por peticion tuya")

  compute_original = term.compute
  robot = base.scene["robot"]
  contador = {"n": 0}

  def compute_con_mando(dt: float) -> None:
    compute_original(dt)
    if mando.alto:
      term.vel_command_b[:, :] = 0.0
    else:
      term.vel_command_b[:, 0] = escalar(sa * mando.avance, *rx)
      term.vel_command_b[:, 1] = escalar(sl * mando.lateral, *ry)
      term.vel_command_b[:, 2] = escalar(sg * mando.giro, *rz)

    # Lectura en vivo: comando vs velocidad real, para no juzgar a ojo.
    contador["n"] += 1
    if contador["n"] % 12 == 0:
      c = term.vel_command_b[0]
      v = robot.data.root_link_lin_vel_b[0]
      wz_real = float(robot.data.root_link_ang_vel_b[0, 2])
      cx, cy, cz = float(c[0]), float(c[1]), float(c[2])
      sigue = 100.0 * float(v[0]) / cx if abs(cx) > 1e-3 else 0.0
      print(
        f"\rpide vx={cx:+.3f} vy={cy:+.3f} wz={cz:+.3f} | "
        f"real vx={float(v[0]):+.3f} vy={float(v[1]):+.3f} wz={wz_real:+.3f} | "
        f"sigue {sigue:5.1f}%   ",
        end="", flush=True,
      )

  term.compute = compute_con_mando  # type: ignore[method-assign]


def fijar_orientacion_inicial(cfg) -> None:
  """Hace que el robot nazca siempre mirando a +x.

  Solo afecta a como se pilota: no toca env_cfgs.py ni el entrenamiento.
  """
  rango = cfg.events["reset_base"].params["pose_range"]
  antes = tuple(rango.get("yaw", (0.0, 0.0)))
  rango["yaw"] = (0.0, 0.0)
  print(f"  orientacion inicial fijada a 0 rad (antes era aleatoria: {antes[0]:.2f}..{antes[1]:.2f})")


def envolver_viewer(clase, mando: MandoPS5, inversiones: dict | None = None):
  inversiones = inversiones or {}
  """Subclase que engancha el mando justo despues de crear el viewer."""

  class ViewerConMando(clase):  # type: ignore[valid-type,misc]
    def __init__(self, env, policy, *a, **k):
      super().__init__(env, policy, *a, **k)
      enganchar_mando(env, mando, **inversiones)

  return ViewerConMando


def main() -> int:
  if len(sys.argv) < 2:
    print(__doc__)
    print("Checkpoints disponibles:")
    for p in sorted(LOGS.glob("*/model_*.pt")):
      print("  ", p)
    return 1

  ckpt = resolver_checkpoint(sys.argv[1])
  visor = "auto"
  if "--viewer" in sys.argv:
    visor = sys.argv[sys.argv.index("--viewer") + 1]
  yaw_libre = "--yaw-libre" in sys.argv
  inversiones = {
    "inv_avance": "--invertir-avance" in sys.argv,
    "inv_lateral": "--invertir-lateral" in sys.argv,
    "inv_giro": "--invertir-giro" in sys.argv,
  }

  try:
    mando = MandoPS5()
  except RuntimeError as e:
    print(e)
    return 1
  print(f"Mando: {mando.nombre}  ({mando.ruta})")

  # Se sustituyen los viewers que play.py mira en su propio modulo.
  play_mod.NativeMujocoViewer = envolver_viewer(NativeMujocoViewer, mando, inversiones)
  play_mod.ViserPlayViewer = envolver_viewer(ViserPlayViewer, mando, inversiones)

  # play.py llama a load_env_cfg desde su propio modulo: se envuelve ahi.
  if not yaw_libre:
    cargar_original = play_mod.load_env_cfg

    def cargar_con_yaw_fijo(*a, **k):
      cfg = cargar_original(*a, **k)
      fijar_orientacion_inicial(cfg)
      return cfg

    play_mod.load_env_cfg = cargar_con_yaw_fijo
  else:
    print("  --yaw-libre: el robot nace mirando a cualquier lado (como al entrenar)")

  cfg = play_mod.PlayConfig(checkpoint_file=str(ckpt), viewer=visor)  # type: ignore[arg-type]
  try:
    play_mod.run_play(TAREA, cfg)
  finally:
    mando.cerrar()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
