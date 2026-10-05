"""Evalua las politicas del Pentabot en condiciones parecidas al robot real.

Uso:  python pentabot/evaluar_sin_encoder.py <checkpoint.pt | ultimo>
          [--modelo sin_encoder|encoders] [--condiciones reales|propias] [--envs 256] [--json salida.json]

  --modelo sin_encoder  tarea Pentabot-Flat-SinEncoder (por defecto)
  --modelo encoders     tarea Pentabot-Flat (la primera, que lee joint_pos/joint_vel)
  --condiciones reales  (por defecto) la aleatorizacion del robot real que se uso para entrenar la
                        version sin encoders, aplicada a cualquiera de los dos modelos: servos de kp
                        0.6-1.4 y par +-20 %, retardo de 4-32 ms, cero del servo +-3 grados, juego de
                        engranajes +-1 grado, torso x0.8-1.4, friccion 0.4-1.2 e IMU con ruido, sesgo
                        y retardo
  --condiciones propias la aleatorizacion con la que se entreno cada modelo

En los dos casos el ruido de las observaciones queda ACTIVO (en play se apaga). Para cada comando fijo
se descartan 2 s de arranque y se miden 8 s:
  sigue   velocidad real / pedida en la direccion pedida (%)
  caidas  envs que terminaron por caida o contacto ilegal (%)
  <3 pies fraccion del tiempo con menos de 3 pies en el suelo (%)
  par p95 percentil 95 del par de los servos respecto a su limite (%)
  incl    inclinacion media del torso (grados)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

REPO = Path(os.environ.get("UNITREE_RL_MJLAB", "~/unitree_rl_mjlab")).expanduser()
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import mjlab.tasks  # noqa: E402,F401
import src.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.managers.event_manager import EventTermCfg  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.utils.noise import UniformNoiseCfg as Unoise  # noqa: E402
from mjlab.utils.noise.noise_cfg import NoiseModelWithAdditiveBiasCfg  # noqa: E402
from src.assets.robots import PENTABOT_ACTION_SCALE  # noqa: E402
from src.tasks.velocity.config.pentabot.sin_encoder import AccionServoCfg, limites_par  # noqa: E402

MODELOS = {
  "sin_encoder": ("Pentabot-Flat-SinEncoder", "pentabot_sin_encoder"),
  "encoders": ("Pentabot-Flat", "pentabot_velocity"),
}
ARRANQUE_S = 2.0
MEDIDA_S = 8.0

# (nombre, vx, vy, wz); frente del Pentabot = +x.
COMANDOS = [
  ("adelante 0.20", 0.20, 0.0, 0.0),
  ("adelante 0.10", 0.10, 0.0, 0.0),
  ("atras    0.15", -0.15, 0.0, 0.0),
  ("lateral  0.15", 0.0, 0.15, 0.0),
  ("diagonal 0.10", 0.07, 0.07, 0.0),
  ("giro     0.50", 0.0, 0.0, 0.5),
  ("quieto",        0.0, 0.0, 0.0),
]


def arg(nombre: str, defecto: str) -> str:
  return sys.argv[sys.argv.index(nombre) + 1] if nombre in sys.argv else defecto


def resolver(arg_ckpt: str, carpeta: str) -> Path:
  if arg_ckpt != "ultimo":
    return Path(arg_ckpt).resolve()
  logs = REPO / "logs" / "rsl_rl" / carpeta
  ultima = sorted((d for d in logs.glob("*/") if d.is_dir()), key=lambda d: d.stat().st_mtime)[-1]
  modelo = sorted(ultima.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))[-1]
  print(f"ultimo -> {modelo.name} (corrida {ultima.name})")
  return modelo


def condiciones_reales(cfg) -> None:
  """Misma aleatorizacion que Pentabot-Flat-SinEncoder (sin_encoder.py), mas el juego de engranajes."""
  cfg.actions["joint_pos"] = AccionServoCfg(
    entity_name="robot", actuator_names=(".*",), scale=PENTABOT_ACTION_SCALE,
    use_default_offset=True, ruido_objetivo=0.02,
  )
  terms = cfg.observations["actor"].terms
  terms["base_ang_vel"].noise = NoiseModelWithAdditiveBiasCfg(
    noise_cfg=Unoise(n_min=-0.15, n_max=0.15), bias_noise_cfg=Unoise(n_min=-0.05, n_max=0.05))
  terms["projected_gravity"].noise = NoiseModelWithAdditiveBiasCfg(
    noise_cfg=Unoise(n_min=-0.03, n_max=0.03), bias_noise_cfg=Unoise(n_min=-0.035, n_max=0.035))
  for t in ("base_ang_vel", "projected_gravity"):
    terms[t].delay_min_lag, terms[t].delay_max_lag = 0, 1
  cfg.events["servo_gains"].params["kp_range"] = (0.6, 1.4)
  cfg.events["servo_par"] = EventTermCfg(mode="startup", func=limites_par, params={"rango": (0.8, 1.2)})
  cfg.events["encoder_bias"].params["bias_range"] = (-0.05, 0.05)
  cfg.events["torso_mass"].params["ranges"] = (0.8, 1.4)
  cfg.events["foot_friction"].params["ranges"] = (0.4, 1.2)
  cfg.events["base_com"].params["ranges"] = {0: (-0.02, 0.02), 1: (-0.02, 0.02), 2: (-0.01, 0.01)}
  cfg.events["servo_friction"].params["ranges"] = (0.5, 1.5)
  cfg.events["reset_robot_joints"].params["position_range"] = (-0.1, 0.1)
  robot = cfg.scene.entities["robot"]
  robot.articulation = replace(
    robot.articulation,
    actuators=tuple(replace(a, delay_min_lag=1, delay_max_lag=8) for a in robot.articulation.actuators),
  )


def main() -> int:
  if len(sys.argv) < 2 or sys.argv[1].startswith("--"):
    print(__doc__)
    return 1
  modelo = arg("--modelo", "sin_encoder")
  condiciones = arg("--condiciones", "reales")
  n = int(arg("--envs", "256"))
  tarea, carpeta = MODELOS[modelo]
  ckpt = resolver(sys.argv[1], carpeta)
  dev = "cuda:0" if torch.cuda.is_available() else "cpu"

  cfg = load_env_cfg(tarea, play=True)
  cfg.scene.num_envs = n
  cfg.observations["actor"].enable_corruption = True
  cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)
  if condiciones == "reales":
    condiciones_reales(cfg)
  agent_cfg = load_rl_cfg(tarea)
  venv = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=dev), clip_actions=agent_cfg.clip_actions)
  runner = (load_runner_cls(tarea) or MjlabOnPolicyRunner)(venv, asdict(agent_cfg), device=dev)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=dev)
  policy = runner.get_inference_policy(device=dev)

  env = venv.unwrapped
  robot = env.scene["robot"]
  pies = env.scene["feet_ground_contact"]
  term = env.command_manager.get_term("twist")
  cmd = torch.zeros(3, device=dev)
  original = term.compute

  def compute(dt: float) -> None:
    original(dt)
    term.vel_command_b[:, :] = cmd

  term.compute = compute  # type: ignore[method-assign]
  limite = env.sim.model.actuator_forcerange[:, robot.indexing.ctrl_ids, 1]
  pasos_arranque = int(ARRANQUE_S / env.step_dt)
  pasos_medida = int(MEDIDA_S / env.step_dt)

  print(f"\n{ckpt}\nmodelo {modelo}, condiciones {condiciones}, {n} envs, {MEDIDA_S:.0f} s por comando\n")
  print(f"{'comando':15s} {'pide':>17s} {'real':>17s} {'sigue':>6s} {'caidas':>7s} {'<3 pies':>8s} {'par p95':>8s} {'incl':>5s}")
  filas = []
  for nombre, vx, vy, wz in COMANDOS:
    cmd[:] = torch.tensor((vx, vy, wz), device=dev)
    obs, _ = venv.reset()
    caido = torch.zeros(n, dtype=torch.bool, device=dev)
    vel = torch.zeros(3, device=dev)
    pocos_pies = 0.0
    pares, incl = [], 0.0
    for i in range(pasos_arranque + pasos_medida):
      with torch.no_grad():
        obs, _, dones, extras = venv.step(policy(obs))
      timeouts = extras.get("time_outs", torch.zeros_like(dones)).bool()
      if i >= pasos_arranque:
        caido |= dones.bool() & ~timeouts
        vivos = ~caido
        if not vivos.any():
          continue
        v = torch.cat((robot.data.root_link_lin_vel_b[:, :2], robot.data.root_link_ang_vel_b[:, 2:3]), dim=1)
        vel += v[vivos].mean(0) / pasos_medida
        pocos_pies += ((pies.data.found > 0).sum(1) < 3)[vivos].float().mean().item() / pasos_medida
        pares.append((robot.data.actuator_force.abs() / limite)[vivos].flatten())
        g = robot.data.projected_gravity_b[vivos]
        incl += torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1, 1))).mean().item() / pasos_medida
    p95 = torch.quantile(torch.cat(pares)[:1_000_000], 0.95).item() if pares else float("nan")
    pedida = torch.tensor((vx, vy, wz), device=dev)
    norma = pedida.norm().item()
    sigue = 100 * float(vel @ pedida) / norma**2 if norma > 0 else float("nan")
    fila = {
      "comando": nombre.strip(), "pide": [vx, vy, wz], "real": [round(float(x), 3) for x in vel],
      "sigue": sigue, "caidas": 100 * caido.float().mean().item(), "menos_3_pies": 100 * pocos_pies,
      "par_p95": 100 * p95, "incl": incl,
    }
    filas.append(fila)
    print(
      f"{nombre:15s} ({vx:+.2f},{vy:+.2f},{wz:+.2f}) ({vel[0]:+.2f},{vel[1]:+.2f},{vel[2]:+.2f})"
      f" {sigue:5.0f}% {fila['caidas']:6.1f}% {fila['menos_3_pies']:7.1f}% {fila['par_p95']:7.0f}% {incl:5.1f}"
    )
  if "--json" in sys.argv:
    Path(arg("--json", "")).write_text(json.dumps(
      {"modelo": modelo, "condiciones": condiciones, "checkpoint": str(ckpt), "envs": n, "filas": filas}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
