"""Evalua la politica Pentabot-Flat-SinEncoder en condiciones parecidas al robot real.

Uso:  python pentabot/evaluar_sin_encoder.py <checkpoint.pt | ultimo> [--envs 256]

Cada env tiene su propia aleatorizacion (kp, par, retardo, cero del servo, masa,
friccion) y el ruido y el sesgo del IMU quedan ACTIVOS (en play se apagan). Para cada
comando fijo se descartan 2 s de arranque y se miden 8 s:
  sigue   velocidad real / pedida en la direccion pedida (%)
  caidas  envs que terminaron por caida o contacto ilegal (%)
  <3 pies fraccion del tiempo con menos de 3 pies en el suelo (%)
  par p95 percentil 95 del par de los servos respecto a su limite (%)
  incl    inclinacion media del torso (grados)
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(os.environ.get("UNITREE_RL_MJLAB", "~/unitree_rl_mjlab")).expanduser()
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import mjlab.tasks  # noqa: E402,F401
import src.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402

TAREA = "Pentabot-Flat-SinEncoder"
LOGS = REPO / "logs" / "rsl_rl" / "pentabot_sin_encoder"
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


def resolver(arg: str) -> Path:
  if arg != "ultimo":
    return Path(arg).resolve()
  ultima = sorted((d for d in LOGS.glob("*/") if d.is_dir()), key=lambda d: d.stat().st_mtime)[-1]
  modelo = sorted(ultima.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))[-1]
  print(f"ultimo -> {modelo.name} (corrida {ultima.name})")
  return modelo


def main() -> int:
  if len(sys.argv) < 2:
    print(__doc__)
    return 1
  ckpt = resolver(sys.argv[1])
  n = int(sys.argv[sys.argv.index("--envs") + 1]) if "--envs" in sys.argv else 256
  dev = "cuda:0" if torch.cuda.is_available() else "cpu"

  cfg = load_env_cfg(TAREA, play=True)
  cfg.scene.num_envs = n
  cfg.observations["actor"].enable_corruption = True  # IMU con ruido y sesgo, como el real
  cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)
  agent_cfg = load_rl_cfg(TAREA)
  venv = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=dev), clip_actions=agent_cfg.clip_actions)
  runner = (load_runner_cls(TAREA) or MjlabOnPolicyRunner)(venv, asdict(agent_cfg), device=dev)
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

  print(f"\n{ckpt}\n{n} envs, IMU con ruido, {MEDIDA_S:.0f} s por comando\n")
  print(f"{'comando':15s} {'pide':>17s} {'real':>17s} {'sigue':>6s} {'caidas':>7s} {'<3 pies':>8s} {'par p95':>8s} {'incl':>5s}")
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
        v = torch.cat((robot.data.root_link_lin_vel_b[:, :2], robot.data.root_link_ang_vel_b[:, 2:3]), dim=1)
        vel += v[vivos].mean(0) / pasos_medida if vivos.any() else 0
        pocos_pies += ((pies.data.found > 0).sum(1) < 3)[vivos].float().mean().item() / pasos_medida
        pares.append((robot.data.actuator_force.abs() / limite)[vivos].flatten())
        g = robot.data.projected_gravity_b[vivos]
        incl += torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1, 1))).mean().item() / pasos_medida
    p95 = torch.quantile(torch.cat(pares)[:1_000_000], 0.95).item()
    pedida = torch.tensor((vx, vy, wz), device=dev)
    norma = pedida.norm().item()
    sigue = 100 * float(vel @ pedida) / norma**2 if norma > 0 else float("nan")
    print(
      f"{nombre:15s} ({vx:+.2f},{vy:+.2f},{wz:+.2f}) ({vel[0]:+.2f},{vel[1]:+.2f},{vel[2]:+.2f})"
      f" {sigue:5.0f}% {100 * caido.float().mean().item():6.1f}% {100 * pocos_pies:7.1f}%"
      f" {100 * p95:7.0f}% {incl:5.1f}"
    )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
