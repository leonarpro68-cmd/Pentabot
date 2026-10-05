"""Graba la marcha del Pentabot con la politica entrenada (sin ventana) y genera el GIF.

Se ejecuta con el entorno de unitree_rl_mjlab y la tarea Pentabot-Flat ya instalada:
  python grabar_marcha.py [checkpoint.pt] [carpeta_salida] [tarea] [prefijo]
  p. ej.: python grabar_marcha.py ~/pentabot/politica/sin_encoder/model_5000.pt ~/pentabot/imagenes Pentabot-Flat-SinEncoder sin_encoder_
"""
import os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.expanduser("~/unitree_rl_mjlab"))
from dataclasses import asdict
import numpy as np, torch
from PIL import Image
import mjlab.tasks  # noqa
import src.tasks  # noqa
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

RAIZ = Path(__file__).resolve().parents[2]
CKPT = sys.argv[1] if len(sys.argv) > 1 else str(RAIZ / "politica" / "model_3000.pt")
OUT = sys.argv[2] if len(sys.argv) > 2 else str(RAIZ / "imagenes")
TASK = sys.argv[3] if len(sys.argv) > 3 else "Pentabot-Flat"
PREFIJO = sys.argv[4] if len(sys.argv) > 4 else ""
env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = 1
env_cfg.events["reset_base"].params["pose_range"]["yaw"] = (0.0, 0.0)
env_cfg.viewer.height, env_cfg.viewer.width = 540, 960
agent_cfg = load_rl_cfg(TASK)
env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode="rgb_array")
venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(venv, asdict(agent_cfg), device="cuda:0")
runner.load(CKPT, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
policy = runner.get_inference_policy(device="cuda:0")

term = env.command_manager.get_term("twist")
r = term.cfg.ranges
print("rangos:", r.lin_vel_x, r.lin_vel_y, r.ang_vel_z)
cmd = torch.zeros(3, device="cuda:0")
orig = term.compute
def compute(dt):
    orig(dt); term.vel_command_b[:, :] = cmd
term.compute = compute

vx = float(r.lin_vel_x[1]) * 0.8
plan = [((0, 0, 0), 50), ((vx, 0, 0), 250), ((0, 0, float(r.ang_vel_z[1]) * 0.8), 150),
        ((0, float(r.lin_vel_y[1]) * 0.8, 0), 150)]
obs = venv.get_observations()
frames = []
robot = env.scene["robot"]
for (c, n) in plan:
    cmd[:] = torch.tensor(c, device="cuda:0")
    vs = []
    for i in range(n):
        with torch.no_grad():
            obs, _, dones, _ = venv.step(policy(obs))
        if dones.any(): print("  caida/reset en", c, i)
        vs.append(robot.data.root_link_lin_vel_b[0, :2].cpu().numpy())
        if i % 2 == 0: frames.append(env.render())
    print(f"cmd {c}: vel media real {np.mean(vs[n//3:], axis=0)}")
# Fotos sueltas de cada tramo y GIF reducido (~3 MB) para el README.
for i, n in [(60, "avance"), (170, "giro"), (260, "lateral")]:
    Image.fromarray(frames[i]).save(f"{OUT}/{PREFIJO}marcha_{n}.png")
ims = [Image.fromarray(x[30:]).resize((480, 254), Image.LANCZOS) for x in frames[::2]]
base = ims[len(ims) // 3].quantize(colors=64, method=Image.Quantize.MEDIANCUT)
pal = [im.quantize(palette=base, dither=Image.Dither.NONE) for im in ims]
pal[0].save(f"{OUT}/{PREFIJO}pentabot_marcha.gif", save_all=True, append_images=pal[1:],
            duration=80, loop=0, optimize=True)
print("frames:", len(frames), "->", OUT)
