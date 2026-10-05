"""Renders estaticos del Pentabot (sin ventana).  Uso: python render_estatico.py [carpeta_salida]"""
import os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco, numpy as np
from PIL import Image
RAIZ = Path(__file__).resolve().parents[2]
M = str(RAIZ / "model" / "scene.xml")
OUT = sys.argv[1] if len(sys.argv) > 1 else str(RAIZ / "imagenes")
m = mujoco.MjModel.from_xml_path(M); d = mujoco.MjData(m)
print("keys:", [m.key(i).name for i in range(m.nkey)])
k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(m, d, k); mujoco.mj_forward(m, d)
for _ in range(300): mujoco.mj_step(m, d)  # que se asiente
r = mujoco.Renderer(m, 1080, 1920)
cam = mujoco.MjvCamera(); cam.lookat[:] = d.qpos[:3] + [0,0,-0.01]
opt = mujoco.MjvOption()
def shot(name, az, el, dist, groups=None):
    cam.azimuth, cam.elevation, cam.distance = az, el, dist
    o = mujoco.MjvOption()
    if groups is not None:
        o.geomgroup[:] = 0
        for g in groups: o.geomgroup[g] = 1
    r.update_scene(d, cam, o)
    Image.fromarray(r.render()).save(f"{OUT}/{name}.png"); print(name)
shot("pentabot_iso", 135, -25, 0.95)
shot("pentabot_frente", 0, -10, 0.7)
shot("pentabot_lateral", 90, -5, 0.7)
shot("pentabot_arriba", 90, -89, 0.9)
shot("pentabot_colisiones", 135, -25, 0.95, groups=[0,1,3])
