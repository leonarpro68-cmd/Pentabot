"""Marcha de onda (wave gait) en lazo abierto: solo para verificar que el modelo camina.
Cada pata levanta por turnos (1 de 5 en el aire) y las 4 de apoyo empujan el cuerpo en +Y.
No es un controlador óptimo: es el punto de partida antes de entrenar.
Uso:  python scripts/demo_marcha.py          (con visor)
      python scripts/demo_marcha.py --sin-ventana   (solo imprime el avance)
"""
import os, sys, time, numpy as np, mujoco

XML = os.path.join(os.path.dirname(__file__), "..", "model", "scene.xml")
m = mujoco.MjModel.from_xml_path(XML); d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)

T = 2.0          # periodo del ciclo [s]
DUTY = 0.8       # fracción en apoyo (1 pata en el aire a la vez)
STEP = 0.035     # avance del pie por paso [m]
LIFT = 0.40      # cuánto sube el fémur en vuelo [rad]

# ángulo de montaje de cada cadera y radio del pie (en pose home)
mujoco.mj_forward(m, d)
phi, rad = [], []
for i in range(1, 6):
    p = d.site(f"L{i}_foot").xpos[:2] - d.qpos[:2]
    phi.append(np.arctan2(p[1], p[0])); rad.append(np.linalg.norm(p))
# giro de cadera necesario para mover el pie -STEP/2..+STEP/2 en Y (empuja el cuerpo hacia +Y)
amp = np.clip([STEP / 2 / (r * np.cos(f)) for f, r in zip(phi, rad)], -0.35, 0.35)
order = [0, 2, 4, 1, 3]                 # secuencia de levantado L1, L3, L5, L2, L4

def targets(t):
    q = np.zeros(15)
    for k, leg in enumerate(order):
        s = (t / T - k / 5) % 1.0       # fase de esa pata
        if s < DUTY:                    # apoyo: barre de +amp a -amp (empuja)
            u = s / DUTY; yaw = amp[leg] * (1 - 2 * u); lift = 0
        else:                           # vuelo: vuelve de -amp a +amp levantada
            u = (s - DUTY) / (1 - DUTY); yaw = amp[leg] * (2 * u - 1); lift = LIFT * np.sin(np.pi * u)
        q[3*leg + 0] = yaw
        q[3*leg + 1] = lift
        q[3*leg + 2] = 0.5 * lift       # la tibia acompaña para despegar el pie
    return q

def step():
    d.ctrl[:] = targets(d.time); mujoco.mj_step(m, d)

if "--sin-ventana" in sys.argv:
    y0 = d.qpos[:2].copy()
    while d.time < 20: step()
    dx = d.qpos[:2] - y0
    print(f"20 s: avance = {dx[1]*100:.1f} cm en Y, deriva X = {dx[0]*100:.1f} cm, z torso = {d.qpos[2]:.3f} m")
else:
    import mujoco.viewer
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            t0 = time.time()
            for _ in range(int(0.02 / m.opt.timestep)): step()
            v.sync()
            time.sleep(max(0, 0.02 - (time.time() - t0)))
