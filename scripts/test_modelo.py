"""Prueba sin ventana: carga el modelo, lo deja de pie 3 s y revisa estabilidad, cargas y pares."""
import os, numpy as np, mujoco

XML = os.path.join(os.path.dirname(__file__), "..", "model", "scene.xml")
m = mujoco.MjModel.from_xml_path(XML); d = mujoco.MjData(m)
legs = [f"L{i}" for i in range(1, 6)]
print(f"MuJoCo {mujoco.__version__} | cuerpos={m.nbody} articulaciones={m.njnt} actuadores={m.nu}")
print(f"Masa total: {m.body_subtreemass[1]:.3f} kg")

mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
for _ in range(int(3 / m.opt.timestep)):
    mujoco.mj_step(m, d)

tilt = np.degrees(2 * np.arccos(min(1.0, abs(d.qpos[3]))))
fz = [d.sensor(f"{n}_foot_touch").data[0] for n in legs]
tau = np.abs(d.actuator_force).reshape(5, 3).max(0)
print(f"Altura torso: {d.qpos[2]:.3f} m | inclinación: {tilt:.3f}°")
print("Fuerza por pie (N):", np.round(fz, 2), f"| suma {sum(fz):.2f} vs peso {m.body_subtreemass[1]*9.81:.2f}")
print("Par máx [hip_yaw, hip_pitch, knee] (N·m):", np.round(tau, 3), f"(bloqueo MG995 = {m.actuator_forcerange[0,1]:.3f})")
ok = tilt < 1 and abs(sum(fz) - m.body_subtreemass[1] * 9.81) < 0.5 and not np.isnan(d.qpos).any()
print("RESULTADO:", "OK" if ok else "REVISAR")
