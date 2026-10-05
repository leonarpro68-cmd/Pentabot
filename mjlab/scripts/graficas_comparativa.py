import json, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as E
S, OUT = sys.argv[1], sys.argv[2]
SURF, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_ENC, C_SIN = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.facecolor": SURF, "figure.facecolor": SURF,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False})
R = "/home/leo/unitree_rl_mjlab/logs/rsl_rl/"
runs = {"Con encoders": (R+"pentabot_velocity/2026-09-30_21-00-04", 1.0, C_ENC),
        "Sin encoders": (R+"pentabot_sin_encoder/2026-10-04_20-22-36", 2.0, C_SIN)}
def serie(run, tag):
    e = E(run, size_guidance={"scalars": 0}); e.Reload()
    t = [x for x in e.Tags()["scalars"] if x.endswith(tag)][0]
    s = e.Scalars(t); return np.array([x.step for x in s]), np.array([x.value for x in s])
def suav(y, k=25): return np.convolve(np.pad(y, (k//2, k-1-k//2), mode="edge"), np.ones(k)/k, mode="valid")

# --- 1. curvas de entrenamiento (2 paneles, mismo eje x) ---
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
for nombre, (run, peso, c) in runs.items():
    x, y = serie(run, "Episode_Reward/track_linear_velocity"); y = suav(y / peso) * 100
    axs[0].plot(x, y, color=c, lw=2, label=nombre)
    axs[0].annotate(nombre, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", color=INK2, va="center", fontsize=10)
    x, y = serie(run, "Train/mean_episode_length"); y = suav(y) * 0.02
    axs[1].plot(x, y, color=c, lw=2, label=nombre)
for ax in axs:
    ax.axvline(2000, color=BASE, lw=1, ls=(0, (4, 3)))
    ax.set_xlabel("Iteración de entrenamiento"); ax.set_xlim(0, 5300)
axs[0].text(2050, 8, "comandos hasta ±0.2 m/s", color=MUTED, fontsize=9)
axs[0].set_title("Recompensa de seguimiento de velocidad (% del máximo)", loc="left", fontsize=12, color=INK)
axs[0].set_ylim(0, 100)
axs[1].set_title("Duración media del episodio (s, máx. 20)", loc="left", fontsize=12, color=INK)
axs[1].set_ylim(0, 21)
axs[1].legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/comparativa_entrenamiento.png", dpi=150); plt.close(fig)

# --- 2. evaluacion en condiciones reales (3 paneles de barras horizontales) ---
d = {m: json.load(open(f"{S}/{m}_reales.json"))["filas"] for m in ("encoders", "sin_encoder")}
cmds = [f["comando"] for f in d["encoders"]]
metr = [("sigue", "Sigue la velocidad pedida (%)", "más es mejor"), ("caidas", "Caídas (%)", "menos es mejor"),
        ("par_p95", "Par de los servos, p95 (% del límite)", "menos es mejor")]
fig, axs = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
yy = np.arange(len(cmds)); h = 0.36
for ax, (k, titulo, sub) in zip(axs, metr):
    for j, (m, c, lab) in enumerate((("encoders", C_ENC, "Con encoders"), ("sin_encoder", C_SIN, "Sin encoders"))):
        v = np.array([f[k] if f[k] == f[k] else 0 for f in d[m]])
        pos = yy - h/2 - 0.02 + j*(h + 0.04)
        ax.barh(pos, v, height=h, color=c, label=lab)
        for p, val, f in zip(pos, v, d[m]):
            if k == "sigue" and f["comando"] == "quieto": continue
            ax.annotate(f"{val:.0f}", (val, p), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    ax.set_title(titulo + "\n", loc="left", fontsize=11.5, color=INK); ax.text(0, 1.015, sub, transform=ax.transAxes, color=MUTED, fontsize=9, va="bottom")
    ax.grid(axis="y", visible=False); ax.set_yticks(yy, cmds); ax.invert_yaxis()
axs[0].set_xlim(0, 110); axs[1].set_xlim(0, 12); axs[2].set_xlim(0, 125)
axs[2].axvline(100, color=BASE, lw=1, ls=(0, (4, 3))); axs[2].text(99, -0.62, "límite", color=MUTED, fontsize=8.5, ha="right")
axs[0].text(2, 6, "(quieto: no aplica)", color=MUTED, fontsize=8.5, va="center")
fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, loc="upper right", ncol=2, fontsize=10, bbox_to_anchor=(0.99, 0.995))
fig.suptitle("Evaluación en condiciones de robot real: 256 robots aleatorizados, IMU con ruido, 8 s por comando",
             x=0.01, ha="left", fontsize=12.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(f"{OUT}/comparativa_condiciones_reales.png", dpi=150); plt.close(fig)
print("ok")
