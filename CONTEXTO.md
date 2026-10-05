# Contexto del proyecto Pentabot (para trabajar en otra computadora)

> Documento de traspaso. Explica qué es el proyecto, en qué punto está, cómo montarlo desde
> cero en tu máquina y qué falta. También sirve como contexto para un asistente de IA:
> pásale este archivo al empezar.
>
> Actualizado: 2026-10-05.

---

## 1. Qué es

El **Pentabot** es un robot caminante de **5 patas** separadas 72°. Cada pata tiene **3 articulaciones**
(`hip_yaw`, `hip_pitch`, `knee`), así que son 15 en total, y cada una la mueve un servo **TowerPro MG995** a 6 V.

- El modelo de MuJoCo se generó desde el CAD de Inventor (`CuerpoBase.iam`).
- La marcha se entrena con **aprendizaje por refuerzo (PPO)** en simulación sobre
  [`unitree_rl_mjlab`](https://github.com/unitreerobotics/unitree_rl_mjlab), que está basado en
  [`mjlab`](https://github.com/mujocolab/mjlab) (MuJoCo + MuJoCo Warp en la GPU).
- **Objetivo final:** que el robot real camine controlado con un mando de PS5. La política correrá en
  una **ESP32 DevKit**, sin computadora a bordo.

| Dato | Valor |
|---|---|
| Masa del modelo | 2.47 kg (incluye 0.18 kg estimados para la electrónica, sin pesar) |
| Altura del torso de pie | 0.196 m (con q = 0 en todas las articulaciones) |
| Servo | MG995: par de bloqueo 0.98 N·m, ~6.5 rad/s en vacío, **sin realimentación de posición** |
| Electrónica prevista | ESP32 DevKit + PCA9685 (PWM de servos, 50 Hz) + IMU + DualSense por Bluetooth |
| Frente del robot | **+x** del modelo |

---

## 2. Repositorios y dónde vive cada cosa

Hay **dos sitios**, y es importante no confundirlos:

1. **`Pentabot`** (este repo, privado en GitHub de `leonarpro68-cmd`) contiene todo lo propio del robot:
   ```
   model/            pentabot.xml (robot), scene.xml (robot + suelo), assets/*.stl (26 mallas)
   scripts/          visor, test del modelo y marcha en lazo abierto (solo MuJoCo, sin RL)
   mjlab/robot/      pentabot_constants.py → robot y servos MG995 para mjlab
   mjlab/tarea/      tareas de RL: env_cfgs.py (Pentabot-Flat), sin_encoder.py (Pentabot-Flat-SinEncoder),
                     rl_cfg.py (PPO), __init__.py (registro de las tareas)
   mjlab/scripts/    entrenar, ver, mando PS5, evaluar, grabar el GIF y los renders
   politica/         con encoders (model_3000.pt, policy.onnx) y sin_encoder/ (la desplegable)
   docs/             MODELO.md (cómo se hizo el modelo) y PENDIENTE.md (plan de despliegue)
   imagenes/         renders de la simulación
   ```
2. **`unitree_rl_mjlab`** (repo público de Unitree) es el entorno donde **se entrena**. El código del
   Pentabot tiene que estar *dentro* de él, en rutas fijas, porque las tareas se importan desde `src/`.
   La sección 3 explica cómo enlazarlo.

> ⚠️ En la máquina de Leo, `unitree_rl_mjlab/src/...` tiene **copias** de los archivos, no enlaces.
> Si tú usas enlaces simbólicos, como se recomienda abajo, tendrás una sola fuente de verdad: este repo.
> Al hacer cambios, haz commit aquí.

---

## 3. Montarlo en tu computadora

### Requisitos
- Ubuntu 22.04 o 24.04, con **GPU NVIDIA** (driver ≥ 550). Leo usa una RTX 4060 Laptop (8 GB).
- Con 4096 entornos en paralelo, 5000 iteraciones tardan unas 5 h en esa GPU.

Versiones con las que funciona:

| Paquete | Versión |
|---|---|
| Python | 3.11 |
| mjlab | 1.2.0 |
| mujoco / mujoco-warp | 3.5.0 |
| torch | 2.14.0 |
| rsl-rl-lib | 5.0.1 |
| unitree_rl_mjlab | commit `1425b15` |

### Pasos

```bash
# 1. Entorno conda
conda create -n unitree_rl_mjlab python=3.11 -y
conda activate unitree_rl_mjlab

# 2. unitree_rl_mjlab en el commit probado
cd ~
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git
cd unitree_rl_mjlab && git checkout 1425b15
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
pip install -e .

# 3. Este repo (necesitas acceso de colaborador al repo privado)
cd ~ && git clone https://github.com/leonarpro68-cmd/Pentabot.git pentabot

# 4. Enlazar el Pentabot dentro de unitree_rl_mjlab
U=~/unitree_rl_mjlab
ln -s ~/pentabot              $U/src/assets/robots/Pentabot_MuJoCo
ln -s ~/pentabot/mjlab/robot  $U/src/assets/robots/pentabot
ln -s ~/pentabot/mjlab/tarea  $U/src/tasks/velocity/config/pentabot
cp ~/pentabot/mjlab/scripts/{jugar_mando.py,mando_ps5.py} $U/
```

**5.** Registra el robot añadiendo esto al **final** de `~/unitree_rl_mjlab/src/assets/robots/__init__.py`:

```python
from .pentabot.pentabot_constants import (
  PENTABOT_ACTION_SCALE as PENTABOT_ACTION_SCALE,
)
from .pentabot.pentabot_constants import (
  get_pentabot_robot_cfg as get_pentabot_robot_cfg,
)
```

> `jugar_mando.py` también tiene una entrada `bipedo` (otro robot de Leo). Si no tienes ese robot, usa
> siempre `--robot pentabot`. Si el import de `bipedo` falla, borra esas líneas.

### Comprobar que funciona

```bash
cd ~/unitree_rl_mjlab
unset PYTHONPATH                     # si tienes ROS instalado, su PYTHONPATH rompe el entorno
conda activate unitree_rl_mjlab

python ~/pentabot/scripts/test_modelo.py           # debe terminar con "RESULTADO: OK"
python scripts/train.py Pentabot-Flat-SinEncoder --env.scene.num-envs=256 --agent.max-iterations=3
#   debe imprimir: actor shape (130,), critic shape (89,) y 3 iteraciones sin errores
```

> Los scripts `.sh` de `mjlab/scripts/` asumen que conda está en `~/miniconda3` y que el repo está en
> `~/unitree_rl_mjlab`. Si en tu máquina es distinto, edita las primeras líneas.

---

## 4. Historia y estado actual

### Fase 1 (hecha): `Pentabot-Flat`, con encoders
- Entrenada con 3000 iteraciones, en unas 3 h y sin caídas. Está en `politica/model_3000.pt`.
- Sigue alrededor del 77 % de la velocidad pedida. En el GIF del README pide 0.16 m/s y consigue 0.15 m/s.
- **No sirve en el robot real**: usa `joint_pos` y `joint_vel` (30 de sus 56 entradas), y el MG995 no
  informa de su posición.

### Fase 2 (hecha): `Pentabot-Flat-SinEncoder`
- Archivo: `mjlab/tarea/sin_encoder.py`; PPO en `rl_cfg.py` (`pentabot_sin_encoder_ppo_runner_cfg`).
- Entrenada del 2026-10-04 al 2026-10-05: 4096 entornos, 5000 iteraciones, ~5 h en una RTX 4060.
- Política en **`politica/sin_encoder/`**: `model_5000.pt`, `policy.onnx` (entrada `obs` [1, 130],
  salida `actions` [1, 15]) y `params/`.
- Evaluación con `evaluar_sin_encoder.py`: 256 robots con toda la aleatorización y el IMU con ruido y sesgo.

| Comando | Sigue | Caídas | < 3 pies | Par p95 | Inclinación |
|---|---|---|---|---|---|
| adelante 0.20 m/s | 83 % | 0 % | 1.3 % | 72 % | 1.2° |
| adelante 0.10 m/s | 85 % | 0 % | 0.1 % | 65 % | 1.2° |
| atrás 0.15 m/s | 88 % | 0 % | 0.4 % | 69 % | 2.0° |
| lateral 0.15 m/s | 90 % | 0 % | 0.3 % | 70 % | 1.3° |
| diagonal 0.10 m/s | 87 % | 0 % | 0.1 % | 67 % | 1.2° |
| giro 0.50 rad/s | 84 % | 0 % | 0.1 % | 72 % | 1.5° |
| quieto | — | 0 % | 0 % | **89 %** | 0.8° |

- Cumple todos los criterios de la sección 7 **salvo el par en parado**. Cuando está quieto con las 5 patas
  apoyadas, los servos se empujan entre sí: `hip_yaw` y `hip_pitch` trabajan de media al 45-48 % de su
  límite y pasan el 11-13 % del tiempo por encima del 80 %. El motivo es el error de cero del servo
  (±3° en la aleatorización): con 5 apoyos la cadena queda cerrada, y con kp = 11.2 N·m/rad un error
  de 0.05 rad ya supone 0.56 N·m, el 57 % del bloqueo. Al caminar baja (p95 de 56-81 %) porque los
  pies se levantan y se recolocan.
  **Consecuencia práctica: calibrar cada servo con un error menor de 1°.** Si no, el robot real se
  calentará estando quieto.
- El "seguimiento" de tensorboard (`track_linear_velocity` ≈ 1.35 de 2.0) no es un porcentaje de
  velocidad: es un kernel exponencial. El número que importa es el `sigue` de esta tabla.

### Fase 3 (pendiente): ejecutar la política en la ESP32
Ver la sección 8 y `docs/PENDIENTE.md`.

---

## 5. Diseño de la política sin encoders

### Qué ve el actor (130 entradas = 26 por paso × 5 pasos de historial)
El actor solo recibe lo que la ESP32 puede medir de verdad:

| Término | Dim | Origen en el robot real |
|---|---|---|
| `base_ang_vel` | 3 | giroscopio del IMU (rad/s) |
| `projected_gravity` | 3 | orientación del IMU → vector gravedad unitario en el marco del torso |
| `command` | 3 | stick del DualSense → (vx, vy, wz) |
| `phase` | 2 | reloj propio: sin/cos de 2π·t/0.8 s; (0, 0) si \|v_xy\| + \|wz\| ≤ 0.05 |
| `actions` | 15 | la última acción cruda que la propia red envió |

**El orden importa para la ESP32.** El historial va **por término**, no por paso. Cada término guarda
sus 5 pasos, del más viejo (t−4) al más nuevo (t), y después se concatenan los términos:

```
[  0: 15] base_ang_vel       5 x 3
[ 15: 30] projected_gravity  5 x 3
[ 30: 45] command            5 x 3
[ 45: 55] phase              5 x 2
[ 55:130] actions            5 x 15
```

Al arrancar, el historial se llena repitiendo la primera lectura. La red es una MLP
130 → 256 → 128 → 128 → 15 con activación ELU, unos 84 000 parámetros (~336 KB en float32). Lleva un
normalizador de entradas (media y desviación) que se exporta dentro del ONNX.

**Salida:** `objetivo[i] = 0.3 × acción[i]` en radianes, relativo a q = 0, a 50 Hz.

### Qué ve el crítico (solo al entrenar)
Todo lo anterior sin ruido, más `joint_pos`, `joint_vel`, velocidad lineal, altura de los pies, tiempo
en el aire, contactos y fuerzas de contacto. El robot real nunca lo necesita.

### Aleatorización: lo que no se sabe del robot real
| Qué | Rango | Por qué |
|---|---|---|
| kp del servo | ×0.6–1.4 | el MG995 no está identificado y varía entre unidades |
| Par máximo | ×0.8–1.2 | ídem |
| Retardo del servo | 4–32 ms | bucle de la ESP32 + I2C + PWM de 20 ms del PCA9685 |
| Cero del servo | ±0.05 rad (±3°) | calibración en µs imperfecta |
| Juego de engranajes | ±0.02 rad por paso | holgura del MG995 |
| IMU | ruido + sesgo constante (~2°) + 0–20 ms de retardo | montaje torcido, deriva, filtro |
| Masa del torso | ×0.8–1.4 | batería y electrónica sin pesar |
| Fricción del pie | 0.4–1.2 | piso desconocido |
| CoM del torso | ±2 cm | |
| Fricción y armadura de las articulaciones | ×0.5–1.5 / ×0.7–1.3 | |
| Empujones | cada 5–6 s | robustez |

### Recompensas (pesos finales)
| Término | Peso | Idea |
|---|---|---|
| `track_linear_velocity` | +2.0 | seguir vx, vy (std 0.1) |
| `track_angular_velocity` | +1.0 | seguir wz (std 0.25) |
| `foot_gait` | +1.0 | marcha de onda: 1 pata en el aire, apoyo 80 %, orden L1-L3-L5-L2-L4, periodo 0.8 s |
| `pose` | +1.0 | cerca de q = 0, más tolerante al caminar |
| `body_orientation_l2` | −2.0 | torso plano |
| `altura_torso` | −100 (×Δz²) | torso a 0.19 m, sin arrastrar la panza |
| `apoyos_minimos` | −1.0 | castiga tener menos de 3 pies en el suelo (con 3 apoyos el MG995 ya trabaja al 64 %) |
| `foot_clearance` | −3.0 | levantar el pie unos 3.5 cm al moverlo |
| `foot_slip` | −2.0 | no arrastrar los pies apoyados |
| `soft_landing` | −2e-3 | aterrizar suave |
| `action_rate_l2` / `action_acc_l2` | −0.1 / −0.05 | objetivos suaves: en lazo abierto un salto brusco no se corrige |
| `torques` | −0.02 | ahorrar par |
| `saturacion_servo` | −0.5 | castiga el par por encima del 80 % del límite de cada servo |
| `velocidad_servo` | −0.1 | castiga más de 5 rad/s, que el MG995 real no alcanza |
| `stand_still` | −1.0 | quieto si el comando es 0 |
| `is_terminated` | −200 | caerse (torso inclinado > 45° o cualquier parte que no sea el pie toca el suelo) |

Todas las recompensas que dependen del comando usan **el mismo umbral, 0.05**, igual que la
observación `phase`. En la tarea base la fase se apagaba con 0.1 mientras la marcha ya se exigía desde
0.05, y eso confundía a un actor que no tiene otra referencia de tiempo.

Comandos: vx, vy ∈ [−0.2, 0.2] m/s y wz ∈ [−0.6, 0.6] rad/s. Durante las primeras 2000 iteraciones
se limitan a la mitad (currículo).

---

## 6. Comandos del día a día

```bash
cd ~/unitree_rl_mjlab; unset PYTHONPATH; conda activate unitree_rl_mjlab

# Entrenar (lanza en segundo plano y abre tensorboard en :6006)
~/pentabot/mjlab/scripts/entrenar_pentabot.sh 4096 Pentabot-Flat-SinEncoder
tail -f logs/entrenamiento_Pentabot-Flat-SinEncoder.log

# Ver una política en el visor
python scripts/play.py Pentabot-Flat-SinEncoder --checkpoint_file=<ruta/model_N.pt>

# Manejarla con el mando de PS5 (por ahora jugar_mando.py usa la tarea Pentabot-Flat;
# para la sin encoders hay que añadirla al diccionario ROBOTS del script)
python jugar_mando.py <ckpt | ultimo> --robot pentabot

# Solo el modelo, sin RL
python ~/pentabot/scripts/ver.py
```

---

## 7. Cómo saber si la política sin encoders es buena

```bash
python ~/pentabot/mjlab/scripts/evaluar_sin_encoder.py ultimo --envs 256
```

Evalúa con toda la aleatorización activa y el IMU con ruido. Criterios propuestos para aceptarla:

| Métrica | Objetivo |
|---|---|
| `sigue` (velocidad real / pedida) | ≥ 70 % en avance, lateral y giro |
| `caidas` | < 2 % |
| `<3 pies` (tiempo con menos de 3 apoyos) | < 5 % |
| `par p95` | < 85 % del límite |
| `incl` (inclinación media) | < 5° |

Si no se cumplen, ajustar las recompensas y la aleatorización de `sin_encoder.py`, y volver a entrenar.

---

## 8. Pendiente: llevarla a la ESP32

1. Exportar los pesos y el normalizador de `policy.onnx` a un `.h` (`const float[]` en flash).
2. Escribir la MLP en C plano (matmul + ELU), sin TFLite. Se estima 1–3 ms por paso frente a los 20 ms disponibles.
3. Bucle a 50 Hz:
   1. Leer el IMU.
   2. Construir las 130 entradas **en el orden de la sección 5**.
   3. Ejecutar la red.
   4. Calcular `0.3 × acción` y convertirlo a µs para cada servo.
4. Leer el DualSense por Bluetooth con la librería `ps5Controller`.
5. Calibrar cada servo: µs en q = 0, sentido de giro y µs/rad (~640 en el MG995).
6. Comprobar que los ejes del IMU coinciden con el sitio `imu` del modelo (x hacia el frente, z hacia arriba).
7. **Antes de probar en el robot:** con las mismas entradas, el `.h` tiene que dar las mismas acciones que el `.pt`.

### Preguntas abiertas (hay que decidirlas antes de la fase 3)
1. ¿Qué IMU lleva: **BNO085** (da la orientación directamente) o **MPU6050** (necesita un filtro Madgwick o Mahony)?
2. ¿Los servos van por el **PCA9685** o directos a la ESP32 por LEDC?
3. ¿Cuánto pesa el robot real? El modelo asume 2.47 kg.

### Riesgos conocidos
- Con 3 patas apoyadas el MG995 ya trabaja al 64 % de su par, así que solo es viable una marcha **lenta**.
- Los parámetros del servo (kp, fricción, inercia) son estimados. Lo ideal es modificar un servo, sacando
  el cable del potenciómetro, para medir su respuesta real y ajustar `pentabot_constants.py`.
- 15 MG995 pueden pedir más de 10 A en picos: hace falta una **fuente de 6 V aparte** de la ESP32,
  con la masa común.

---

## 9. Problemas típicos

| Síntoma | Causa / solución |
|---|---|
| `ImportError` o versión de Python rara al lanzar | `unset PYTHONPATH` (ROS lo contamina) |
| `KeyError: Pentabot-Flat...` | faltan los enlaces de la sección 3 o el registro en `robots/__init__.py` |
| `AssertionError` en `pentabot_constants.py` | el enlace `Pentabot_MuJoCo` no apunta a la raíz de este repo, que debe contener `model/pentabot.xml` |
| Visor nativo no abre | añadir `--viewer viser` y abrir en el navegador |
| Renders sin ventana fallan | `export MUJOCO_GL=egl` |
| Sin memoria en la GPU | bajar `num_envs`, por ejemplo a 2048 |

---

## 10. Convenciones

- Código y documentación en **español**; los nombres internos de mjlab se dejan en inglés.
- Patas `L1`…`L5`, articulaciones `Lk_hip_yaw`, `Lk_hip_pitch`, `Lk_knee`; pies `Lk_foot` (sitio) y `Lk_foot_collision` (geometría).
- Las tareas nuevas van en `mjlab/tarea/` y se registran en su `__init__.py`.
- No subir `logs/` ni `.venv/`. Las políticas buenas se copian a mano a `politica/`.
