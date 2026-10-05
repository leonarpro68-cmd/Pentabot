# Pendiente: Pentabot al robot real (ESP32 + MG995 sin encoder)

Aplazado el 2026-10-01. Al empezar la sesión: ver el `README.md` del repo.

## Dónde estamos

- Tarea de simulación `Pentabot-Flat` funcionando y entrenada (3000 iteraciones, ~3 h, sin caídas).
  - Modelo final: `logs/rsl_rl/pentabot_velocity/2026-09-30_21-00-04/model_3000.pt` (+ `policy.onnx`)
  - Al final sigue la velocidad pedida en torno a un 77 %; la marcha de onda se cumple a medias (0.37 de 0.5).
- Archivos (sin commit todavía):
  - `src/assets/robots/pentabot/pentabot_constants.py`: robot, servos MG995 y retardo del PCA9685
  - `src/tasks/velocity/config/pentabot/`: tarea, recompensas y PPO
  - `mjlab/scripts/entrenar_pentabot.sh`, `mjlab/scripts/ver_pentabot.sh`
  - `jugar_mando.py`: nueva opción `--robot pentabot`
  - `src/assets/robots/__init__.py`: registro del robot

Comandos útiles:
```bash
cd ~/unitree_rl_mjlab
unset PYTHONPATH; source ~/miniconda3/etc/profile.d/conda.sh; conda activate unitree_rl_mjlab
python jugar_mando.py ultimo --robot pentabot     # manejarlo con el mando de PS5
~/pentabot/mjlab/scripts/ver_pentabot.sh ultimo                       # ver la política
~/pentabot/mjlab/scripts/entrenar_pentabot.sh 4096                    # volver a entrenar
```

## El problema

La política actual **no sirve en el robot real**. Sus 56 entradas son: `base_ang_vel`(3), `projected_gravity`(3), `command`(3), `phase`(2), `joint_pos`(15), `joint_vel`(15) y `actions`(15). Los MG995 no tienen realimentación, así que las 30 entradas de `joint_pos`/`joint_vel` no existen en el robot real.

## Tarea 1: volver a entrenar sin encoders (simulación): HECHA el 2026-10-05

Crear una variante `Pentabot-Flat-SinEncoder`:
- [x] **Actor:** quitar `joint_pos` y `joint_vel`. Dejar el IMU (gyro y gravedad), el comando, la fase y la última acción, con **historial de unos 5 pasos**.
- [x] **Crítico:** mantener toda la información (posiciones, velocidades y contactos). Solo se usa al entrenar.
- [x] **Más aleatorización del servo:** retardo, `kp` entre 0.6 y 1.4, juego de engranajes (ruido en el objetivo) y par máximo ±20 %.
- [x] Entrenar (~3 h) y probarlo con `jugar_mando.py`.

Resultado: `politica/sin_encoder/`. Sigue el 83–89 % de la velocidad pedida y no se cae en condiciones de robot real. Ver [`COMPARATIVA.md`](../COMPARATIVA.md).

## Tarea 2: ejecutar la política en la ESP32 DevKit (sin Raspberry Pi)

La red (entrada → 256 → 128 → 128 → 15, ELU) tiene unos 85 000 parámetros, unos 340 KB en float32. Cabe en la flash y debería tardar unos 1–3 ms por paso, frente a los 20 ms disponibles a 50 Hz (estimado, sin medir).

- [ ] Exportar los pesos y la normalización de las entradas desde el `.pt` a un `.h` (`const float[]`, en flash).
- [ ] Escribir la red en C plano (multiplicaciones de matriz + ELU), sin TFLite.
- [ ] Bucle a 50 Hz: leer el IMU, construir las entradas en **el mismo orden que en el simulador**, ejecutar la red, calcular `objetivo = 0.3 × acción` [rad] y enviar los pulsos a los servos.
- [ ] Mando DualSense por Bluetooth (librería `ps5Controller` para ESP32).
- [ ] Archivo de calibración por servo: pulso en µs para q = 0, sentido y µs/rad (unos 640 en un MG995).
- [ ] Comprobar que los ejes del IMU coinciden con el sitio `imu` del modelo.
- [ ] Antes de usarlo en el robot, validar que el `.h` da las mismas acciones que el `.pt` con las mismas entradas.

## Preguntas a resolver al empezar

1. ¿Qué IMU lleva el robot: **BNO085** (da la orientación directamente) o **MPU6050** (necesita un filtro Madgwick o Mahony)?
2. ¿Los servos van por el **PCA9685** o directos a la ESP32 (por LEDC, que tiene 16 canales)?
3. ¿Cuánto pesa el robot real? La electrónica está estimada en 0.18 kg en `docs/MODELO.md`.

## Riesgos conocidos

- Con 3 patas apoyadas el MG995 ya trabaja al 64 % de su par máximo, así que solo es viable una marcha lenta.
- Los parámetros del servo (`kp`, inercia, fricción) son estimados. Lo ideal es modificar **un** servo, sacando un cable del potenciómetro, para medir su respuesta real y ajustar el modelo.
- 15 MG995 pueden pedir más de 10 A en picos: necesitan una fuente de 6 V aparte de la ESP32.
